import Foundation
import Observation

/// The authoring model behind "instantiate this template N times".
///
/// Everything here is decision logic the Studies panel RENDERS: which castings
/// are legal, what a batch will cost before it is minted, and the sequencing
/// and failure isolation of the submissions it produces. None of it is
/// measurement — a minted study is an ordinary draft (see `StudyTemplate`), and
/// nothing in this file is read by a run, a freeze, an analysis or the Python
/// engine.
///
/// It lives in ExperimentKit rather than the view for the ordinary reason: the
/// arithmetic that tells a researcher a click is about to queue 240 transcripts
/// is exactly the kind of thing that must be unit-tested, and a SwiftUI body is
/// not testable.

// MARK: - The cost of a batch, before it is minted

/// What a batch will actually produce, in the units the substrate charges for.
///
/// A composition sweep over a four-seat panel is five castings; at two arms and
/// twenty play-throughs that is two hundred transcripts and, sharded four ways,
/// twenty Slurm jobs. Those three numbers differ by an order of magnitude and
/// the researcher chooses between them by adding one row. Showing only the row
/// count hides the choice being made.
public struct TemplateBatchTotals: Sendable, Equatable {
    /// Rows in the table — one study each.
    public var studies: Int
    /// Run arms summed ACROSS rows. Not uniform for compare-agents: a row's
    /// arms are its agents plus the baseline the run loop always prepends, and
    /// two rows may cast different numbers of agents.
    public var arms: Int
    /// Play-throughs (multi-agent) / samples per item (model output) — the
    /// manifest's `samplesPerItem`, which both engines read as the replicate
    /// count.
    public var playThroughs: Int
    /// Task-prompt rows per arm. 1 for a panel: the scenario IS the item.
    public var items: Int
    /// Parallel jobs one submission fans out into (the Remote options'
    /// stepper). 1 when sharding is off.
    public var shardsPerStudy: Int
    /// "transcripts" for a panel, "generations" for model output.
    public var unitNoun: String
    /// "castings" for a panel, "studies" otherwise — a row of a panel batch is
    /// a casting, and calling it a study buries the thing that varies.
    public var rowNoun: String
    /// "Slurm jobs" / "local jobs" — whatever the active executor is called.
    public var jobNoun: String

    /// Generations (or transcripts) the whole batch will produce.
    public var units: Int { arms * playThroughs * items }
    public var jobs: Int { studies * shardsPerStudy }

    /// The live totals line, e.g.
    /// `6 castings × 2 arms × 20 play-throughs = 240 transcripts · 24 Slurm jobs`.
    ///
    /// Factors that equal 1 are dropped — a batch with one arm and one
    /// play-through should read `3 studies × 40 items = 120 generations`, not
    /// carry two multiplications by one that make the reader hunt for the
    /// number that matters.
    public var summary: String {
        guard studies > 0 else { return "no rows yet — add a casting below" }
        var factors = ["\(studies) \(studies == 1 ? String(rowNoun.dropLast()) : rowNoun)"]
        // Arms are reported as an AVERAGE-free total when they vary across
        // rows: "6 castings × 2 arms" is only honest if every row has two.
        if arms != studies {
            let perRow = arms / studies
            if perRow * studies == arms, perRow > 1 {
                factors.append("\(perRow) arms")
            } else {
                factors.append("\(arms) arms total")
            }
        }
        if playThroughs > 1 { factors.append("\(playThroughs) play-throughs") }
        if items > 1 { factors.append("\(items) items") }
        var line = factors.joined(separator: " × ") + " = \(units) \(unitNoun)"
        line += " · \(jobs) \(jobs == 1 ? String(jobNoun.dropLast()) : jobNoun)"
        return line
    }

    /// Totals for a table of castings against one template.
    ///
    /// `armsPerRow` is supplied by the caller because it is the only quantity
    /// that depends on the casting itself: a panel row always runs the
    /// manifest's arm pair, while a compare-agents row runs one arm per cast
    /// agent plus the baseline the run loop prepends.
    public static func totals(
        template: StudyTemplate,
        armsPerRow: [Int],
        taskItemCount: Int,
        shardsPerStudy: Int,
        jobNoun: String = "jobs"
    ) -> TemplateBatchTotals {
        let panel = template.intent == .multiAgent
        return TemplateBatchTotals(
            studies: armsPerRow.count,
            arms: armsPerRow.reduce(0, +),
            playThroughs: max(1, template.study.samplesPerItem ?? 1),
            items: panel ? 1 : max(1, taskItemCount),
            shardsPerStudy: max(1, shardsPerStudy),
            unitNoun: panel ? "transcripts" : "generations",
            rowNoun: panel ? "castings" : "studies",
            jobNoun: jobNoun)
    }

    /// Arms one row runs, by study type.
    ///
    /// Multi-agent: the manifest's fixed pair (`baseline` + `configured`), or
    /// the single configured arm when the baseline is switched off — castings
    /// never become conditions, so the arm count does not depend on the cast.
    /// Model output: the baseline `runVariantComparison` always prepends, plus
    /// one per cast agent.
    public static func armsInRow(
        template: StudyTemplate, castAgentCount: Int
    ) -> Int {
        template.intent == .multiAgent
            ? (template.study.multiAgentIncludeBaseline ? 2 : 1)
            : 1 + castAgentCount
    }

    /// Rows of a task-prompts file, counted once (never from a SwiftUI body).
    /// Absent or unreadable reads as 1 so the totals line degrades to the row
    /// count instead of claiming zero work.
    public static func taskItemCount(_ template: StudyTemplate) -> Int {
        guard let file = template.study.taskPromptsFile, !file.isEmpty,
            let text = try? String(
                contentsOf: ExperimentStore.resolveProjectPath(file), encoding: .utf8)
        else { return 1 }
        let rows = text.split(whereSeparator: \.isNewline)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        return max(1, rows.count)
    }
}

// MARK: - Minting a batch with per-row isolation

extension StudyTemplateStore {

    /// What one row's mint produced.
    public struct RowMint: Sendable, Equatable {
        public let row: Int
        public let study: String?
        /// The refusal, VERBATIM. `instantiate`'s messages are actionable
        /// ("re-mint the template rather than minting a study whose pins are
        /// already stale") and paraphrasing them costs the researcher the
        /// instruction.
        public let failure: String?

        public init(row: Int, study: String?, failure: String?) {
            self.row = row
            self.study = study
            self.failure = failure
        }
    }

    public struct BatchMint: Sendable {
        public let batchGroup: String
        public let results: [RowMint]
        public var minted: [String] { results.compactMap(\.study) }
        public var failures: [RowMint] { results.filter { $0.failure != nil } }
    }

    /// Mints one draft per casting under a SHARED batch id, isolating each
    /// row's refusal.
    ///
    /// `instantiateBatch` is the headless contract and is right for a CLI: it
    /// maps and rethrows, so a batch either happens or does not. Interactively
    /// that is worse than useless — the throw comes from row k with rows
    /// 0..<k already written to disk, and the researcher is told a message
    /// about one casting while N orphan drafts they cannot see have appeared
    /// in the study list. Isolating per row keeps the shared `batchGroup` (the
    /// only thing tying panel siblings together) and reports exactly which
    /// castings landed and which refused.
    public static func mintBatch(
        templateName: String,
        cells: [Cell],
        names: [String?] = [],
        batchID: String? = nil,
        onRow: ((Int, RowMint) -> Void)? = nil
    ) -> BatchMint {
        let batch = batchID ?? newBatchID()
        var results: [RowMint] = []
        for (index, cell) in cells.enumerated() {
            let requested = index < names.count ? names[index] : nil
            let result: RowMint
            do {
                let manifest = try instantiate(
                    templateName: templateName, cell: cell,
                    studyName: requested, batchGroup: batch)
                result = RowMint(row: index, study: manifest.name, failure: nil)
            } catch {
                result = RowMint(
                    row: index, study: nil,
                    failure: (error as? ExperimentError)?.reason ?? "\(error)")
            }
            results.append(result)
            onRow?(index, result)
        }
        return BatchMint(batchGroup: batch, results: results)
    }
}

// MARK: - Submitting a batch, sequentially

/// Sequencing and failure isolation for "submit every study this batch minted".
///
/// Sequential on purpose: each submission packages a bundle, uploads it and
/// waits on the scheduler, and firing six of those concurrently at a login node
/// is how a researcher gets rate-limited off a shared cluster mid-batch. The
/// contract that matters is that one failure does not stop the rest AND is
/// named afterwards — a batch that reports "4 of 6 submitted" without saying
/// which two are missing leaves the researcher diffing job lists by hand.
public enum StudyBatchSubmission {

    /// Why one study's submission did not happen.
    ///
    /// Its own type rather than `ExperimentError` because a submitter crosses
    /// isolation domains (the sequencer is `@Sendable`, tests inject a fake)
    /// and the message is carried verbatim to the researcher either way.
    public struct Failure: Error, Sendable, Equatable {
        public let reason: String
        public init(reason: String) { self.reason = reason }
    }

    public struct Outcome: Sendable, Equatable {
        public let study: String
        public let jobID: String?
        /// The submission failure, verbatim from the submitting path.
        public let failure: String?
        public var succeeded: Bool { jobID != nil }

        public init(study: String, jobID: String?, failure: String?) {
            self.study = study
            self.jobID = jobID
            self.failure = failure
        }
    }

    /// Submits each study in order, never stopping on a failure.
    ///
    /// `submit` is injected so the sequencing is testable without a server:
    /// the panel passes its own bundle-submit path, tests pass a fake.
    public static func submit(
        studies: [String],
        onProgress: (@MainActor @Sendable (Int, Outcome) -> Void)? = nil,
        submit: @MainActor @Sendable (String) async -> Result<String, StudyBatchSubmission.Failure>
    ) async -> [Outcome] {
        var outcomes: [Outcome] = []
        for (index, study) in studies.enumerated() {
            let outcome: Outcome
            switch await submit(study) {
            case .success(let jobID):
                outcome = Outcome(study: study, jobID: jobID, failure: nil)
            case .failure(let error):
                outcome = Outcome(study: study, jobID: nil, failure: error.reason)
            }
            outcomes.append(outcome)
            await onProgress?(index, outcome)
        }
        return outcomes
    }

    /// The one line a researcher reads after a batch: how many landed, and the
    /// NAME of everything that did not.
    public static func summary(_ outcomes: [Outcome]) -> String {
        guard !outcomes.isEmpty else { return "nothing to submit" }
        let failed = outcomes.filter { !$0.succeeded }
        let landed = outcomes.count - failed.count
        var line = "submitted \(landed) of \(outcomes.count)"
        guard !failed.isEmpty else { return line }
        line += " — failed: "
        line += failed.map { "\($0.study) (\($0.failure ?? "no reason given"))" }
            .joined(separator: "; ")
        return line
    }
}

// MARK: - The one-shot handoff that opens the table

/// "Open the new-studies table on this design", as a value.
///
/// A bare design name was enough while the Templates tab was the only caller.
/// The Seats section's "Create permuted siblings…" is the second one, and it
/// arrives with a casting in hand: the sibling studies it wants are every
/// distinct re-seating of the occupants the study is already running, which the
/// table must hold as rows before the researcher sees it. Carrying the pool on
/// the invitation keeps that a preload of the EXISTING table rather than a
/// second minting path.
public struct TemplateInstantiationInvitation: Sendable, Equatable {
    public var design: String
    /// Occupants to expand into distinct castings, in no particular order.
    /// Empty = the plain "instantiate this design" invitation.
    public var permuting: [SeatOccupant]

    public init(design: String, permuting: [SeatOccupant] = []) {
        self.design = design
        self.permuting = permuting
    }
}

// MARK: - The instantiation table

/// One row of the instantiation table: one study to mint.
///
/// A row carries the casting in the shape the researcher edits it — agent
/// library ids for a comparison, seat → occupant for a panel — and converts to
/// a `StudyTemplateStore.Cell` only at mint time. Keeping the edit shape
/// separate is what lets a half-filled panel row exist (and say what it is
/// missing) rather than being unrepresentable.
public struct TemplateCellRow: Identifiable, Sendable, Equatable {
    public let id: UUID
    /// Compare-agents casting: `ModelVariantRecord.ID` values, in arm order.
    public var agentIDs: [String]
    /// Multi-agent casting: seat id → who sits there. Missing key = uncast.
    public var seating: [String: SeatOccupant]
    /// Study name override. Empty = the store's default
    /// (`<template>-<casting descriptor>`).
    public var name: String
    /// Filled in as the batch runs.
    public var state: State

    public enum State: Sendable, Equatable {
        case pending
        case minted(String)
        case submitted(study: String, jobID: String)
        case failed(String)
    }

    public init(
        id: UUID = UUID(), agentIDs: [String] = [],
        seating: [String: SeatOccupant] = [:], name: String = "",
        state: State = .pending
    ) {
        self.id = id
        self.agentIDs = agentIDs
        self.seating = seating
        self.name = name
        self.state = state
    }
}

/// The instantiation sheet's state: the template, the table, the presets and
/// the totals.
///
/// `@MainActor` and observable because the sheet binds it directly; every
/// non-trivial rule on it is a pure function so the tests can drive it without
/// a view.
@Observable @MainActor
public final class TemplateInstantiation {

    public private(set) var template: StudyTemplate?
    public private(set) var templateName: String
    /// Seats of the template's semantic panel, in panel order. Empty for a
    /// compare-agents template.
    public private(set) var seatIDs: [String] = []
    /// The agent library filtered to the template's base model — the same
    /// eligibility rule `ExperimentStore.attachAgent` enforces, applied BEFORE
    /// the researcher can pick an agent that would refuse.
    public private(set) var agents: [ModelVariantRecord] = []
    /// A load-time refusal (missing template, drifted semantic panel). Shown
    /// verbatim; the table is not offered.
    public private(set) var loadFailure: String?
    /// Non-blocking things the researcher must see (e.g. no eligible agents).
    public private(set) var advisories: [String] = []

    public var rows: [TemplateCellRow] = []
    /// The agent chosen for "Add composition sweep" (nil = baseline-only
    /// sweep, which is a legal but pointless request — the UI disables it).
    public var sweepAgentID: String?
    /// The agent multiset chosen for "Add all permutations".
    public var permutationAgentIDs: [String] = []
    /// Whether the permutation set includes baseline occupants to fill the
    /// remaining seats.
    public var permutationPadsWithBaseline = true

    /// Shards one submission fans out into — mirrored from the panel's Remote
    /// options so the totals line counts the jobs the batch will really make.
    public var shardsPerStudy = 1
    public var jobNoun = "jobs"

    /// Set while a mint/submit batch runs (single-flight).
    public private(set) var isWorking = false
    public private(set) var lastSummary: String?
    public private(set) var lastBatchGroup: String?
    /// Studies the last mint wrote, in TABLE order.
    ///
    /// Order is the contract: "Load Only" opens the first one in the Studies
    /// editor, and the first row is the one the researcher filled in first.
    public private(set) var mintedStudies: [String] = []

    /// The study "Load Only" hands to the Studies editor.
    public var firstMintedStudy: String? { mintedStudies.first }

    /// True when every row that was attempted landed — the condition for the
    /// sheet to close itself rather than stay up with failures to read.
    public private(set) var lastMintWasClean = false

    private var taskItems = 1

    public init(templateName: String) {
        self.templateName = templateName
        load()
    }

    // MARK: Loading

    public func load() {
        loadFailure = nil
        advisories = []
        do {
            let template = try StudyTemplateStore.load(name: templateName)
            self.template = template
            taskItems = TemplateBatchTotals.taskItemCount(template)
            agents = ModelVariantStore.scan().filter {
                $0.artifact.baseModelID == template.study.modelID
            }
            if template.intent == .multiAgent {
                seatIDs = try StudyTemplateStore.semanticSeatIDs(
                    templateName: templateName)
            } else {
                seatIDs = []
            }
            if agents.isEmpty {
                advisories.append(
                    "no saved agents use this template's base model "
                        + "(\(template.study.modelID)) — every casting can only "
                        + "be baseline until one exists")
            }
            if rows.isEmpty { addRow() }
        } catch {
            loadFailure = (error as? ExperimentError)?.reason ?? "\(error)"
        }
    }

    // MARK: Table editing

    public func addRow() {
        var row = TemplateCellRow()
        // A panel row starts fully cast at baseline rather than empty: an
        // all-baseline casting is the control composition, and it is also the
        // only starting point from which every seat picker reads "baseline"
        // instead of "uncast".
        for seat in seatIDs { row.seating[seat] = .baseline }
        rows.append(row)
    }

    public func removeRow(_ id: TemplateCellRow.ID) {
        rows.removeAll { $0.id == id }
    }

    public func clearRows() {
        rows.removeAll()
    }

    /// The occupant a library agent contributes to a seat, pinned by path and
    /// artifact hash exactly as a variant condition is.
    public func occupant(forAgentID id: String?) -> SeatOccupant {
        guard let id, let record = agents.first(where: { $0.id == id }) else {
            return .baseline
        }
        return occupant(for: record)
    }

    public func occupant(for record: ModelVariantRecord) -> SeatOccupant {
        SeatCasting.occupant(for: record)
    }

    // MARK: Presets

    /// All-baseline, each seat solo-treated, all-treated — as rows.
    public func addCompositionSweep() {
        guard let template, template.intent == .multiAgent else { return }
        let agent = occupant(forAgentID: sweepAgentID)
        do {
            let cells = try StudyTemplateStore.compositionSweepCells(
                templateName: templateName, agent: agent)
            append(cells: cells)
        } catch {
            loadFailure = (error as? ExperimentError)?.reason ?? "\(error)"
        }
    }

    /// The occupant multiset "Add all permutations" would expand, padded with
    /// baseline to fill the panel. Exposed so the UI can show the DEDUPED
    /// count before the researcher commits to it.
    public var permutationOccupants: [SeatOccupant] {
        var pool = permutationAgentIDs.map { occupant(forAgentID: $0) }
        if permutationPadsWithBaseline, pool.count < seatIDs.count {
            pool += Array(repeating: .baseline, count: seatIDs.count - pool.count)
        }
        return pool
    }

    /// How many DISTINCT castings "Add all permutations" would add — the
    /// multiset dedupe, computed before minting. `[A, A, baseline]` over three
    /// seats is 3, not 6, and the difference is GPU hours plus a double count
    /// in the analysis.
    public var permutationCount: Int {
        (try? PanelComposition.distinctAssignments(
            seatIDs: seatIDs, occupants: permutationOccupants))?.count ?? 0
    }

    /// Why the permutation preset cannot run right now, or nil.
    public var permutationRefusal: String? {
        guard !seatIDs.isEmpty else { return "this template declares no panel" }
        let pool = permutationOccupants
        guard pool.count == seatIDs.count else {
            return "\(pool.count) agent(s) for \(seatIDs.count) seat(s) — a "
                + "casting fills every seat exactly once"
        }
        return nil
    }

    /// Preloads the table with every DISTINCT casting of `occupants` — what
    /// "Create permuted siblings…" hands over from a study that is already
    /// cast.
    ///
    /// The same expansion the preset performs, reached with the pool supplied
    /// rather than picked, so a permuted batch minted from a study and one
    /// assembled in this sheet are the same rows. A pool that does not fill the
    /// panel is an advisory, not a load failure: the design's scenario may
    /// simply have a different number of seats than the study's, and the table
    /// stays usable.
    public func preloadPermutations(occupants: [SeatOccupant]) {
        guard let template, template.intent == .multiAgent else { return }
        guard !occupants.isEmpty else { return }
        guard occupants.count == seatIDs.count else {
            advisories.append(
                "the study's casting fills \(occupants.count) seat(s) but this "
                    + "design's scenario has \(seatIDs.count) — cast the rows "
                    + "below by hand, or check that the design declares the "
                    + "scenario you meant")
            return
        }
        do {
            let cells = try PanelComposition.distinctAssignments(
                seatIDs: seatIDs, occupants: occupants)
                .map(StudyTemplateStore.Cell.seating)
            append(cells: cells)
            // Leave the preset menu holding the same pool, so pressing "Add all
            // permutations" reproduces what was preloaded instead of an empty
            // set.
            permutationAgentIDs = occupants.compactMap { occupant in
                guard case .agent(_, let path, _) = occupant else { return nil }
                return agents.first {
                    ModelVariantStore.relativePath(for: $0) == path
                }?.id
            }
            permutationPadsWithBaseline = occupants.contains(.baseline)
        } catch {
            advisories.append((error as? ExperimentError)?.reason ?? "\(error)")
        }
    }

    public func addAllPermutations() {
        guard permutationRefusal == nil else { return }
        do {
            let cells = try PanelComposition.distinctAssignments(
                seatIDs: seatIDs, occupants: permutationOccupants)
                .map(StudyTemplateStore.Cell.seating)
            append(cells: cells)
        } catch {
            loadFailure = (error as? ExperimentError)?.reason ?? "\(error)"
        }
    }

    private func append(cells: [StudyTemplateStore.Cell]) {
        // A preset REPLACES a single untouched starter row rather than sitting
        // under it — the starter is an artefact of opening the sheet, and
        // minting an unwanted all-baseline study is a real cost.
        if rows.count == 1, rows[0].isUntouched { rows.removeAll() }
        for cell in cells {
            guard case .seating(let assignment) = cell else { continue }
            rows.append(
                TemplateCellRow(seating: assignment.occupants))
        }
    }

    // MARK: Readiness

    /// Why this row cannot be MINTED, or nil.
    ///
    /// The same conditions `instantiate` refuses on, checked BEFORE the mint so
    /// the table can say which row is the problem — `instantiate`'s message is
    /// about a casting, and in a six-row table that is not enough to act on.
    ///
    /// "No agents cast" is deliberately NOT here (2026-08-06). A zero-agent
    /// draft is a legal study — the run loop's baseline arm exists, the
    /// readiness check surfaces the missing agents, and the researcher may well
    /// want the draft first and the casting after. Refusing to WRITE it made
    /// the sheet a gate on a decision that belongs in the Studies editor; it is
    /// an advisory now, and only submission requires a runnable selection.
    public func refusal(for row: TemplateCellRow) -> String? {
        guard let template else { return loadFailure }
        if template.intent == .multiAgent {
            let uncast = seatIDs.filter { row.seating[$0] == nil }
            guard uncast.isEmpty else {
                return "no agent assigned to seat(s): "
                    + uncast.joined(separator: ", ")
                    + " — every seat runs, so every seat must be cast "
                    + "(use baseline for an unsteered seat)"
            }
            return nil
        }
        for id in row.agentIDs {
            guard let record = agents.first(where: { $0.id == id }) else {
                return "an agent in this row is no longer in the library"
            }
            guard record.artifact.baseModelID == template.study.modelID else {
                return "agent '\(record.artifact.name)' uses "
                    + "\(record.artifact.baseModelID), not this design's base "
                    + "model \(template.study.modelID)"
            }
        }
        return nil
    }

    /// What this row will mint but could not RUN as it stands, or nil.
    ///
    /// Non-blocking by construction: it is the same fact the study's own
    /// readiness check reports once the draft exists, said early.
    public func advisory(for row: TemplateCellRow) -> String? {
        guard let template, template.intent != .multiAgent else { return nil }
        guard row.agentIDs.isEmpty else { return nil }
        return "no agents cast — this mints a legal draft whose only arm is "
            + "the baseline; add agents here or in the study before running"
    }

    /// Enough to WRITE the drafts.
    public var readyToMint: Bool {
        loadFailure == nil && !rows.isEmpty
            && rows.allSatisfy { refusal(for: $0) == nil }
    }

    /// Enough to write them AND queue them: every row has a runnable casting.
    ///
    /// Submitting a baseline-only comparison spends cluster time measuring one
    /// arm against nothing, which is the one case worth stopping at the button
    /// rather than in the queue.
    public var readyToSubmit: Bool {
        readyToMint && rows.allSatisfy { advisory(for: $0) == nil }
    }

    /// The default name a row will mint under, so the table can show it before
    /// it exists. Mirrors `instantiate`'s own derivation.
    public func defaultName(for row: TemplateCellRow) -> String {
        guard let cell = cell(for: row) else { return "" }
        return "\(templateName)-\(cell.nameFragment)"
    }

    // MARK: Cells

    public func cell(for row: TemplateCellRow) -> StudyTemplateStore.Cell? {
        guard let template else { return nil }
        if template.intent == .multiAgent {
            return .seating(
                SeatAssignment(seatIDs: seatIDs, occupants: row.seating))
        }
        let records = row.agentIDs.compactMap { id in
            agents.first(where: { $0.id == id })
        }
        return .agents(records)
    }

    // MARK: Totals

    public var totals: TemplateBatchTotals {
        guard let template else {
            return TemplateBatchTotals(
                studies: 0, arms: 0, playThroughs: 1, items: 1,
                shardsPerStudy: 1, unitNoun: "generations",
                rowNoun: "studies", jobNoun: jobNoun)
        }
        let arms = rows.map {
            TemplateBatchTotals.armsInRow(
                template: template, castAgentCount: $0.agentIDs.count)
        }
        return TemplateBatchTotals.totals(
            template: template, armsPerRow: arms, taskItemCount: taskItems,
            shardsPerStudy: shardsPerStudy, jobNoun: jobNoun)
    }

    // MARK: Mint (and optionally submit)

    /// Mints every row, then — when `submit` is supplied — submits each minted
    /// draft in turn. Returns the summary line.
    ///
    /// Minting and submission are separate decisions on purpose: a batch that
    /// mints drafts is reviewable in the study list before it costs GPU time,
    /// which is why "Mint only" exists and why this takes the submitter as an
    /// argument rather than reaching for one.
    @discardableResult
    public func mint(
        submit: (@MainActor @Sendable (String) async -> Result<String, StudyBatchSubmission.Failure>)? = nil
    ) async -> String {
        guard !isWorking, readyToMint else { return lastSummary ?? "" }
        isWorking = true
        defer { isWorking = false }

        let cells = rows.compactMap { cell(for: $0) }
        let names = rows.map { row -> String? in
            let trimmed = row.name.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        let mint = StudyTemplateStore.mintBatch(
            templateName: templateName, cells: cells, names: names)
        lastBatchGroup = mint.batchGroup
        mintedStudies = mint.minted
        lastMintWasClean = mint.failures.isEmpty
        for result in mint.results where result.row < rows.count {
            if let study = result.study {
                rows[result.row].state = .minted(study)
            } else if let failure = result.failure {
                rows[result.row].state = .failed(failure)
            }
        }

        var summary = "minted \(mint.minted.count) of \(cells.count) "
            + "draft(s) in batch \(mint.batchGroup)"
        for failure in mint.failures {
            summary += " — row \(failure.row + 1): \(failure.failure ?? "")"
        }

        if let submit, !mint.minted.isEmpty {
            let outcomes = await StudyBatchSubmission.submit(
                studies: mint.minted, submit: submit)
            lastMintWasClean = lastMintWasClean && outcomes.allSatisfy(\.succeeded)
            for outcome in outcomes {
                guard let index = rows.firstIndex(where: {
                    if case .minted(let name) = $0.state { return name == outcome.study }
                    return false
                }) else { continue }
                if let jobID = outcome.jobID {
                    rows[index].state = .submitted(study: outcome.study, jobID: jobID)
                } else {
                    rows[index].state = .failed(outcome.failure ?? "submission failed")
                }
            }
            summary += " · " + StudyBatchSubmission.summary(outcomes)
        }
        lastSummary = summary
        return summary
    }
}

extension TemplateCellRow {
    /// True for the row `addRow` creates and nobody has edited — the only row
    /// a preset may discard.
    var isUntouched: Bool {
        agentIDs.isEmpty
            && name.isEmpty
            && seating.values.allSatisfy { $0 == .baseline }
    }
}

extension StudyTemplateStore.Cell {
    /// The name fragment this casting contributes, exposed so the table can
    /// show the auto-name a row WOULD take before minting it.
    public var nameFragment: String { descriptor }
}
