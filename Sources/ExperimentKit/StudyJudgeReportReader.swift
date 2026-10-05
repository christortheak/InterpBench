import Foundation

/// Reads a `judge-report.json` written by EITHER engine into the one shape
/// the Results view shows (`PairedJudgeReportView`).
///
/// The two engines record the same evaluation under different keys:
///
/// | fact                   | Mac engine              | Python engine              |
/// |------------------------|-------------------------|----------------------------|
/// | the run judged         | `sourceRunDirectory`    | `sourceRun` (a name)       |
/// | the non-baseline wins  | `conditionWins`         | `variantWins`              |
/// | verdicts in a tally    | `pairs`                 | `n`                        |
/// | per-judge tallies      | none (one panel tally)  | `judges[].conditions`      |
/// | judge agreement        | `judgeAgreement[]` with | `agreement[]` with         |
/// |                        | `judgeA`/`judgeB`/`items` | `judges: [a, b]`/`n`     |
/// | human agreement count  | `items`                 | `n`                        |
///
/// A strict decoder written against one key set reads the other as "no
/// report", which is how the judged section went missing for every server
/// and cluster run. This reader is lenient on purpose and changes nothing
/// about what either engine writes. It reads; it never recomputes — a
/// value an engine does not store stays nil.
public enum StudyJudgeReportReader {

    /// nil when the bytes are not a judge report at all (not a JSON
    /// object, or no per-condition tally under `conditions`).
    public static func read(_ data: Data) -> PairedJudgeReportView? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let report = object as? [String: Any],
            let conditionBlocks = report["conditions"] as? [String: Any]
        else { return nil }

        let rawJudges = report["judges"] as? [Any] ?? []
        let judgeBlocks = rawJudges.compactMap { $0 as? [String: Any] }
            .compactMap(judgeBlock)
        // The Mac engine's `judges` is a list of bare panel names.
        let bareNames = rawJudges.compactMap { $0 as? String }
        let judgeNames = bareNames.isEmpty ? judgeBlocks.map(\.name) : bareNames

        let macSource = report["sourceRunDirectory"] as? String
        let pythonSource = report["sourceRun"] as? String
        let dialect: PairedJudgeReportView.Dialect?
        if macSource != nil {
            dialect = .macEngine
        } else if pythonSource != nil || !judgeBlocks.isEmpty {
            dialect = .pythonEngine
        } else {
            dialect = nil
        }

        let tallies = conditions(from: conditionBlocks)
        // Neither engine's marks and no judge tally: some other JSON that
        // happens to have a `conditions` object (a run report has one).
        if dialect == nil, tallies.isEmpty { return nil }

        var view = PairedJudgeReportView(
            sourceRunDirectory: macSource ?? pythonSource ?? "",
            judgeModel: judgeModel(report: report, blocks: judgeBlocks),
            conditions: tallies)
        view.dialect = dialect
        view.judgeNames = judgeNames
        view.judgeBlocks = judgeBlocks
        view.judgeAgreement =
            ((report["judgeAgreement"] ?? report["agreement"]) as? [Any] ?? [])
            .compactMap { $0 as? [String: Any] }
            .compactMap(judgeAgreement)
        if let human = report["humanAgreement"] as? [Any] {
            view.humanAgreement = human.compactMap { $0 as? [String: Any] }
                .compactMap(humanAgreement)
        }
        view.noncompliantJudgments = int(report["noncompliantJudgments"])
        if bool(report["epochUnverified"]) == true {
            view.epochUnverified = true
        }
        if let drift = report["measurementDrift"] as? String, !drift.isEmpty {
            view.measurementDrift = drift
        }
        if let exclusions = report["exclusions"] as? [String: Any] {
            view.excludedRecords = int(exclusions["excludedRecords"])
        }
        if let sessions = report["judgingSessions"] as? [String: Any] {
            view.judgingSessions = PairedJudgeReportView.JudgingSessions(
                resumedFrom: sessions["resumedFrom"] as? String,
                reusedJudgments: int(sessions["reusedJudgments"]) ?? 0,
                freshJudgments: int(sessions["freshJudgments"]) ?? 0)
        }
        return view
    }

    /// The name of the run a report judges, whichever way the engine wrote
    /// it — what links a judge directory to its run when the full path a
    /// Mac-engine report stores no longer matches (a moved or copied
    /// workspace) or was never a path (a Python-engine report).
    public static func sourceRunName(_ report: PairedJudgeReportView) -> String {
        URL(filePath: report.sourceRunDirectory).lastPathComponent
    }

    // MARK: - Pieces

    private static func conditions(
        from blocks: [String: Any]
    ) -> [PairedJudgeReportView.Condition] {
        blocks.compactMap { name, raw -> PairedJudgeReportView.Condition? in
            guard let block = raw as? [String: Any] else { return nil }
            return condition(name: name, block: block)
        }
        .sorted { $0.name < $1.name }
    }

    private static func condition(
        name: String, block: [String: Any]
    ) -> PairedJudgeReportView.Condition? {
        guard
            let wins = int(block["conditionWins"]) ?? int(block["variantWins"]),
            let baselineWins = int(block["baselineWins"]),
            let ties = int(block["ties"])
        else { return nil }
        var summaries: [String: StructuredFieldSummaryView] = [:]
        if let raw = block["structuredSummaries"] as? [String: Any],
            let data = try? JSONSerialization.data(withJSONObject: raw),
            let decoded = try? JSONDecoder().decode(
                [String: StructuredFieldSummaryView].self, from: data)
        {
            summaries = decoded
        }
        return PairedJudgeReportView.Condition(
            name: name,
            // Both engines count verdicts here, under different keys. A
            // tally with neither key still has its three counts, and a
            // tally of verdicts is their sum by definition.
            pairs: int(block["pairs"]) ?? int(block["n"])
                ?? (wins + baselineWins + ties),
            conditionWins: wins,
            baselineWins: baselineWins,
            ties: ties,
            meanConfidence: double(block["meanConfidence"]),
            structuredSummaries: summaries)
    }

    private static func judgeBlock(
        _ block: [String: Any]
    ) -> PairedJudgeReportView.JudgeBlock? {
        guard let name = block["name"] as? String else { return nil }
        return PairedJudgeReportView.JudgeBlock(
            name: name,
            requestedModel: block["requestedModel"] as? String,
            actualModel: block["actualModel"] as? String,
            pairs: int(block["pairs"]),
            conditions: conditions(
                from: block["conditions"] as? [String: Any] ?? [:]),
            noncompliantJudgments: int(block["noncompliantJudgments"]),
            salvagedVerdicts: int(block["salvagedVerdicts"]))
    }

    private static func judgeModel(
        report: [String: Any], blocks: [PairedJudgeReportView.JudgeBlock]
    ) -> String {
        // A Python-engine report's top-level `judgeModel` is the FIRST
        // judge's only; the blocks name every judge's model.
        let fromBlocks = blocks.compactMap(\.requestedModel)
        if !fromBlocks.isEmpty { return fromBlocks.joined(separator: ", ") }
        return report["judgeModel"] as? String ?? ""
    }

    private static func judgeAgreement(
        _ entry: [String: Any]
    ) -> PairedJudgeReportView.JudgeAgreement? {
        let pair = (entry["judges"] as? [Any])?.compactMap { $0 as? String }
        guard
            let judgeA = entry["judgeA"] as? String ?? pair?.first,
            let judgeB = entry["judgeB"] as? String
                ?? (pair.flatMap { $0.count > 1 ? $0[1] : nil }),
            let items = int(entry["items"]) ?? int(entry["n"]),
            let percent = double(entry["percentAgreement"])
        else { return nil }
        return PairedJudgeReportView.JudgeAgreement(
            judgeA: judgeA, judgeB: judgeB, items: items,
            percentAgreement: percent, kappa: double(entry["kappa"]))
    }

    private static func humanAgreement(
        _ entry: [String: Any]
    ) -> PairedJudgeReportView.HumanAgreement? {
        guard
            let judge = entry["judge"] as? String,
            let items = int(entry["items"]) ?? int(entry["n"]),
            let percent = double(entry["percentAgreement"])
        else { return nil }
        return PairedJudgeReportView.HumanAgreement(
            judge: judge, items: items, percentAgreement: percent,
            kappa: double(entry["kappa"]))
    }

    // MARK: - Lenient scalars (shared with the row readers)

    private static func int(_ raw: Any?) -> Int? { StudyReviewJSON.int(raw) }
    private static func double(_ raw: Any?) -> Double? { StudyReviewJSON.double(raw) }
    private static func bool(_ raw: Any?) -> Bool? { StudyReviewJSON.bool(raw) }
}

/// One line of the judged section's reliability summary: what the report
/// holds, and one plain sentence saying what it means.
public struct StudyJudgeReliabilityLine: Identifiable, Sendable, Equatable {
    public enum Tone: String, Sendable {
        case plain
        /// Something a reader should weigh before relying on the tallies.
        case caution
    }

    /// Stable per line (a kind, plus the judges it names).
    public let id: String
    public let label: String
    public let value: String
    /// One sentence, for a reader with no machine-learning background.
    public let explanation: String
    public let tone: Tone
}

extension PairedJudgeReportView.Condition {
    /// "condition 12 · baseline 9 · ties 3 · mean confidence 0.82". A
    /// report that stores no mean confidence says so instead of showing a
    /// number.
    public var tallyLine: String {
        let confidence = meanConfidence
            .map { "mean confidence " + String(format: "%.2f", $0) }
            ?? "mean confidence not stored"
        return "condition \(StudyReviewText.grouped(conditionWins))"
            + " · baseline \(StudyReviewText.grouped(baselineWins))"
            + " · ties \(StudyReviewText.grouped(ties))"
            + " · \(confidence)"
    }
}

extension PairedJudgeReportView {

    /// One block of per-condition tallies as the judged section lists them.
    public struct TallyGroup: Identifiable, Sendable, Equatable {
        public let id: String
        /// The judge the tallies belong to; nil for a whole-panel tally.
        public let title: String?
        /// What a reader needs to know to read the counts correctly.
        public let note: String?
        public let conditions: [Condition]
    }

    /// The tallies the report holds, without merging or re-adding them: one
    /// group per judge where the report stores per-judge tallies (the
    /// Python engine), otherwise the report's single panel-wide tally (the
    /// Mac engine) — labelled as a sum when more than one judge is in it.
    public var tallyGroups: [TallyGroup] {
        if !judgeBlocks.isEmpty {
            return judgeBlocks.map { block in
                var note = block.requestedModel
                if let actual = block.actualModel,
                    let requested = block.requestedModel, actual != requested
                {
                    note = "\(requested), answered by \(actual)"
                }
                return TallyGroup(
                    id: "judge:\(block.name)", title: block.name,
                    note: note, conditions: block.conditions)
            }
        }
        return [
            TallyGroup(
                id: "panel", title: nil,
                note: judgeNames.count > 1
                    ? "These counts add up the verdicts of all "
                        + "\(judgeNames.count) judges "
                        + "(\(judgeNames.joined(separator: ", ")))."
                    : nil,
                conditions: conditions)
        ]
    }

    /// The reliability numbers and stamps this report holds, in the order
    /// a reader needs them: how far the judges can be trusted (agreement),
    /// then what is missing or qualified (no-verdict pairs, stamps).
    ///
    /// Display only. Every value is read from the report; a line appears
    /// only for something the report states, with two exceptions that say
    /// so in words: agreement between judges and pairs with no verdict are
    /// always listed, because "the report holds none" is itself what the
    /// reader needs to know about them.
    public var reliabilityLines: [StudyJudgeReliabilityLine] {
        var lines: [StudyJudgeReliabilityLine] = []

        let agreementExplanation =
            "How often these two judges reached the same verdict on the "
            + "pairs both of them judged; kappa is that agreement with the "
            + "part expected from chance removed, so 1 means they always "
            + "agreed and 0 means no more than chance."
        if judgeAgreement.isEmpty {
            let judgeCount = max(judgeNames.count, judgeBlocks.count)
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "agreement",
                    label: "Agreement between judges",
                    value: judgeCount == 1
                        ? "Not available: one judge" : "Not in this report",
                    explanation: judgeCount == 1
                        ? "This evaluation used one judge, so there is no "
                            + "second verdict to compare and the report "
                            + "holds no agreement figure."
                        : "This report stores no comparison between judges, "
                            + "so none is shown.",
                    tone: .plain))
        }
        for entry in judgeAgreement {
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "agreement:\(entry.judgeA):\(entry.judgeB)",
                    label: "Agreement between judges: "
                        + "\(entry.judgeA) and \(entry.judgeB)",
                    value: entry.items > 0
                        ? Self.agreementValue(
                            percent: entry.percentAgreement, items: entry.items,
                            kappa: entry.kappa, noun: "pairs both judged")
                        : "No pair was judged by both",
                    explanation: agreementExplanation,
                    tone: .plain))
        }

        for entry in humanAgreement ?? [] {
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "human:\(entry.judge)",
                    label: "Agreement with human ratings: \(entry.judge)",
                    value: entry.items > 0
                        ? Self.agreementValue(
                            percent: entry.percentAgreement, items: entry.items,
                            kappa: entry.kappa, noun: "rated pairs")
                        : "No pair this judge judged has a human rating",
                    explanation:
                        "How often this judge reached the same verdict as "
                        + "the human ratings attached to the study, on the "
                        + "pairs a person rated; kappa removes the agreement "
                        + "expected from chance.",
                    tone: .plain))
        }

        let noncompliant = noncompliantJudgments ?? 0
        let perJudge = judgeBlocks.compactMap { block -> String? in
            guard let count = block.noncompliantJudgments, count > 0 else {
                return nil
            }
            return "\(block.name) \(StudyReviewText.grouped(count))"
        }
        lines.append(
            StudyJudgeReliabilityLine(
                id: "noncompliant",
                label: "Pairs with no verdict",
                value: noncompliant > 0
                    ? StudyReviewText.grouped(noncompliant)
                        + (perJudge.isEmpty
                            ? "" : " (" + perJudge.joined(separator: ", ") + ")")
                    : "None recorded",
                explanation:
                    "Pairs where a judge answered but gave no usable "
                    + "verdict, even on a second try; each is kept as a row "
                    + "you can review, and none is counted in the wins, "
                    + "ties, or agreement figures.",
                tone: noncompliant > 0 ? .caution : .plain))

        if epochUnverified == true {
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "epochUnverified",
                    label: "Source run not checked against the study",
                    value: "Accepted without the check",
                    explanation:
                        "The run these verdicts describe carries no stamp "
                        + "tying it to this version of the study, and the "
                        + "evaluation was told to go ahead anyway, so the "
                        + "verdicts may describe a different version of the "
                        + "study.",
                    tone: .caution))
        }

        if let drift = measurementDrift {
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "measurementDrift",
                    label: "Study changed after the run",
                    value: drift,
                    explanation:
                        "The study's measurement settings were different "
                        + "when the judges ran than when the responses were "
                        + "generated, and the evaluation went ahead because "
                        + "only measurement settings had changed; the "
                        + "differences are listed as recorded.",
                    tone: .caution))
        }

        if let excluded = excludedRecords {
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "excludedRecords",
                    label: "Responses left out before judging",
                    value: StudyReviewText.grouped(excluded),
                    explanation:
                        "Responses removed by the exclusion rules the study "
                        + "declared in advance, before any judge saw them; "
                        + "a pair that lost a response was not judged.",
                    tone: excluded > 0 ? .caution : .plain))
        }

        let salvaged = judgeBlocks.compactMap { block -> String? in
            guard let count = block.salvagedVerdicts, count > 0 else { return nil }
            return "\(block.name) \(StudyReviewText.grouped(count))"
        }
        if !salvaged.isEmpty {
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "salvagedVerdicts",
                    label: "Verdicts read from cut-off answers",
                    value: salvaged.joined(separator: ", "),
                    explanation:
                        "The judge's answer ran out of room before it "
                        + "finished; the winner was still readable and was "
                        + "kept, and the judge's stated reason may be "
                        + "incomplete.",
                    tone: .caution))
        }

        if let sessions = judgingSessions {
            let source = sessions.resumedFrom.map { " from \($0)" } ?? ""
            lines.append(
                StudyJudgeReliabilityLine(
                    id: "judgingSessions",
                    label: "Judged in more than one sitting",
                    value:
                        "\(StudyReviewText.grouped(sessions.reusedJudgments)) "
                        + "kept\(source), "
                        + "\(StudyReviewText.grouped(sessions.freshJudgments)) new",
                    explanation:
                        "This evaluation was finished by resuming an earlier "
                        + "one that had stopped, so some verdicts are from "
                        + "the earlier sitting, and a judge model run by an "
                        + "outside provider may have changed in between.",
                    tone: .caution))
        }
        return lines
    }

    /// "83% of 24 pairs both judged · kappa 0.61". The percentage is the
    /// report's fraction, shown to the nearest whole percent.
    static func agreementValue(
        percent: Double, items: Int, kappa: Double?, noun: String
    ) -> String {
        let percentText = "\(Int((percent * 100).rounded()))%"
        let kappaText = kappa.map { "kappa " + String(format: "%.2f", $0) }
            ?? "kappa not defined"
        return "\(percentText) of \(StudyReviewText.grouped(items)) \(noun)"
            + " · \(kappaText)"
    }
}
