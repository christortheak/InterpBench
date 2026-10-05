import Foundation
import SteeringKit

/// Read-only results access scoped to an explicit workspace.
public struct StudyResultRepository: Sendable {
    public let workspaceRoot: URL

    public init(workspaceRoot: URL) {
        self.workspaceRoot = workspaceRoot
    }

    private var runsDirectory: URL {
        workspaceRoot.appending(component: "runs")
    }

    public func list(experimentName: String) -> [StudyRunListItem] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }

        return entries.compactMap { url in
            guard
                let data = try? Data(contentsOf: url.appending(component: "experiment.json")),
                let manifest = try? JSONDecoder().decode(ExperimentManifest.self, from: data),
                manifest.name == experimentName
            else { return nil }
            let name = url.lastPathComponent
            let kind: StudyRunListItem.Kind =
                name.contains("-run") ? .run
                : name.contains("-validate") ? .validate
                : name.contains("-evaluate") ? .evaluate
                : .other
            let generationsURL = url.appending(component: "generations.jsonl")
            return StudyRunListItem(
                directoryName: name,
                path: url.path,
                kind: kind,
                createdAt: String(name.prefix(24)),
                generationCount: StudyRecordFile.recordCount(at: generationsURL),
                hasReport: fm.fileExists(atPath: url.appending(component: "report.json").path))
        }
        .filter { $0.kind != .evaluate }
        .sorted { lhs, rhs in lhs.directoryName > rhs.directoryName }
    }

    /// How many of a file's first lines the detail's bounded previews read.
    /// These bound a PREVIEW (the browser client's cards and the short list
    /// under the review button), never the review: the review sheet pages
    /// through every record (`StudyRecordReview`), and the counts beside
    /// the previews are the file's own.
    public static let previewResponseLimit = 80
    public static let previewJudgmentLimit = 200

    public func detail(for item: StudyRunListItem) -> StudyRunDetail {
        let url = URL(filePath: item.path)
        let judgeArtifactURL =
            item.kind == .run
            ? latestEvaluationDirectory(forSourceRun: item.path)
            : (item.kind == .evaluate ? url : nil)
        let responses = StudyRecordFile(url: Self.responsesURL(runDirectory: url))
        let judgments = StudyRecordFile(
            url: Self.judgmentsURL(directory: judgeArtifactURL ?? url))
        var detail = StudyRunDetail(
            item: item,
            judgeArtifactDirectory: judgeArtifactURL?.path,
            report: loadRunReport(url.appending(component: "report.json")),
            validationReportText: loadValidationReport(url.appending(component: "report.json")),
            pairedJudgeReport: loadPairedJudgeReport(
                (judgeArtifactURL ?? url).appending(component: "judge-report.json")),
            robustnessReports: loadRobustnessReports(url.appending(component: "robustness-report.json")),
            generations: previewGenerations(responses),
            judgments: previewJudgments(judgments))
        detail.responseRecordCount = responses.count
        detail.judgmentRecordCount = judgments.count
        if item.kind == .run {
            detail.unfinishedEvaluations = unfinishedEvaluations(
                forSourceRun: item.directoryName, excluding: judgeArtifactURL)
        }
        return detail
    }

    public static func responsesURL(runDirectory: URL) -> URL {
        runDirectory.appending(component: "generations.jsonl")
    }

    public static func judgmentsURL(directory: URL) -> URL {
        directory.appending(component: "judgments.jsonl")
    }

    private func loadRobustnessReports(_ url: URL) -> [String: VariantRobustnessReport] {
        guard let data = try? Data(contentsOf: url),
            let reports = try? JSONDecoder().decode([String: VariantRobustnessReport].self, from: data)
        else { return [:] }
        return reports
    }

    private struct RawReport: Decodable {
        struct Condition: Decodable {
            let generations: Int
            let meanWordCount: Float
            let meanDistinct2: Float
            let meanMarkerDensity: [String: Float]
        }
        let experiment: String
        let promptCount: Int?
        let conditionCount: Int?
        let seedCount: Int?
        let taskPromptsFile: String?
        let conditions: [String: Condition]?
    }

    private func loadRunReport(_ url: URL) -> StudyRunReportView? {
        guard let data = try? Data(contentsOf: url),
            let raw = try? JSONDecoder().decode(RawReport.self, from: data),
            let conditions = raw.conditions
        else { return nil }
        return StudyRunReportView(
            experiment: raw.experiment,
            promptCount: raw.promptCount,
            conditionCount: raw.conditionCount,
            seedCount: raw.seedCount,
            taskPromptsFile: raw.taskPromptsFile,
            conditions: conditions.map { name, condition in
                StudyRunReportView.Condition(
                    name: name,
                    generations: condition.generations,
                    meanWordCount: condition.meanWordCount,
                    meanDistinct2: condition.meanDistinct2,
                    meanMarkerDensity: condition.meanMarkerDensity)
            }.sorted { $0.name < $1.name })
    }

    private func loadValidationReport(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            raw["validation"] != nil
        else { return nil }
        let pretty = (try? JSONSerialization.data(withJSONObject: raw, options: [.prettyPrinted, .sortedKeys]))
            ?? data
        return String(decoding: pretty, as: UTF8.self)
    }

    /// Either engine's judge-report.json (`StudyJudgeReportReader`): a
    /// strict decoder for one engine's keys read the other's as "no report",
    /// and the judged section went missing for server and cluster runs.
    private func loadPairedJudgeReport(_ url: URL) -> PairedJudgeReportView? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return StudyJudgeReportReader.read(data)
    }

    /// Whether a judge report is about the run at `sourceRunPath`.
    ///
    /// A Mac-engine report stores the run's full path; a Python-engine
    /// report stores its directory name. Names are compared as well as
    /// paths because both directories sit in this workspace's `runs/`,
    /// where a run's name is unique — and a path stored before the
    /// workspace was moved or copied no longer equals anything.
    static func report(
        _ report: PairedJudgeReportView, judgesRunAt sourceRunPath: String
    ) -> Bool {
        if report.sourceRunDirectory == sourceRunPath { return true }
        let name = URL(filePath: sourceRunPath).lastPathComponent
        return !name.isEmpty && StudyJudgeReportReader.sourceRunName(report) == name
    }

    private func evaluationDirectories() -> [URL] {
        let entries =
            (try? FileManager.default.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        return entries
            .filter { $0.lastPathComponent.contains("-evaluate") }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    private func latestEvaluationDirectory(forSourceRun sourceRunPath: String) -> URL? {
        evaluationDirectories().first { url in
            guard
                let report = loadPairedJudgeReport(
                    url.appending(component: "judge-report.json"))
            else { return false }
            return Self.report(report, judgesRunAt: sourceRunPath)
        }
    }

    /// Judge evaluations of a run that stopped before writing a report.
    ///
    /// Both engines stamp the source run's name in the evaluate directory's
    /// run-status.json and write judge-report.json only on completion, so a
    /// stopped evaluation is exactly: an evaluate directory with no report
    /// whose status names this run and is not `completed`. Its rows were
    /// kept on purpose; a view that lists only finished evaluations made
    /// them — and the fact that an evaluation stopped — invisible.
    private func unfinishedEvaluations(
        forSourceRun runName: String, excluding finished: URL?
    ) -> [StudyUnfinishedEvaluation] {
        evaluationDirectories().compactMap { url in
            guard url.path != finished?.path,
                !FileManager.default.fileExists(
                    atPath: url.appending(component: "judge-report.json").path),
                Self.statusSourceRun(at: url) == runName,
                let summary = Self.unfinishedSummary(RunStatusFile.reading(at: url))
            else { return nil }
            return StudyUnfinishedEvaluation(
                directoryName: url.lastPathComponent,
                path: url.path,
                summary: summary,
                judgmentRecordCount: StudyRecordFile.recordCount(
                    at: Self.judgmentsURL(directory: url)))
        }
    }

    /// The `sourceRun` a stage stamped in its status file — a key both
    /// engines write and `RunStatusFile.Status` does not carry.
    static func statusSourceRun(at directory: URL) -> String? {
        guard
            let data = try? Data(
                contentsOf: directory.appending(component: RunStatusFile.filename)),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object["sourceRun"] as? String
    }

    /// What happened to an evaluation that has no report, in plain words —
    /// nil when its status says it completed, or when it has no status file
    /// (a directory that predates the status contract, or one still
    /// waiting for judgments made elsewhere).
    static func unfinishedSummary(_ reading: RunStatusFile.Reading) -> String? {
        switch reading {
        case .absent:
            return nil
        case .unreadable:
            return "Its status file cannot be read, so it is treated as "
                + "unfinished."
        case .present(let status):
            var parts: [String]
            switch status.status ?? "" {
            case "completed":
                return nil
            case "checkpointed":
                parts = ["Paused at a checkpoint; it can be resumed."]
            case "inProgress":
                parts = ["Still running, or stopped without recording why."]
            default:
                if status.errorType == "Cancelled" {
                    parts = ["Cancelled before it finished."]
                } else if let error = status.error, !error.isEmpty {
                    parts = ["Stopped with an error: \(error)"]
                } else {
                    parts = ["Stopped before it finished."]
                }
            }
            if let pending = status.pendingUnits, !pending.isEmpty {
                parts.append(
                    "Judges that did not finish: "
                        + pending.joined(separator: ", ") + ".")
            }
            return parts.joined(separator: " ")
        }
    }

    private struct RawGeneration: Decodable {
        let interventionDecisions: JSONValue?
        let probeMeasurements: JSONValue?
        let sampleIndex: Int?
        let condition: String
        let promptID: String
        let prompt: String
        let output: String
        let wordCount: Int
        let distinct2: Float
        let markerDensity: [String: Float]?
    }

    /// The bounded preview only — see `StudyRunDetail.generations`. Lines
    /// that are not generated responses are not in it; they are in the
    /// review, and in `responseRecordCount`.
    private func previewGenerations(_ file: StudyRecordFile) -> [StudyGenerationPreview] {
        file.lines(at: 0..<min(Self.previewResponseLimit, file.count)).compactMap { line in
            guard let raw = try? JSONDecoder().decode(RawGeneration.self, from: line)
            else { return nil }
            let limit = 1_800
            let truncated = raw.output.count > limit
            let output = truncated ? String(raw.output.prefix(limit)) : raw.output
            return StudyGenerationPreview(
                interventionDecisions: raw.interventionDecisions,
                probeMeasurements: raw.probeMeasurements,
                sampleIndex: raw.sampleIndex,
                condition: raw.condition,
                promptID: raw.promptID,
                prompt: raw.prompt,
                output: output,
                wordCount: raw.wordCount,
                distinct2: raw.distinct2,
                markerDensity: raw.markerDensity ?? [:],
                truncated: truncated)
        }
    }

    private struct RawJudgment: Decodable {
        let condition: String
        /// New rows carry the pair cell + both sides' seeds; legacy rows
        /// (pre sample-cell join) carried a single "seed". All optional so
        /// both generations of judgment files stay viewable.
        let sampleIndex: UInt64?
        let baselineSeed: UInt64?
        let variantSeed: UInt64?
        let seed: UInt64?
        let promptID: String
        let prompt: String
        let baselineWas: String
        let conditionWas: String
        let judgment: PairedJudgeResponse
        let conditionResult: String
    }

    /// The bounded preview only — see `StudyRunDetail.judgments`. It holds
    /// Mac-engine verdict rows; noncompliant rows and Python-engine rows are
    /// in the review (`StudyJudgmentRow`), and in `judgmentRecordCount`.
    private func previewJudgments(_ file: StudyRecordFile) -> [StudyJudgePreview] {
        file.lines(at: 0..<min(Self.previewJudgmentLimit, file.count)).compactMap { data in
            guard let raw = try? JSONDecoder().decode(RawJudgment.self, from: data) else {
                return nil
            }
            let pretty =
                (try? JSONSerialization.jsonObject(with: data))
                .flatMap {
                    try? JSONSerialization.data(
                        withJSONObject: $0, options: [.prettyPrinted, .sortedKeys])
                }
                .map { String(decoding: $0, as: UTF8.self) }
                ?? String(decoding: data, as: UTF8.self)
            return StudyJudgePreview(
                condition: raw.condition,
                // Legacy rows keyed by seed: reuse it as the row id's cell
                // so multi-seed legacy files keep distinct preview rows.
                sampleIndex: raw.sampleIndex ?? raw.seed ?? 0,
                baselineSeed: raw.baselineSeed,
                variantSeed: raw.variantSeed,
                promptID: raw.promptID,
                prompt: raw.prompt,
                baselineWas: raw.baselineWas,
                conditionWas: raw.conditionWas,
                winner: raw.judgment.winner,
                conditionResult: raw.conditionResult,
                confidence: raw.judgment.confidence,
                briefReason: raw.judgment.briefReason,
                aScores: raw.judgment.aScores,
                bScores: raw.judgment.bScores,
                structuredFields: raw.judgment.structuredFields,
                rawJSON: pretty)
        }
    }
}
