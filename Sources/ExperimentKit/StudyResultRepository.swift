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
                generationCount: lineCount(generationsURL),
                hasReport: fm.fileExists(atPath: url.appending(component: "report.json").path))
        }
        .filter { $0.kind != .evaluate }
        .sorted { lhs, rhs in lhs.directoryName > rhs.directoryName }
    }

    public func detail(for item: StudyRunListItem) -> StudyRunDetail {
        let url = URL(filePath: item.path)
        let judgeArtifactURL =
            item.kind == .run
            ? latestEvaluationDirectory(forSourceRun: item.path)
            : (item.kind == .evaluate ? url : nil)
        return StudyRunDetail(
            item: item,
            judgeArtifactDirectory: judgeArtifactURL?.path,
            report: loadRunReport(url.appending(component: "report.json")),
            validationReportText: loadValidationReport(url.appending(component: "report.json")),
            pairedJudgeReport: loadPairedJudgeReport(
                (judgeArtifactURL ?? url).appending(component: "judge-report.json")),
            robustnessReports: loadRobustnessReports(url.appending(component: "robustness-report.json")),
            generations: loadGenerations(url.appending(component: "generations.jsonl")),
            judgments: loadJudgments((judgeArtifactURL ?? url).appending(component: "judgments.jsonl")))
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

    private struct RawJudgeReport: Decodable {
        struct Condition: Decodable {
            let pairs: Int
            let conditionWins: Int
            let baselineWins: Int
            let ties: Int
            let meanConfidence: Double
            let structuredSummaries: [String: StructuredFieldSummaryView]?

            enum CodingKeys: String, CodingKey {
                case pairs
                case conditionWins
                case baselineWins
                case ties
                case meanConfidence
                case structuredSummaries = "structuredSummaries"
            }
        }

        let sourceRunDirectory: String
        let judgeModel: String
        let conditions: [String: Condition]
    }

    private func loadPairedJudgeReport(_ url: URL) -> PairedJudgeReportView? {
        guard let data = try? Data(contentsOf: url),
            let raw = try? JSONDecoder().decode(RawJudgeReport.self, from: data)
        else { return nil }
        return PairedJudgeReportView(
            sourceRunDirectory: raw.sourceRunDirectory,
            judgeModel: raw.judgeModel,
            conditions: raw.conditions.map { name, condition in
                PairedJudgeReportView.Condition(
                    name: name,
                    pairs: condition.pairs,
                    conditionWins: condition.conditionWins,
                    baselineWins: condition.baselineWins,
                    ties: condition.ties,
                    meanConfidence: condition.meanConfidence,
                    structuredSummaries: condition.structuredSummaries ?? [:])
            }.sorted { $0.name < $1.name })
    }

    private func latestEvaluationDirectory(forSourceRun sourceRunPath: String) -> URL? {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey])
        else { return nil }

        return entries
            .filter { $0.lastPathComponent.contains("-evaluate") }
            .filter { url in
                guard
                    let data = try? Data(contentsOf: url.appending(component: "judge-report.json")),
                    let raw = try? JSONDecoder().decode(RawJudgeReport.self, from: data)
                else { return false }
                return raw.sourceRunDirectory == sourceRunPath
            }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .first
    }

    private struct RawGeneration: Decodable {
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

    private func loadGenerations(_ url: URL) -> [StudyGenerationPreview] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").prefix(80).compactMap { line in
            guard
                let raw = try? JSONDecoder().decode(
                    RawGeneration.self, from: Data(line.utf8))
            else { return nil }
            let limit = 1_800
            let truncated = raw.output.count > limit
            let output = truncated ? String(raw.output.prefix(limit)) : raw.output
            return StudyGenerationPreview(
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

    private func loadJudgments(_ url: URL) -> [StudyJudgePreview] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").prefix(200).compactMap { line in
            let data = Data(line.utf8)
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
                ?? String(line)
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

    private func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(separator: "\n").count
    }
}
