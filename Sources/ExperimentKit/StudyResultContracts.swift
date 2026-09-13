import Foundation
import SteeringKit

public struct StudyRunListItem: Identifiable, Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable {
        case run
        case validate
        case evaluate
        case other
    }

    public var id: String { directoryName }
    public let directoryName: String
    public let path: String
    public let kind: Kind
    public let createdAt: String
    public let generationCount: Int
    public let hasReport: Bool
}

public struct StudyRunDetail: Codable, Sendable, Equatable {
    public let item: StudyRunListItem
    public let judgeArtifactDirectory: String?
    public let report: StudyRunReportView?
    public let validationReportText: String?
    public let pairedJudgeReport: PairedJudgeReportView?
    public let robustnessReports: [String: VariantRobustnessReport]
    public let generations: [StudyGenerationPreview]
    public let judgments: [StudyJudgePreview]
}

public struct LiveStudyGeneration: Codable, Sendable, Equatable {
    public let condition: String
    public let promptID: String
    public let prompt: String
    public let output: String
}

public struct LiveStudyJudgment: Codable, Sendable, Equatable {
    public let condition: String
    public let promptID: String
}

public struct PairedJudgeReportView: Codable, Sendable, Equatable {
    public struct Condition: Codable, Sendable, Equatable {
        public let name: String
        public let pairs: Int
        public let conditionWins: Int
        public let baselineWins: Int
        public let ties: Int
        public let meanConfidence: Double
        public let structuredSummaries: [String: StructuredFieldSummaryView]
    }

    public let sourceRunDirectory: String
    public let judgeModel: String
    public let conditions: [Condition]
}

public struct StructuredFieldSummaryView: Codable, Sendable, Equatable {
    public let count: Int
    public let numericMean: Double?
    public let trueCount: Int?
    public let falseCount: Int?
    public let stringCounts: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case count
        case numericMean = "numeric_mean"
        case trueCount = "true_count"
        case falseCount = "false_count"
        case stringCounts = "string_counts"
    }
}

public struct StudyRunReportView: Codable, Sendable, Equatable {
    public struct Condition: Codable, Sendable, Equatable {
        public let name: String
        public let generations: Int
        public let meanWordCount: Float
        public let meanDistinct2: Float
        public let meanMarkerDensity: [String: Float]
    }

    public let experiment: String
    public let promptCount: Int?
    public let conditionCount: Int?
    public let seedCount: Int?
    public let taskPromptsFile: String?
    public let conditions: [Condition]
}

public struct StudyGenerationPreview: Identifiable, Codable, Sendable, Equatable {
    public var interventionDecisions: JSONValue? = nil
    public var probeMeasurements: JSONValue? = nil
    public var sampleIndex: Int? = nil
    public var id: String { "\(condition)-\(promptID)" + (sampleIndex.map { "-\($0)" } ?? "") }
    public let condition: String
    public let promptID: String
    public let prompt: String
    public let output: String
    public let wordCount: Int
    public let distinct2: Float
    public let markerDensity: [String: Float]
    public let truncated: Bool
}

public struct StudyJudgePreview: Identifiable, Codable, Sendable, Equatable {
    public var id: String { "\(condition)-\(promptID)-\(sampleIndex)" }
    public let condition: String
    /// The pair's sample cell (the cross-engine pairing join key half);
    /// 0 for greedy single-sample runs.
    public let sampleIndex: UInt64
    /// Seed provenance for the two sides of the pair (cross-engine keys);
    /// nil on rows loaded from legacy judgment files.
    public let baselineSeed: UInt64?
    public let variantSeed: UInt64?
    public let promptID: String
    public let prompt: String
    public let baselineWas: String
    public let conditionWas: String
    public let winner: String
    public let conditionResult: String
    public let confidence: Double
    public let briefReason: String
    public let aScores: [String: Int]?
    public let bScores: [String: Int]?
    public let structuredFields: [String: JSONValue]?
    public let rawJSON: String
}
