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
    /// The FIRST `StudyResultRepository.previewResponseLimit` lines of
    /// generations.jsonl that are generated responses — a bounded preview
    /// for the browser client and the short list under the review button.
    /// It is not the run: `responseRecordCount` is how many records the
    /// file holds, and the review sheet pages through all of them
    /// (`StudyRecordReview`).
    public let generations: [StudyGenerationPreview]
    /// The FIRST `StudyResultRepository.previewJudgmentLimit` lines of
    /// judgments.jsonl that carry a verdict — the same bounded preview.
    /// `judgmentRecordCount` is the file's row count, noncompliant rows
    /// included.
    public let judgments: [StudyJudgePreview]
    /// Every record in generations.jsonl: responses, cut-off responses,
    /// failure records, answer-option readings, and lines that cannot be
    /// read. A count of lines, so nothing a decoder dislikes can lower it.
    public var responseRecordCount: Int = 0
    /// Every row in the judge artifact's judgments.jsonl, noncompliant and
    /// unreadable rows included.
    public var judgmentRecordCount: Int = 0
    /// Judge evaluations of this run that stopped before writing a judge
    /// report. Their rows are real and reviewable, and they are not a
    /// result — shown so a stopped evaluation is never simply absent.
    public var unfinishedEvaluations: [StudyUnfinishedEvaluation] = []
}

/// A judge evaluation that did not finish: the directory holds the rows it
/// wrote and a status file, and no judge-report.json.
public struct StudyUnfinishedEvaluation: Identifiable, Codable, Sendable, Equatable {
    public var id: String { directoryName }
    public let directoryName: String
    public let path: String
    /// What the status file says happened, in one line.
    public let summary: String
    /// Rows the evaluation wrote before it stopped.
    public let judgmentRecordCount: Int
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

/// A judge report as the Results view shows it, read from EITHER engine's
/// judge-report.json (`StudyJudgeReportReader`). The two engines write
/// different key sets for the same facts; this is the one shape the views
/// read. Every value here is the report's own — nothing is recomputed, and
/// a value an engine does not store is nil rather than filled in.
public struct PairedJudgeReportView: Codable, Sendable, Equatable {
    public struct Condition: Codable, Sendable, Equatable {
        public let name: String
        public let pairs: Int
        public let conditionWins: Int
        public let baselineWins: Int
        public let ties: Int
        /// nil when the report stores none (the Python engine's tallies
        /// carry counts only).
        public let meanConfidence: Double?
        public let structuredSummaries: [String: StructuredFieldSummaryView]
    }

    /// Which engine's key set the report was written in.
    public enum Dialect: String, Codable, Sendable {
        /// The Mac engine: `sourceRunDirectory`, `conditionWins`, one
        /// tally for the whole panel.
        case macEngine
        /// The Python engine (server and cluster runs): `sourceRun`,
        /// `variantWins`, one tally per judge.
        case pythonEngine
    }

    /// One judge's own tallies. The Python engine stores these; the Mac
    /// engine stores one panel-wide tally instead, so its reports have none.
    public struct JudgeBlock: Codable, Sendable, Equatable {
        public let name: String
        public let requestedModel: String?
        public let actualModel: String?
        public let pairs: Int?
        public let conditions: [Condition]
        public let noncompliantJudgments: Int?
        public let salvagedVerdicts: Int?
    }

    /// Two judges compared over the pairs both judged.
    public struct JudgeAgreement: Codable, Sendable, Equatable {
        public let judgeA: String
        public let judgeB: String
        public let items: Int
        /// A fraction from 0 to 1 on both engines.
        public let percentAgreement: Double
        /// nil when the report stores none: kappa is undefined when each
        /// judge gave one label throughout and the two labels differ.
        public let kappa: Double?
    }

    /// One judge compared with the human ratings pinned to the study.
    public struct HumanAgreement: Codable, Sendable, Equatable {
        public let judge: String
        public let items: Int
        public let percentAgreement: Double
        public let kappa: Double?
    }

    /// An evaluation finished by resuming an earlier, unfinished one.
    public struct JudgingSessions: Codable, Sendable, Equatable {
        public let resumedFrom: String?
        public let reusedJudgments: Int
        public let freshJudgments: Int
    }

    /// The run this report judges. The Mac engine stores a full path; the
    /// Python engine stores the run directory's name.
    public let sourceRunDirectory: String
    public let judgeModel: String
    /// The report's top-level tally. On a Mac-engine report this adds up
    /// the whole panel (pairs count judges × items). On a Python-engine
    /// report it is the FIRST judge's tally, which that engine repeats at
    /// the top level for older readers — `judgeBlocks` holds every judge.
    public let conditions: [Condition]

    public var dialect: Dialect? = nil
    /// Panel names in evaluation order; empty on reports that name none.
    public var judgeNames: [String] = []
    public var judgeBlocks: [JudgeBlock] = []
    public var judgeAgreement: [JudgeAgreement] = []
    /// nil when the report has no human-agreement entry (no human ratings
    /// were pinned); an empty list is never written by either engine.
    public var humanAgreement: [HumanAgreement]? = nil
    /// Pairs a judge answered without a usable verdict, panel total. Both
    /// engines write this only when it is not zero.
    public var noncompliantJudgments: Int? = nil
    /// true only when the source run carried no study stamp and the
    /// evaluation was allowed to proceed anyway.
    public var epochUnverified: Bool? = nil
    /// The fields that differed from the source run's stamp when a
    /// difference was accepted, as the engine recorded them.
    public var measurementDrift: String? = nil
    /// Responses removed by declared exclusion rules before judging.
    public var excludedRecords: Int? = nil
    public var judgingSessions: JudgingSessions? = nil
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
