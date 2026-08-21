import Foundation
import Testing

@testable import ExperimentKit

/// The `emptyAnalysis` advisory's DETAIL, and the `effectSizesSchema` stamp
/// that names which CSV dialect the analysis wrote (WP0 dry run #2).
///
/// The advisory used to assert ONE cause unconditionally — "the source run
/// has no non-baseline condition to pair against" — and on the run that
/// produced the finding that was false: two conditions, 24 records, and what
/// actually failed was reading and pairing them. The refusal half of that
/// finding is the foreign-substrate epoch gate; this half is the honesty of
/// the sentence that remains when analyze legitimately measures nothing.
///
/// Server twins: `cli_payloads.empty_analysis_detail`,
/// `cli_payloads.EMPTY_ANALYSIS_NO_CONTRAST`,
/// `cli_payloads.EFFECT_SIZES_SCHEMA` (tested in
/// `Server/tests/test_cli_envelope.py`).
struct EmptyAnalysisAdvisoryTests {

    // MARK: - The detail says what was observed

    @Test func baselineOnlyKeepsTheHistoricalSentence() {
        #expect(
            ExperimentTasks.emptyAnalysisDetail(
                runName: "r", recordCount: 12, conditions: ["baseline"])
                == ExperimentTasks.emptyAnalysisNoContrast)
        // Byte-identical to the server's EMPTY_ANALYSIS_NO_CONTRAST.
        #expect(
            ExperimentTasks.emptyAnalysisNoContrast
                == "0 effect-size entries — the source run has no non-baseline "
                + "condition to pair against")
    }

    /// The dry-run case: records exist, a contrast exists, nothing paired.
    /// The detail must NOT claim the study has no contrast, must say the
    /// records could not be read/paired, and must name the likeliest cause.
    @Test func aContrastThatDidNotPairIsNamedAsSuch() {
        let detail = ExperimentTasks.emptyAnalysisDetail(
            runName: "20260818T120000000-exp-s-run", recordCount: 24,
            conditions: ["baseline", "fear-a2"])
        #expect(!detail.contains("no non-baseline condition"))
        #expect(detail.contains("24 records"))
        #expect(detail.contains("baseline, fear-a2"))
        #expect(detail.contains("could not read or pair those records"))
        #expect(detail.contains("artifacts produced on the other engine"))
        #expect(detail.contains("record-schema mismatch"))
    }

    @Test func theEmptyAndUnreadableCasesAreDistinct() {
        let empty = ExperimentTasks.emptyAnalysisDetail(
            runName: "r", recordCount: 0, conditions: [])
        #expect(empty.contains("holds no records at all"))

        let unreadable = ExperimentTasks.emptyAnalysisDetail(
            runName: "r", recordCount: nil, conditions: [])
        #expect(unreadable.contains("could not be read here"))
        #expect(empty != unreadable)
    }

    @Test func oneRecordOneConditionReadsInTheSingular() {
        let detail = ExperimentTasks.emptyAnalysisDetail(
            runName: "r", recordCount: 1, conditions: ["fear-a2"])
        #expect(detail.contains("holds 1 record across condition fear-a2"))
    }

    // MARK: - The reader

    @Test func theReaderCountsRecordsAndConditions() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "empty-analysis-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Missing file: a nil count, which is NOT "it was empty".
        #expect(ExperimentTasks.analysisSourceRecords(at: directory).recordCount == nil)

        let lines = [
            #"{"condition":"baseline","promptID":"p1"}"#,
            #"{"condition":"fear-a2","promptID":"p1"}"#,
            #"{"condition":"fear-a2","promptID":"p2"}"#,
        ]
        try lines.joined(separator: "\n").write(
            to: directory.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        let read = ExperimentTasks.analysisSourceRecords(at: directory)
        #expect(read.recordCount == 3)
        #expect(read.conditions == ["baseline", "fear-a2"])
    }

    // MARK: - The CSV dialect is NAMED, not unified

    /// Both engines write paired effect sizes; the column names differ by
    /// idiom and are deliberately NOT being unified (runs are immutable and
    /// each engine's readers depend on its own columns). The envelope names
    /// the dialect so a cross-engine reader can tell them apart, and any
    /// future unification is a schema-versioned change announced by this
    /// field. Server twin: `cli_payloads.EFFECT_SIZES_SCHEMA`.
    @Test func theAnalyzePayloadNamesItsEffectSizeDialect() throws {
        #expect(ExperimentCLIRunner.effectSizesSchema == "metric-meanDiff")
        #expect(ExperimentCLIRunner.effectSizesSchema != "endpoint-deltaMean")

        let directory = FileManager.default.temporaryDirectory
            .appending(component: "analysis-schema-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try #"{"experiment":"s","effectSizes":[]}"#.write(
            to: directory.appending(component: "analysis.json"),
            atomically: true, encoding: .utf8)
        let payload = ExperimentCLIRunner.analysisPayload(inRunAt: directory)
        guard case .string(let schema)? = payload["effectSizesSchema"] else {
            Issue.record("analyze payload did not name its CSV dialect")
            return
        }
        #expect(schema == "metric-meanDiff")
        // The two engines' effect-size CSV headers, as they stand. A change
        // to either is a schema-versioned change, never a silent rewrite.
        #expect(
            ExperimentTasks.effectSizesCSV([]).hasPrefix(
                "condition,metric,n,meanDiff,"))
    }
}
