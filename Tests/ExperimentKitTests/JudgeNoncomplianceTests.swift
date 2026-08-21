import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Judge noncompliance is a RECORDED ROW, not a run-killer (Christian,
/// 2026-08-09; server twin: `Server/tests/test_paired_judge.py` +
/// `Server/tests/test_response_coding.py`).
///
/// A judge that answers but will not produce a valid verdict/code set for
/// ONE item was aborting entire evaluations after hours of generation and
/// hundreds of good judgments. The refusal to invent data stands — no
/// winner, no codes — but the failure becomes a per-item row kept for later
/// classification, excluded from every tally, aggregate, and agreement
/// statistic, with the count stamped on the report.
///
/// Two boundaries hold the line and are tested here as hard as the policy
/// itself: `JudgeNoncompliantError` is a distinct type, so TRANSPORT
/// failures still fail the session (a rate-limited call is not
/// noncompliance); and a judge failing more than `JudgeNoncompliance.cap`
/// of its column still fails the evaluation, because a report built mostly
/// on holes is not a result.
@Suite(.serialized) struct JudgeNoncomplianceTests {

    // MARK: - shared harness

    private func realFrenchHash() throws -> String {
        try StimulusSet(
            directory: VectorCatalog.conceptsDirectory
                .appending(component: "french")
        ).hash
    }

    /// A verified draft plus a source run of raw JSONL rows stamped with the
    /// live manifest hash (the `PairedJudgePairingTests` harness, reduced).
    private func plantStudy(
        name: String, rows: [String], judges: [String] = []
    ) throws -> URL {
        var manifest = try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french", stimulusSetHash: try realFrenchHash(),
                options: .init()))
        if !judges.isEmpty {
            manifest.judges = judges.map {
                .init(name: $0, kind: "local", model: nil)
            }
        }
        try ExperimentStore.save(manifest)
        let liveHash = ExperimentStore.manifestHash(manifest)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260809T000000000Z-exp-\(name)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: source.appending(component: "experiment.json"))
        try liveHash.write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        return source
    }

    private func row(
        study: String, condition: String, seed: UInt64,
        promptID: String, output: String
    ) -> String {
        "{\"experiment\":\"\(study)\",\"condition\":\"\(condition)\","
            + "\"seed\":\(seed),\"promptID\":\"\(promptID)\","
            + "\"prompt\":\"q-\(promptID)\",\"output\":\"\(output)\"}"
    }

    /// `count` baseline/steered pairs, one prompt each.
    private func pairedRows(study: String, count: Int) -> [String] {
        var rows: [String] = []
        var seed: UInt64 = 0
        for condition in ["baseline", "steered"] {
            for index in 0..<count {
                seed += 1
                rows.append(
                    row(
                        study: study, condition: condition, seed: seed,
                        promptID: "p\(index)",
                        output: "\(condition)-p\(index)"))
            }
        }
        return rows
    }

    private static let pairedEvaluation = ExperimentManifest.EvaluationSpec(
        kind: .pairedJudge, judgeModel: "claude-test",
        judgePrompt: "prefer the better response")

    private static let codingRubric = """
        ---
        mode: perResponseCoding
        field: mentionsLegalRule boolean
        field: mentionsEquity boolean
        ---
        Code the two booleans exactly as defined.
        """

    private static let codingEvaluation = ExperimentManifest.EvaluationSpec(
        kind: .pairedJudge, judgeModel: "",
        judgePrompt: JudgeNoncomplianceTests.codingRubric)

    private func withPairedSeams<T>(
        judge: @escaping @Sendable (
            String, String, String, String
        ) async throws -> PairedJudgeResponse,
        _ body: () async throws -> T
    ) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "noncompliance-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.judgeOverrideForTesting = judge
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body()
    }

    private func withCodingSeams<T>(
        coder: @escaping @Sendable (String, String) async throws -> String,
        _ body: () async throws -> T
    ) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "noncompliance-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.codingOverrideForTesting = coder
        defer {
            ExperimentTasks.codingOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body()
    }

    private func evaluateDirectory() throws -> URL {
        let runs = try FileManager.default.contentsOfDirectory(
            at: ExperimentStore.runsDirectory,
            includingPropertiesForKeys: nil)
        return try #require(
            runs.first { $0.lastPathComponent.hasSuffix("-evaluate") },
            "no evaluate run directory found")
    }

    private func jsonRows(_ url: URL) throws -> [[String: Any]] {
        try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n")
            .map { line in
                try #require(
                    try JSONSerialization.jsonObject(with: Data(line.utf8))
                        as? [String: Any])
            }
    }

    private func jsonObject(_ url: URL) throws -> [String: Any] {
        try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
    }

    private static func verdict(_ winner: String) -> PairedJudgeResponse {
        PairedJudgeResponse(
            winner: winner, confidence: 0.8, briefReason: "raw \(winner)")
    }

    // MARK: - the cap constant is one shared number

    @Test func theCapIsOneSharedQuarter() {
        // Server twin: `paired_judge.NONCOMPLIANCE_CAP`. Both the paired
        // judge and the coding loop read THIS constant — a second copy is
        // how the two engines drift.
        #expect(JudgeNoncompliance.cap == 0.25)
        #expect(JudgeNoncompliance.capPercentText == "25%")
    }

    // MARK: - the typed boundary at the retry wrappers

    /// Where the distinction is MADE: two invalid verdicts are
    /// `JudgeNoncompliantError`; anything thrown by the judge CALL comes
    /// back unchanged, so no caller can accidentally file a transport
    /// failure as noncompliance.
    @Test func theValidWinnerWrapperTypesTheTwoFailuresApart() async {
        await #expect(throws: JudgeNoncompliantError.self) {
            _ = try await ExperimentTasks.judgmentWithValidWinner(
                judgeName: "judge-1", item: "pair x/p0"
            ) { Self.verdict("both") }
        }
        await #expect(throws: PairedJudgeError.self) {
            _ = try await ExperimentTasks.judgmentWithValidWinner(
                judgeName: "judge-1", item: "pair x/p0"
            ) { throw PairedJudgeError(reason: "HTTP 503") }
        }
    }

    /// The coding wrapper's twin of the same boundary.
    @Test func theValidCodesWrapperTypesTheTwoFailuresApart() async throws {
        let schema = try #require(
            try ResponseCoding.parseRubric(Self.codingRubric))
        await #expect(throws: JudgeNoncompliantError.self) {
            _ = try await ResponseCoding.codesWithValidSchema(
                judgeName: "judge-1", item: "record x/p0[0]", schema: schema
            ) { ("garbage", nil) }
        }
        await #expect(throws: PairedJudgeError.self) {
            _ = try await ResponseCoding.codesWithValidSchema(
                judgeName: "judge-1", item: "record x/p0[0]", schema: schema
            ) { throw PairedJudgeError(reason: "HTTP 503") }
        }
    }

    // MARK: - paired judge

    /// One flaky pair no longer kills hours of evaluation: the failure
    /// becomes a recorded, classifiable row and the run completes.
    /// Refusal-to-invent stands — no winner is recorded.
    @Test func oneNoncompliantPairIsRecordedAndTheRunCompletes() async throws {
        try await withPairedSeams(judge: { _, prompt, _, _ in
            // p3 is garbage on BOTH attempts; every other pair is fine.
            prompt.contains("q-p3")
                ? Self.verdict("both") : Self.verdict("A")
        }) {
            let study = "noncompliant-one"
            let source = try plantStudy(
                name: study, rows: pairedRows(study: study, count: 8))
            let out = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study, sourceRunDirectory: source,
                evaluation: Self.pairedEvaluation)

            let rows = try jsonRows(
                out.appending(component: "judgments.jsonl"))
            #expect(rows.count == 8)  // the hole is written, not skipped
            let bad = rows.filter { $0["noncompliant"] as? Bool == true }
            #expect(bad.count == 1)
            let hole = try #require(bad.first)
            #expect(hole["promptID"] as? String == "p3")
            #expect(hole["condition"] as? String == "steered")
            // Present-and-null, not absent: the reader sees the hole.
            #expect(hole.keys.contains("outcome"))
            #expect(hole["outcome"] is NSNull)
            #expect(hole.keys.contains("judgment"))
            #expect(hole["judgment"] is NSNull)
            #expect(hole["judge"] as? String == "judge-1")
            #expect(hole["baselineWas"] as? String != nil)
            #expect(hole["baselineSeed"] as? UInt64 != nil)
            #expect(hole["variantSeed"] as? UInt64 != nil)
            let reason = try #require(
                hole["noncomplianceReason"] as? String)
            #expect(reason.contains("invalid verdict twice"))
            #expect(reason.contains("refusing to record invented data"))
            #expect(reason.count <= 2000)
            // Compliant rows gain NOTHING from any of this.
            for row in rows where row["noncompliant"] == nil {
                #expect(row["noncomplianceReason"] == nil)
                #expect(row["conditionResult"] as? String != nil)
            }

            let report = try jsonObject(
                out.appending(component: "judge-report.json"))
            #expect(report["noncompliantJudgments"] as? Int == 1)
            // The tally counts only real verdicts.
            let conditions = try #require(
                report["conditions"] as? [String: Any])
            let steered = try #require(conditions["steered"] as? [String: Any])
            #expect(steered["pairs"] as? Int == 7)
            let tallied =
                (steered["conditionWins"] as? Int ?? 0)
                + (steered["baselineWins"] as? Int ?? 0)
                + (steered["ties"] as? Int ?? 0)
            #expect(tallied == 7, "the hole is in no win/tie bucket")

            // Every written row is counted on disk, holes included.
            let status = try #require(RunStatusFile.read(at: out))
            #expect(status.status == "completed")
            #expect(status.itemsWritten == 8)
        }
    }

    /// A clean column is byte-compatible with before the policy: no new key
    /// on any row and none on the report.
    @Test func aCleanEvaluateGainsNoNoncomplianceKeys() async throws {
        try await withPairedSeams(judge: { _, _, _, _ in
            Self.verdict("A")
        }) {
            let study = "noncompliant-none"
            let source = try plantStudy(
                name: study, rows: pairedRows(study: study, count: 3))
            let out = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study, sourceRunDirectory: source,
                evaluation: Self.pairedEvaluation)
            for row in try jsonRows(
                out.appending(component: "judgments.jsonl"))
            {
                #expect(row["noncompliant"] == nil)
                #expect(row["noncomplianceReason"] == nil)
            }
            let report = try jsonObject(
                out.appending(component: "judge-report.json"))
            #expect(report["noncompliantJudgments"] == nil)
        }
    }

    /// TRANSPORT failure is not noncompliance: the judge's CALL failed, the
    /// session still dies, and resume machinery — not a recorded hole — is
    /// what completes it. A rate-limited judge must never be filed as a
    /// garbage one.
    @Test func aTransportErrorStillFailsTheWholeEvaluate() async throws {
        try await withPairedSeams(judge: { _, prompt, _, _ in
            if prompt.contains("q-p2") {
                throw PairedJudgeError(reason: "HTTP 429 rate limited")
            }
            return Self.verdict("A")
        }) {
            let study = "noncompliant-transport"
            let source = try plantStudy(
                name: study, rows: pairedRows(study: study, count: 8))
            await #expect(throws: PairedJudgeError.self) {
                try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: study, sourceRunDirectory: source,
                    evaluation: Self.pairedEvaluation)
            }
            let out = try evaluateDirectory()
            // Nothing was recorded as a hole, and no report summarizes a
            // half-judged panel.
            for row in try jsonRows(
                out.appending(component: "judgments.jsonl"))
            {
                #expect(row["noncompliant"] == nil)
            }
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "judge-report.json")
                        .path))
            let status = try #require(RunStatusFile.read(at: out))
            #expect(status.status == "failed")
            #expect(status.error?.contains("rate limited") == true)
        }
    }

    /// A judge failing more than a quarter of its column is broken, and a
    /// "completed" evaluation built on it would be worse than a failed one
    /// — but every row, holes included, is already on disk.
    @Test func systemicNoncomplianceFailsUnderTheCap() async throws {
        try await withPairedSeams(judge: { _, _, _, _ in
            Self.verdict("both")
        }) {
            let study = "noncompliant-systemic"
            let source = try plantStudy(
                name: study, rows: pairedRows(study: study, count: 4))
            do {
                _ = try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: study, sourceRunDirectory: source,
                    evaluation: Self.pairedEvaluation)
                Issue.record("expected the systemic-failure refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("noncompliant on 4 of 4 pairs"))
                #expect(error.reason.contains("> 25% cap"))
                #expect(error.reason.contains("systemic judge failure"))
                #expect(error.reason.contains("not flakiness"))
                #expect(error.reason.contains("re-run evaluate"))
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
            let out = try evaluateDirectory()
            let rows = try jsonRows(
                out.appending(component: "judgments.jsonl"))
            #expect(rows.count == 4)
            #expect(rows.allSatisfy { $0["noncompliant"] as? Bool == true })
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "judge-report.json")
                        .path))
        }
    }

    // MARK: - per-response coding

    private func codingRows(study: String, count: Int) -> [String] {
        (0..<count).map { index in
            row(
                study: study, condition: "baseline", seed: UInt64(index + 1),
                promptID: "p\(index)", output: "Ruling \(index).")
        }
    }

    private static let goodCodes =
        "{\"codes\": {\"mentionsLegalRule\": true, "
        + "\"mentionsEquity\": false}, \"brief_reason\": \"r\"}"

    /// A few refusals no longer abort hours of coding: the record becomes a
    /// classifiable `codes: null` row, the run completes, and the report
    /// says how many holes each column has.
    @Test func oneNoncompliantCodingIsRecordedAndTheRunCompletes()
        async throws
    {
        try await withCodingSeams(coder: { judge, prompt in
            // judge-1 is garbage on p2 only (1 of 5 — under the cap);
            // judge-2 codes everything.
            judge == "judge-1" && prompt.contains("Ruling 2.")
                ? "garbage" : Self.goodCodes
        }) {
            let study = "coding-noncompliant-one"
            let source = try plantStudy(
                name: study, rows: codingRows(study: study, count: 5),
                judges: ["judge-1", "judge-2"])
            let out = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study, sourceRunDirectory: source,
                evaluation: Self.codingEvaluation)

            let rows = try jsonRows(out.appending(component: "codings.jsonl"))
            #expect(rows.count == 10)
            let bad = rows.filter { $0["noncompliant"] as? Bool == true }
            #expect(bad.count == 1)
            let hole = try #require(bad.first)
            #expect(hole["judge"] as? String == "judge-1")
            #expect(hole["promptID"] as? String == "p2")
            // Identity keys match a normal row; codes is present-and-null.
            #expect(hole["experiment"] as? String == study)
            #expect(hole["condition"] as? String == "baseline")
            #expect(hole["sampleIndex"] as? UInt64 == 0)
            #expect(hole["seed"] as? UInt64 == 3)
            #expect(hole["judgeKind"] as? String == "local")
            #expect(hole["judgeModel"] as? String == "test/model")
            #expect(hole.keys.contains("codes"))
            #expect(hole["codes"] is NSNull)
            #expect(
                (hole["noncomplianceReason"] as? String)?
                    .contains("invalid codes twice") == true)

            let report = try jsonObject(
                out.appending(component: "coding-report.json"))
            #expect(report["noncompliantCodings"] as? Int == 1)
            // Every row WRITTEN (the server's len(rows)) — holes included.
            #expect(report["codings"] as? Int == 10)
            let details = try #require(
                report["judgeDetails"] as? [[String: Any]])
            let first = try #require(details.first { $0["name"] as? String == "judge-1" })
            #expect(first["noncompliantCodings"] as? Int == 1)
            let second = try #require(
                details.first { $0["name"] as? String == "judge-2" })
            #expect(second["noncompliantCodings"] == nil)
            // Aggregates count only real codings…
            let conditions = try #require(
                report["conditions"] as? [String: Any])
            let baseline = try #require(conditions["baseline"] as? [String: Any])
            #expect(baseline["codings"] as? Int == 9)
            let fields = try #require(baseline["fields"] as? [String: Any])
            let rule = try #require(
                fields["mentionsLegalRule"] as? [String: Any])
            #expect(rule["n"] as? Int == 9)
            // …and agreement never compares a judgment to a hole: only the
            // four cells BOTH coders reached.
            let agreement = try #require(
                report["fieldAgreement"] as? [[String: Any]])
            let ruleEntry = try #require(
                agreement.first { $0["field"] as? String == "mentionsLegalRule" })
            #expect(ruleEntry["n"] as? Int == 4)
        }
    }

    /// A clean coding column carries no new key anywhere.
    @Test func aCleanCodingEvaluateGainsNoNoncomplianceKeys() async throws {
        try await withCodingSeams(coder: { _, _ in Self.goodCodes }) {
            let study = "coding-noncompliant-none"
            let source = try plantStudy(
                name: study, rows: codingRows(study: study, count: 3),
                judges: ["judge-1"])
            let out = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study, sourceRunDirectory: source,
                evaluation: Self.codingEvaluation)
            for row in try jsonRows(out.appending(component: "codings.jsonl")) {
                #expect(row["noncompliant"] == nil)
            }
            let report = try jsonObject(
                out.appending(component: "coding-report.json"))
            #expect(report["noncompliantCodings"] == nil)
            let details = try #require(
                report["judgeDetails"] as? [[String: Any]])
            #expect(details.allSatisfy { $0["noncompliantCodings"] == nil })
        }
    }

    /// Transport failure on the coding path is the paired path's rule: the
    /// coder's CALL failed, so the session dies rather than recording a
    /// hole.
    @Test func aCodingTransportErrorStillFailsTheWholeRun() async throws {
        try await withCodingSeams(coder: { _, prompt in
            if prompt.contains("Ruling 1.") {
                throw PairedJudgeError(reason: "HTTP 500 from the provider")
            }
            return Self.goodCodes
        }) {
            let study = "coding-noncompliant-transport"
            let source = try plantStudy(
                name: study, rows: codingRows(study: study, count: 8),
                judges: ["judge-1"])
            await #expect(throws: PairedJudgeError.self) {
                try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: study, sourceRunDirectory: source,
                    evaluation: Self.codingEvaluation)
            }
            let out = try evaluateDirectory()
            for row in try jsonRows(out.appending(component: "codings.jsonl")) {
                #expect(row["noncompliant"] == nil)
            }
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "coding-report.json")
                        .path))
        }
    }

    /// Half a column refused is past the cap: the run still fails — but the
    /// refused record survives as a `codes: null` row beside the finished
    /// one, so nothing already judged is lost.
    @Test func systemicCodingNoncomplianceFailsUnderTheCap() async throws {
        try await withCodingSeams(coder: { _, prompt in
            prompt.contains("Ruling 1.") ? "garbage" : Self.goodCodes
        }) {
            let study = "coding-noncompliant-systemic"
            let source = try plantStudy(
                name: study, rows: codingRows(study: study, count: 2),
                judges: ["judge-1"])
            do {
                _ = try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: study, sourceRunDirectory: source,
                    evaluation: Self.codingEvaluation)
                Issue.record("expected the systemic-failure refusal")
            } catch let error as ExperimentError {
                #expect(
                    error.reason.contains("noncompliant on 1 of 2 record(s)"))
                #expect(error.reason.contains("> 25% cap"))
                #expect(error.reason.contains("systemic coder failure"))
                #expect(error.reason.contains("codings.jsonl"))
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
            let out = try evaluateDirectory()
            let rows = try jsonRows(out.appending(component: "codings.jsonl"))
            #expect(rows.count == 2)  // the finished row AND the refusal
            #expect(rows.filter { $0["noncompliant"] as? Bool == true }.count == 1)
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "coding-report.json")
                        .path))
        }
    }
}
