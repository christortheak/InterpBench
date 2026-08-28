import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The per-response coding instrument (2026-08-04; server twin:
/// `Server/tests/test_response_coding.py`).
///
/// The K&Z §S9 procedure codes each individual response (two booleans, no
/// winner); forcing it through the paired-preference machinery produced an
/// improvised A/B verdict and codes that could not be unblinded to arms.
/// Under test: the strict rubric-frontmatter grammar, the byte-pinned
/// coding prompt (goldens shared with the Python builder), typed code
/// validation with the retry-once-then-refuse closure, the coding evaluate
/// end to end (codings.jsonl + coding-report.json, per-field κ), and the
/// paired-only consumers refusing a coding rubric.
@Suite(.serialized) struct ResponseCodingTests {

    static let codingRubric = """
        ---
        mode: perResponseCoding
        field: mentionsLegalRule boolean
        field: mentionsEquity boolean
        ---
        Code the two booleans exactly as defined.
        """

    // MARK: rubric frontmatter grammar

    @Test func plainRubricIsNotACodingRubric() throws {
        #expect(try ResponseCoding.parseRubric("Judge which is better.") == nil)
        // A --- later in the text is not frontmatter.
        #expect(try ResponseCoding.parseRubric("Rubric\n---\nmode: x\n---\n") == nil)
    }

    @Test func validDeclarationParsesFieldsAndBody() throws {
        let schema = try #require(
            try ResponseCoding.parseRubric(
                "---\nmode: perResponseCoding\n\n"
                    + "field: a boolean\nfield: b integer optional\n"
                    + "field: c enum(x|y|w)\nfield: d string\n"
                    + "field: e number\n---\nBody text.\n"))
        #expect(schema.fields.map(\.name) == ["a", "b", "c", "d", "e"])
        #expect(schema.fields.map(\.type) == [
            "boolean", "integer", "enum", "string", "number",
        ])
        #expect(schema.fields.map(\.optional) == [
            false, true, false, false, false,
        ])
        #expect(schema.fields[2].values == ["x", "y", "w"])
        #expect(schema.body == "Body text.")
    }

    @Test(arguments: [
        ("---\nmode: perResponseCoding\nfield: a boolean\n", "never closes"),
        ("---\nfield: a boolean\n---\n", "no 'mode:' line"),
        ("---\nmode: pairwiseElo\nfield: a boolean\n---\n",
         "unknown rubric mode 'pairwiseElo'"),
        ("---\nmode: perResponseCoding\n---\n", "at least one"),
        ("---\nmode: perResponseCoding\nfield: a boolean\n"
            + "field: a integer\n---\n", "twice"),
        ("---\nmode: perResponseCoding\nfield: a tristate\n---\n",
         "unknown field type"),
        ("---\nmode: perResponseCoding\nfield: 2fast boolean\n---\n",
         "invalid field name"),
        ("---\nmode: perResponseCoding\nfield: a boolean maybe\n---\n",
         "only modifier"),
        ("---\nmode: perResponseCoding\nfeild: a boolean\n---\n",
         "unrecognized rubric frontmatter line"),
        ("---\nmode: perResponseCoding\nfield: a enum()\n---\n",
         "malformed enum"),
        ("---\nmode: perResponseCoding\nfield: a\n---\n",
         "malformed field declaration"),
    ]) func malformedDeclarationsRefuse(text: String, fragment: String) {
        #expect(throws: ExperimentError.self) {
            try ResponseCoding.parseRubric(text)
        }
        do {
            _ = try ResponseCoding.parseRubric(text)
        } catch let error as ExperimentError {
            #expect(
                error.reason.contains(fragment),
                "expected '\(fragment)' in: \(error.reason)")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: the byte-pinned prompt wrapper

    /// Repo root derived from this file's compile-time path (same anchor
    /// as the paired-judge golden tests).
    private static var goldenDirectory: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "prompts", "fixtures", "coding-judge")
    }

    private struct GoldenInputs: Decodable {
        let rubric: String
        let response: String
        let taskPrompt: String
    }

    @Test func promptWrapperMatchesTheCommittedGoldens() throws {
        let inputs = try JSONDecoder().decode(
            GoldenInputs.self,
            from: Data(
                contentsOf: Self.goldenDirectory.appending(
                    component: "inputs.json")))
        let schema = try #require(try ResponseCoding.parseRubric(inputs.rubric))
        let built = ResponseCoding.buildPrompt(
            schema: schema, response: inputs.response,
            taskPrompt: inputs.taskPrompt)
        let expected = try String(
            contentsOf: Self.goldenDirectory.appending(
                component: "prompt.golden.txt"),
            encoding: .utf8)
        #expect(
            built == expected,
            """
            coding-prompt wrapper drifted from its committed golden — the \
            wrapper is byte-pinned cross-engine (the server's \
            response_coding.build_prompt reads the same fixture); change it \
            only deliberately on BOTH engines and regenerate \
            prompts/fixtures/coding-judge/.
            """)
        let noTask = ResponseCoding.buildPrompt(
            schema: schema, response: inputs.response, taskPrompt: nil)
        let noTaskExpected = try String(
            contentsOf: Self.goldenDirectory.appending(
                component: "prompt-no-task.golden.txt"),
            encoding: .utf8)
        #expect(noTask == noTaskExpected)
        #expect(!noTask.contains("=== Task prompt"))
    }

    // MARK: code validation

    static let validationSchema = try! ResponseCoding.parseRubric(
        "---\nmode: perResponseCoding\n"
            + "field: flag boolean\nfield: count integer optional\n"
            + "field: score number\nfield: label enum(a|b)\n"
            + "field: note string\n---\nbody\n")!

    private func codes(_ overrides: [String: JSONValue] = [:]) -> [String: JSONValue] {
        var codes: [String: JSONValue] = [
            "flag": .bool(true), "count": .number(3), "score": .number(1.5),
            "label": .string("a"), "note": .string("n"),
        ]
        for (key, value) in overrides { codes[key] = value }
        return codes
    }

    @Test func validCodesPassAndIntegralFloatsCountAsIntegers() {
        let schema = Self.validationSchema
        #expect(ResponseCoding.validate(codes: codes(), schema: schema) == [])
        // JSON does not distinguish 3 from 3.0; optional fields may be null.
        #expect(
            ResponseCoding.validate(
                codes: codes(["count": .number(3.0)]), schema: schema) == [])
        #expect(
            ResponseCoding.validate(
                codes: codes(["count": .null]), schema: schema) == [])
    }

    @Test func invalidCodesNameTheProblem() {
        let schema = Self.validationSchema
        let cases: [([String: JSONValue], String)] = [
            (codes(["flag": .string("true")]), "'flag' must be a boolean"),
            (codes(["count": .number(2.5)]), "'count' must be an integer"),
            (codes(["count": .bool(true)]), "'count' must be an integer"),
            (codes(["score": .string("high")]), "'score' must be a number"),
            (codes(["label": .string("c")]), "'label' must be one of a|b"),
            (codes(["note": .number(7)]), "'note' must be a string"),
            (codes(["flag": .null]), "required field 'flag' is null"),
        ]
        for (invalid, fragment) in cases {
            let problems = ResponseCoding.validate(
                codes: invalid, schema: schema)
            #expect(
                problems.contains { $0.contains(fragment) },
                "expected '\(fragment)' in: \(problems)")
        }
        var missing = codes()
        missing.removeValue(forKey: "label")
        #expect(
            ResponseCoding.validate(codes: missing, schema: schema)
                == ["missing field 'label'"])
    }

    @Test func parseCodesExtractsFromFencedResponses() throws {
        let schema = Self.validationSchema
        let verdict = try ResponseCoding.parseCodes(
            "Here are my codes:\n```json\n"
                + "{\"codes\": {\"flag\": true, \"count\": null, "
                + "\"score\": 2, \"label\": \"b\", \"note\": \"x\"}, "
                + "\"brief_reason\": \"r\"}\n```",
            schema: schema)
        #expect(verdict.codes["flag"] == .bool(true))
        #expect(verdict.briefReason == "r")
        #expect(throws: ResponseCoding.InvalidCodesError.self) {
            try ResponseCoding.parseCodes("no json here", schema: schema)
        }
        #expect(throws: ResponseCoding.InvalidCodesError.self) {
            try ResponseCoding.parseCodes(
                "{\"brief_reason\": \"no codes\"}", schema: schema)
        }
    }

    /// The 2026-08-06 review finding (server twin:
    /// `test_undeclared_keys_are_kept_but_kept_out_of_the_measurement`). A
    /// coder that volunteers an undeclared field is saying something about
    /// the rubric, so the key is never dropped — but sitting inside `codes`
    /// it was indistinguishable from the pinned measurement. It moves to its
    /// own non-evidence block under the same key on both engines.
    @Test func undeclaredKeysAreKeptOutOfTheMeasurementNotDropped() throws {
        let schema = Self.validationSchema
        let verdict = try ResponseCoding.parseCodes(
            "{\"codes\": {\"flag\": true, \"count\": 1, \"score\": 2, "
                + "\"label\": \"a\", \"note\": \"n\", "
                + "\"confidence\": 0.9, \"mood\": \"stern\"}, "
                + "\"brief_reason\": \"r\"}",
            schema: schema)
        #expect(
            Set(verdict.codes.keys) == Set(schema.fields.map(\.name)))
        #expect(
            verdict.undeclaredCodes
                == ["confidence": .number(0.9), "mood": .string("stern")])

        // The ordinary case carries nothing: the row omits the block.
        let plain = try ResponseCoding.parseCodes(
            "{\"codes\": {\"flag\": true, \"count\": 1, \"score\": 2, "
                + "\"label\": \"a\", \"note\": \"n\"}, "
                + "\"brief_reason\": \"r\"}",
            schema: schema)
        #expect(plain.undeclaredCodes.isEmpty)
    }

    // MARK: retry-once-then-refuse closure

    @Test func oneMalformedResponseRetriesAndRecordsTheAttempt() async throws {
        let schema = Self.validationSchema
        let responses = AsyncQueue([
            "not json at all",
            "{\"codes\": {\"flag\": true, \"count\": 1, \"score\": 2, "
                + "\"label\": \"a\", \"note\": \"n\"}, "
                + "\"brief_reason\": \"r\"}",
        ])
        let invalid = InvalidLog()
        let verdict = try await ResponseCoding.codesWithValidSchema(
            judgeName: "j", item: "record fear/p0[0]", schema: schema,
            onInvalid: { await invalid.append($0) }
        ) { (await responses.next(), nil) }
        #expect(verdict.codes["flag"] == .bool(true))
        let attempts = await invalid.entries
        #expect(attempts.count == 1)
        #expect(attempts.first?["rawResponse"] == "not json at all")
    }

    @Test func twoInvalidResponsesRefuseNamingJudgeAndItem() async {
        let schema = Self.validationSchema
        // The TYPE is the contract (2026-08-09): a coder that answered
        // garbage is noncompliance the coding loop records as a row, not
        // the transport failure that still kills the session.
        await #expect(throws: JudgeNoncompliantError.self) {
            _ = try await ResponseCoding.codesWithValidSchema(
                judgeName: "j", item: "record fear/p0[0]", schema: schema
            ) { ("{\"codes\": {\"flag\": \"yes\"}}", nil) }
        }
        do {
            _ = try await ResponseCoding.codesWithValidSchema(
                judgeName: "j", item: "record fear/p0[0]", schema: schema
            ) { ("{\"codes\": {\"flag\": \"yes\"}}", nil) }
        } catch let error as JudgeNoncompliantError {
            #expect(error.reason.contains(
                "judge 'j' returned invalid codes twice"))
            #expect(error.reason.contains("record fear/p0[0]"))
            #expect(error.reason.contains("refusing to record invented data"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: paired-only machinery refuses coding rubrics

    @Test func refuseIfCodingNamesTheConsumer() throws {
        do {
            try ResponseCoding.refuseIfCoding(
                Self.codingRubric,
                context: "the sweep's judgeScore objective",
                rubricFile: "prompts/rubrics/c.md")
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("declares perResponseCoding"))
            #expect(error.reason.contains("sweep's judgeScore objective"))
        }
        // A paired rubric passes untouched.
        try ResponseCoding.refuseIfCoding("plain paired rubric", context: "x")
    }

    // MARK: engine-computed word count

    @Test func wordCountIsWhitespaceRuns() {
        #expect(ResponseCoding.wordCount("The rule controls. Affirmed.") == 4)
        #expect(ResponseCoding.wordCount("  a\n b\t\tc ") == 3)
        #expect(ResponseCoding.wordCount("") == 0)
    }

    // MARK: the coding evaluate, end to end

    /// A coding-mode rubric routes evaluate into the coding instrument:
    /// every sampled record — baseline INCLUDED — is coded by every judge,
    /// rows stream to codings.jsonl, the report aggregates per condition
    /// per field with per-field inter-judge agreement, and no paired
    /// artifact (judgments.jsonl / judge-report.json) is written. Server
    /// twin: `test_coding_evaluate_codes_every_sampled_record_…`.
    @Test func codingEvaluateCodesEveryRecordAndReportsAgreement() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "coding-eval-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.codingOverrideForTesting = { judge, prompt in
            // judge-2 flips the codes on the steered record — the report's
            // agreement must reflect the per-field disagreement.
            let flipped = judge == "judge-2" && prompt.contains("Fairness")
            return "{\"codes\": {\"mentionsLegalRule\": \(!flipped), "
                + "\"mentionsEquity\": \(flipped)}, "
                + "\"brief_reason\": \"r\"}"
        }
        defer {
            ExperimentTasks.codingOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        let study = "vg-coding"
        var manifest = try ExperimentStore.create(
            name: study, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french",
                stimulusSetHash: try StimulusSet(
                    directory: VectorCatalog.conceptsDirectory
                        .appending(component: "french")
                ).hash,
                options: .init()))
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "judge-2", kind: "local", model: nil),
        ]
        try ExperimentStore.save(manifest)
        let liveHash = ExperimentStore.manifestHash(manifest)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260804T000000000Z-exp-\(study)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try liveHash.write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        let rows = [
            "{\"experiment\":\"\(study)\",\"condition\":\"baseline\","
                + "\"seed\":1,\"promptID\":\"p0\",\"prompt\":\"q\","
                + "\"output\":\"The rule controls. Affirmed.\"}",
            "{\"experiment\":\"\(study)\",\"condition\":\"fear\","
                + "\"seed\":2,\"promptID\":\"p0\",\"prompt\":\"q\","
                + "\"output\":\"Fairness to the donor matters most here.\"}",
        ]
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)

        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: study,
            sourceRunDirectory: source,
            evaluation: ExperimentManifest.EvaluationSpec(
                kind: .pairedJudge, judgeModel: "",
                judgePrompt: Self.codingRubric))

        let lines = try String(
            contentsOf: out.appending(component: "codings.jsonl"),
            encoding: .utf8
        ).split(separator: "\n")
        // 2 records × 2 judges — the baseline is coded, never paired away.
        #expect(lines.count == 4)
        var conditions: Set<String> = []
        for line in lines {
            let object = try #require(
                try JSONSerialization.jsonObject(with: Data(line.utf8))
                    as? [String: Any])
            #expect(object["judgeModel"] as? String == "test/model")
            // Nothing was invented here, so the non-evidence block is
            // ABSENT from the row rather than present and empty.
            #expect(object["undeclaredCodes"] == nil)
            conditions.insert(try #require(object["condition"] as? String))
        }
        #expect(conditions == ["baseline", "fear"])

        let report = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(
                        component: "coding-report.json")))
                as? [String: Any])
        #expect(report["mode"] as? String == "perResponseCoding")
        #expect(report["judges"] as? [String] == ["judge-1", "judge-2"])
        #expect(report["codings"] as? Int == 4)
        let reportConditions = try #require(
            report["conditions"] as? [String: Any])
        let fear = try #require(reportConditions["fear"] as? [String: Any])
        // The engine computed word counts — no judge estimated one.
        #expect(fear["meanWordCount"] as? Double == 7)
        let fearFields = try #require(fear["fields"] as? [String: Any])
        let legalRule = try #require(
            fearFields["mentionsLegalRule"] as? [String: Any])
        #expect(legalRule["n"] as? Int == 2)
        #expect(legalRule["trueShare"] as? Double == 0.5)
        // Per-field agreement: the judges agree on baseline, disagree on
        // fear → 50% per field.
        let agreement = try #require(
            report["fieldAgreement"] as? [[String: Any]])
        let byField = Dictionary(
            uniqueKeysWithValues: agreement.map {
                ($0["field"] as? String ?? "", $0)
            })
        #expect(
            byField["mentionsLegalRule"]?["percentAgreement"] as? Double
                == 0.5)
        #expect(byField["mentionsLegalRule"]?["n"] as? Int == 2)
        // The confusion counts beside the statistic (cross-engine key
        // "confusion"): computed over the same label pairs, so they sum to
        // n, and the one disagreement sits off the diagonal.
        let confusion = try #require(
            byField["mentionsLegalRule"]?["confusion"]
                as? [String: [String: Int]])
        #expect(
            confusion.values.map { $0.values.reduce(0, +) }.reduce(0, +) == 2)
        let offDiagonal = confusion.flatMap { a, row in
            row.filter { $0.key != a }.values
        }.reduce(0, +)
        #expect(offDiagonal == 1)
        // No paired artifacts anywhere on this path.
        #expect(
            !FileManager.default.fileExists(
                atPath: out.appending(component: "judge-report.json").path))
        #expect(
            !FileManager.default.fileExists(
                atPath: out.appending(component: "judgments.jsonl").path))
        // The status file names the coding artifact.
        let status = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(
                        component: "run-status.json")))
                as? [String: Any])
        #expect(status["status"] as? String == "completed")
        #expect(status["itemLabel"] as? String == "coding")
        #expect(status["itemsWritten"] as? Int == 4)
    }

    /// One judge is a legal design (maintainer ruling, 2026-08-28), so the
    /// report must not carry an EMPTY `fieldAgreement` — that reads as
    /// "agreement was measured and there was none", a different and false
    /// claim. The block is absent and a reason says why. Server twin:
    /// `test_a_single_coder_report_records_agreement_as_absent_with_a_reason`.
    @Test func aSingleCoderReportRecordsAgreementAsAbsent() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "coding-solo-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.codingOverrideForTesting = { _, _ in
            "{\"codes\": {\"mentionsLegalRule\": true, "
                + "\"mentionsEquity\": false}, \"brief_reason\": \"r\"}"
        }
        defer {
            ExperimentTasks.codingOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        let study = "vg-solo-coding"
        var manifest = try ExperimentStore.create(
            name: study, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french",
                stimulusSetHash: try StimulusSet(
                    directory: VectorCatalog.conceptsDirectory
                        .appending(component: "french")
                ).hash,
                options: .init()))
        manifest.judges = [.init(name: "solo", kind: "local", model: nil)]
        try ExperimentStore.save(manifest)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260804T000000000Z-exp-\(study)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try ExperimentStore.manifestHash(manifest).write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        try ("{\"experiment\":\"\(study)\",\"condition\":\"fear\","
            + "\"seed\":2,\"promptID\":\"p0\",\"prompt\":\"q\","
            + "\"output\":\"Fairness to the donor matters most here.\"}\n")
            .write(
                to: source.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)

        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: study,
            sourceRunDirectory: source,
            evaluation: ExperimentManifest.EvaluationSpec(
                kind: .pairedJudge, judgeModel: "",
                judgePrompt: Self.codingRubric))
        let report = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(
                        component: "coding-report.json")))
                as? [String: Any])
        #expect(report["judges"] as? [String] == ["solo"])
        #expect(report["fieldAgreement"] == nil, "empty is not absent")
        #expect(
            report["fieldAgreementAbsentReason"] as? String
                == ExperimentStore.singleCoderAgreementAbsentReason)
        // The per-condition aggregates are untouched: a single coder still
        // measures, it just cannot be compared with anyone.
        let conditions = try #require(report["conditions"] as? [String: Any])
        let fear = try #require(conditions["fear"] as? [String: Any])
        #expect(fear["codings"] as? Int == 1)
    }
}

/// Serialized async fixture helpers for the retry-closure tests.
private actor AsyncQueue {
    private var items: [String]
    init(_ items: [String]) { self.items = items }
    func next() -> String {
        items.isEmpty ? "" : items.removeFirst()
    }
}

private actor InvalidLog {
    private(set) var entries: [[String: String]] = []
    func append(_ entry: [String: String]) { entries.append(entry) }
}
