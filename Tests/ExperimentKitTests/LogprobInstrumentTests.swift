import Foundation
import Testing

@testable import ExperimentKit

/// Mirrors the pure-math fixtures of `Server/tests/test_logprob.py` — the
/// answer-token logprob instrument's cross-engine parity rests on both
/// substrates agreeing on these exact values. No model, no GPU (a plain
/// `swift test` cannot load the Metal shaders — CLAUDE.md › Build & run).
@Suite struct LogprobInstrumentTests {

    private func isClose(_ a: Double, _ b: Double, tolerance: Double = 1e-6) -> Bool {
        abs(a - b) <= tolerance * max(1, max(abs(a), abs(b)))
    }

    /// Reference stable log-softmax, computed independently in the test.
    private func logSoftmax(_ row: [Float]) -> [Double] {
        let peak = Double(row.max() ?? 0)
        let z = row.reduce(0.0) { $0 + exp(Double($1) - peak) }
        return row.map { Double($0) - peak - log(z) }
    }

    // MARK: - Pure scoring math (server test_logprob.py twins)

    @Test func optionScoresFromStepLogitsMultiTokenJoint() {
        var rowA = [Float](repeating: 0, count: 8)
        rowA[3] = 2.0
        var rowB = [Float](repeating: 0, count: 8)
        rowB[5] = 1.0
        let scores = LogprobInstrument.optionScores(
            options: ["yes", "no way"],
            optionTokenIDs: [[3], [4, 5]],
            stepLogits: [[rowA], [rowA, rowB]])
        let lsmA = logSoftmax(rowA)
        let lsmB = logSoftmax(rowB)
        #expect(isClose(scores[0].logprob, lsmA[3]))
        // Hand-computed: 2 − ln(7 + e²) = −0.6664679245109557.
        #expect(isClose(scores[0].logprob, -0.6664679245109557))
        let expectedJoint = lsmA[4] + lsmB[5]
        #expect(isClose(scores[1].logprob, expectedJoint))
        // Hand-computed: −2.6664679245109557 + −1.2740088362278477.
        #expect(isClose(scores[1].logprob, -3.9404767607388034))
        #expect(isClose(scores[1].meanTokenLogprob, expectedJoint / 2))
        #expect(scores[1].tokenIDs == [4, 5])
    }

    @Test func choiceResultProbabilityMarginLogOdds() throws {
        let a = OptionScore(option: "A", tokenIDs: [1], tokenLogprobs: [-1.0])
        let b = OptionScore(option: "B", tokenIDs: [2], tokenLogprobs: [-2.0])
        let result = ChoiceResult(options: [a, b])
        let probabilities = result.probability
        #expect(isClose(probabilities.values.reduce(0, +), 1.0, tolerance: 1e-9))
        let pA = try #require(probabilities["A"])
        let pB = try #require(probabilities["B"])
        #expect(pA > pB)
        #expect(result.selected == "A")
        #expect(isClose(result.margin, 1.0, tolerance: 1e-9))
        let oddsA = try #require(result.logOdds["A"])
        #expect(isClose(oddsA, log(pA / (1 - pA)), tolerance: 1e-9))
        // Record-shape helpers (as_record_fields twins).
        #expect(result.optionNames == ["A", "B"])
        #expect(result.tokenIDsByOption["B"] == [2])
        #expect(isClose(try #require(result.logprobByOption["A"]), -1.0, tolerance: 1e-9))
    }

    /// Option-length guardrail (twin of the server's
    /// `test_option_length_guardrail_fields`): counts {1, 4} → ratio 4.0.
    @Test func optionLengthGuardrailFields() {
        let a = OptionScore(option: "A", tokenIDs: [1], tokenLogprobs: [-1.0])
        let b = OptionScore(
            option: "Nebraska law governs the damages",
            tokenIDs: [2, 3, 4, 5],
            tokenLogprobs: [-0.5, -0.5, -0.5, -0.5])
        let result = ChoiceResult(options: [a, b])
        #expect(
            result.tokenCountsByOption
                == ["A": 1, "Nebraska law governs the damages": 4])
        #expect(result.optionLengthRatio == 4.0)
    }

    /// Twin of the server's `test_science_layer_tasks.py::test_option_length_gate`:
    /// equal counts pass, unequal counts throw with the shared message shape,
    /// and `acknowledgeUnequalOptionLengths` accepts the bias knowingly.
    @Test func optionLengthGateMatchesServerSemantics() throws {
        func result(_ counts: [Int]) -> ChoiceResult {
            ChoiceResult(
                options: counts.enumerated().map { index, count in
                    OptionScore(
                        option: "o\(index)",
                        tokenIDs: Array(0 ..< max(1, count)),
                        tokenLogprobs: [Float](repeating: -1.0, count: max(1, count)))
                })
        }
        var manifest = ExperimentManifest(name: "gate", description: "", modelID: "m")

        // Equal token counts: no throw.
        try ExperimentTasks.checkOptionLengths(
            result([2, 2]), manifest: manifest, promptID: "p1")

        // Unequal counts: refused, with the server's message shape.
        #expect {
            try ExperimentTasks.checkOptionLengths(
                result([1, 4]), manifest: manifest, promptID: "p1")
        } throws: { error in
            let message = "\(error)"
            return message.contains("unequal token counts")
                && message.contains("item 'p1'")
                && message.contains("acknowledgeUnequalOptionLengths")
        }

        // Acknowledged in the manifest: the bias is accepted knowingly.
        manifest.acknowledgeUnequalOptionLengths = true
        try ExperimentTasks.checkOptionLengths(
            result([1, 4]), manifest: manifest, promptID: "p1")
    }

    /// Old manifests decode without the field (nil) and the synthesized
    /// encoder omits it — content hashes of pre-existing manifests are stable.
    @Test func acknowledgeUnequalOptionLengthsRoundTripsAndOmitsNil() throws {
        let manifest = ExperimentManifest(name: "gate", description: "", modelID: "m")
        let encoded = String(
            decoding: try JSONEncoder().encode(manifest), as: UTF8.self)
        #expect(!encoded.contains("acknowledgeUnequalOptionLengths"))

        var acknowledged = manifest
        acknowledged.acknowledgeUnequalOptionLengths = true
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: try JSONEncoder().encode(acknowledged))
        #expect(decoded.acknowledgeUnequalOptionLengths == true)
    }

    // MARK: - Task-prompts loader (options + science metadata)

    @Test func taskPromptsLoaderRoundTripsScienceMetadata() throws {
        let jsonl = """
            {"id": "case-1", "prompt": "Choose the governing law.", "options": ["Apply Kansas law", "Apply Nebraska law"], "target": "Apply Kansas law", "arm": "rule", "caseID": "silicon-1"}
            {"prompt": "Sentence the defendant.", "anchorMonths": 36, "severity": 4.5}

            {"text": "legacy text-key line"}
            """
        let prompts = try ExperimentTasks.parseTaskPrompts(Data(jsonl.utf8))
        #expect(prompts.count == 3)
        #expect(prompts[0].id == "case-1")
        #expect(prompts[0].text == "Choose the governing law.")
        #expect(prompts[0].options == ["Apply Kansas law", "Apply Nebraska law"])
        #expect(prompts[0].target == "Apply Kansas law")
        #expect(prompts[0].arm == "rule")
        #expect(prompts[0].caseID == "silicon-1")
        #expect(prompts[0].anchorMonths == nil)
        #expect(prompts[1].id == "prompt-2")
        #expect(prompts[1].anchorMonths == 36)
        #expect(prompts[1].severity == 4.5)
        #expect(prompts[1].options == nil)
        #expect(prompts[1].target == nil)
        #expect(prompts[2].id == "prompt-3")
        #expect(prompts[2].text == "legacy text-key line")
    }

    @Test func taskPromptsLoaderRejectsMalformedLines() {
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.parseTaskPrompts(Data("{\"id\": \"no-text\"}".utf8))
        }
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.parseTaskPrompts(Data("not json".utf8))
        }
    }

    // MARK: - Record shapes (parity with the server's generations.jsonl)

    @Test func choiceRecordFieldNamesMatchServer() throws {
        let record = ExperimentTasks.ChoiceRecord(
            experiment: "e", experimentHash: "h", modelID: "m", modelRevision: nil,
            taskPromptsFile: "f", taskPromptsHash: "th", promptMode: "chatAssistant",
            systemPrompt: nil, qwenThinkingEnabled: false, condition: "baseline",
            promptIndex: 1, promptID: "case-1", prompt: "p",
            target: "A", targetSource: "declared",
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
            instrument: "answerTokenLogprob", options: ["A", "B"],
            optionTokenCounts: ["A": 1, "B": 1],
            optionLengthRatio: 1.0,
            optionTokenIDs: ["A": [1], "B": [2]],
            optionTokenLogprobs: ["A": [-1.0], "B": [-2.0]],
            optionLogprobs: ["A": -1.0, "B": -2.0],
            optionMeanTokenLogprobs: ["A": -1.0, "B": -2.0],
            choiceProbability: ["A": 0.7311, "B": 0.2689],
            logOdds: ["A": 1.0, "B": -1.0],
            selected: "A", margin: 1.0)
        let object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(record))
        let keys = Set(try #require(object as? [String: Any]).keys)
        // The server's `ChoiceResult.as_record_fields` names, verbatim.
        let serverInstrumentFields: Set<String> = [
            "instrument", "options", "optionTokenCounts", "optionLengthRatio",
            "optionTokenIDs", "optionTokenLogprobs",
            "optionLogprobs", "optionMeanTokenLogprobs", "choiceProbability",
            "logOdds", "selected", "margin",
        ]
        #expect(serverInstrumentFields.isSubset(of: keys))
        #expect(keys.contains("target"))
        // Open-issues #6: the record says WHERE its target came from, so a
        // consumer can tell a declared choice endpoint from a synthesized one.
        #expect(keys.contains("targetSource"))
        #expect(!keys.contains("seed"))  // temperature-free: no seed on choice records
    }

    @Test func generationRecordEncodesParseFailureAsNullAndOmitsWhenInapplicable() throws {
        func record(parsedMonths: Double??, parsedChoice: String??)
            -> ExperimentTasks.GenerationRecord
        {
            ExperimentTasks.GenerationRecord(
                experiment: "e", experimentHash: "h", modelID: "m",
                modelRevision: nil, taskPromptsFile: "f", taskPromptsHash: "th",
                promptMode: "chatAssistant", systemPrompt: nil,
                qwenThinkingEnabled: false, condition: "baseline", seed: 1,
                seedInert: true, promptIndex: 1, promptID: "case-1", prompt: "p",
                output: "o", wordCount: 1, distinct2: 1, markerDensity: [:],
                variantArtifactPath: nil, variantArtifactHash: nil,
                target: nil, anchorMonths: nil, severity: nil, arm: nil,
                caseID: nil, parsedMonths: parsedMonths, parsedChoice: parsedChoice)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // Parse failure ⇒ the key appears with JSON null (server writes
        // `"parsedMonths": null`; failure rate is a first-class endpoint).
        let failed = String(
            decoding: try encoder.encode(
                record(parsedMonths: .some(nil), parsedChoice: .some(nil))),
            as: UTF8.self)
        #expect(failed.contains("\"parsedMonths\":null"))
        #expect(failed.contains("\"parsedChoice\":null"))
        // Inapplicable (no sentencing family, no options) ⇒ keys omitted.
        let inapplicable = String(
            decoding: try encoder.encode(record(parsedMonths: nil, parsedChoice: nil)),
            as: UTF8.self)
        #expect(!inapplicable.contains("parsedMonths"))
        #expect(!inapplicable.contains("parsedChoice"))
        // Successful parse ⇒ plain value.
        let parsed = String(
            decoding: try encoder.encode(
                record(parsedMonths: .some(18.0), parsedChoice: .some("A"))),
            as: UTF8.self)
        #expect(parsed.contains("\"parsedMonths\":18"))
        #expect(parsed.contains("\"parsedChoice\":\"A\""))
    }
}
