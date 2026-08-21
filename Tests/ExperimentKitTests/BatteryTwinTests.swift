import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The Swift battery twin: loading, ARMING and scoring, mirroring the
/// server's `tests/test_battery_format_v2.py` case for case. The engine that
/// defines the contract is the Python one — every expectation here is the
/// behaviour `Server/steerlab_server/experiment/battery.py` already has.
///
/// Numbers come from fixtures and injected back-ends, never a live model
/// (CLAUDE.md › conventions): both scoring back-ends are closures for exactly
/// that reason, on both engines.
@Suite struct BatteryTwinTests {

    private static let hostile = "You are a federal district judge. Respond in JSON."

    private func battery(_ lines: String..., file: String = "b.jsonl") throws
        -> CapabilityBattery
    {
        try CapabilityBattery(
            data: Data(lines.joined(separator: "\n").utf8), file: file)
    }

    private var v2Header: String {
        #"{"batteryFormat":2,"scoring":"choiceProbability","maxTokens":8,"#
            + #""promptMode":"chatAssistant","qwenThinkingEnabled":false}"#
    }

    private var choiceItem: String {
        #"{"id":"cap-1","prompt":"Capital of France?","answer":"Paris",""#
            + #"options":["Paris","Rome","Berlin"]}"#
    }

    /// Back-ends whose GENERATION obeys the arming's system prompt (as a live
    /// model does) while the choice reader does not — the asymmetry the whole
    /// repair rests on (server `_fake_backends`).
    private func backends(
        seen: Recorder? = nil
    ) -> (
        generate: (String, BatteryArming) async throws -> String,
        choice: (String, [String], BatteryArming) async throws -> (
            selected: String, probability: [String: Double]
        )
    ) {
        (
            generate: { prompt, arming in
                await seen?.note(kind: "generate", arming: arming)
                if arming.systemPrompt?.isEmpty == false {
                    return #"```json{"case_determination": {"issue": "Choice of Law""#
                }
                return prompt.contains("capital") || prompt.contains("Capital")
                    ? "Paris" : "43"
            },
            choice: { prompt, options, arming in
                await seen?.note(kind: "choice", arming: arming)
                let correct =
                    prompt.contains("capital") || prompt.contains("Capital")
                    ? "Paris" : "43"
                return (
                    correct,
                    Dictionary(
                        uniqueKeysWithValues: options.map {
                            ($0, $0 == correct ? 0.8 : 0.1)
                        })
                )
            }
        )
    }

    actor Recorder {
        private(set) var kinds: [String] = []
        private(set) var armings: [BatteryArming] = []
        func note(kind: String, arming: BatteryArming) {
            kinds.append(kind)
            armings.append(arming)
        }
    }

    // MARK: - Loading

    @Test func aLegacyFileLoadsWithTheHistoricalItemShape() throws {
        let legacy = try battery(
            #"{"prompt":"What is the capital of France?","answer":"paris","grading":"token_exact"}"#,
            #"{"prompt":"What is 17 + 26?","answer":"43","grading":"exact_number"}"#)
        #expect(legacy.formatVersion == CapabilityBattery.legacyFormat)
        #expect(!legacy.isolated)
        #expect(legacy.defaultScoring == .generatedText)
        #expect(legacy.maxTokens == VariantRobustness.batteryMaxTokens)
        #expect(legacy.systemPrompt == nil)
        // The format-2 keys are simply absent — nothing is invented.
        #expect(legacy.items.allSatisfy { $0.id == nil && $0.options == nil })
        #expect(legacy.items.allSatisfy { legacy.scoring(for: $0) == .generatedText })
    }

    @Test func aV2HeaderDeclaresArmingAndScoring() throws {
        let loaded = try battery(
            #"{"batteryFormat":2,"maxTokens":12,"promptMode":"rawCompletion","#
                + #""systemPrompt":"Answer plainly.","qwenThinkingEnabled":true}"#,
            choiceItem)
        #expect(loaded.formatVersion == CapabilityBattery.currentFormat)
        #expect(loaded.isolated)
        // The header's own defaults: scoring defaults to choiceProbability.
        #expect(loaded.defaultScoring == .choiceProbability)
        #expect(loaded.maxTokens == 12)
        #expect(loaded.promptMode == .rawCompletion)
        #expect(loaded.systemPrompt == "Answer plainly.")
        #expect(loaded.qwenThinkingEnabled)
        #expect(loaded.items[0].id == "cap-1")
        #expect(loaded.items[0].options == ["Paris", "Rome", "Berlin"])
    }

    @Test(arguments: [
        (#"{"batteryFormat":3}"#, "batteryFormat"),
        (#"{"batteryFormat":1}"#, "batteryFormat"),
        (#"{"batteryFormat":2,"scoring":"vibes"}"#, "scoring"),
        (#"{"batteryFormat":2,"maxTokens":0}"#, "maxTokens"),
        (#"{"batteryFormat":2,"maxTokens":true}"#, "maxTokens"),
        (#"{"batteryFormat":2,"promptMode":""}"#, "promptMode"),
        (#"{"batteryFormat":2,"systemPrompt":5}"#, "systemPrompt"),
        (#"{"batteryFormat":2,"qwenThinkingEnabled":"yes"}"#, "qwenThinkingEnabled"),
    ])
    func aJunkHeaderRefusesNamingWhy(header: String, fragment: String) throws {
        // The RUNNER refuses with the same message the pin refuses with —
        // one validator, so "it pinned but would not run" cannot happen.
        do {
            _ = try battery(header, choiceItem)
            Issue.record("the runner accepted header \(header)")
        } catch let error as ExperimentError {
            #expect(error.reason.contains(fragment))
        }
    }

    @Test(arguments: [
        (#"{"prompt":"Capital?","answer":"Paris"}"#, "options"),
        (#"{"prompt":"Capital?","answer":"Paris","options":["Rome","Berlin"]}"#,
            "never be scored correct"),
        (#"{"prompt":"Capital?","answer":"Paris","options":["Paris","Paris"]}"#,
            "repeats an option"),
        (#"{"prompt":"","answer":"Paris","options":["Paris","Rome"]}"#, "prompt"),
        (#"{"prompt":"Capital?","answer":"Paris","options":["Paris",""]}"#, "options"),
    ])
    func aV2ItemRefusesUndecidableShapes(item: String, fragment: String) throws {
        do {
            _ = try battery(v2Header, item)
            Issue.record("the runner accepted item \(item)")
        } catch let error as ExperimentError {
            #expect(error.reason.contains(fragment))
        }
    }

    @Test func aV2GeneratedTextItemMustDeclareItsGrading() throws {
        do {
            _ = try battery(
                #"{"batteryFormat":2,"scoring":"generatedText"}"#,
                #"{"prompt":"2+2?","answer":"4"}"#)
            Issue.record("the runner inferred a grading mode for a v2 item")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("grading"))
        }
        let declared = try battery(
            #"{"batteryFormat":2,"scoring":"generatedText"}"#,
            #"{"prompt":"2+2?","answer":"4","grading":"exact_number"}"#)
        #expect(declared.items[0].grading == .exactNumber)
        #expect(declared.scoring(for: declared.items[0]) == .generatedText)
    }

    @Test func aLegacyBatteryStillRefusesJunkTheSameWay() throws {
        // Unchanged failure MODE for format 1: the historical malformed-line
        // error, naming the line, not the new ExperimentError wording.
        do {
            _ = try battery(#"{"prompt":"only a prompt"}"#, file: "legacy.jsonl")
            Issue.record("the legacy loader accepted an answerless item")
        } catch let error as StimulusSetError {
            #expect("\(error)".contains("legacy.jsonl"))
        }
    }

    // MARK: - Arming isolation (the substantive behaviour)

    @Test func legacyArmingStillTakesTheInstrumentsContext() throws {
        let legacy = try battery(
            #"{"prompt":"What is the capital of France?","answer":"paris","grading":"token_exact"}"#)
        let arming = legacy.resolveArming(
            promptMode: .chatAssistant, systemPrompt: Self.hostile,
            qwenThinkingEnabled: true)
        #expect(arming.systemPrompt == Self.hostile)
        #expect(arming.qwenThinkingEnabled)
        #expect(arming.maxTokens == VariantRobustness.batteryMaxTokens)
        #expect(!arming.isolated)
        let advisory = legacy.contaminationAdvisory(arming)
        #expect(advisory?.contains("not comparable across instruments") == true)
    }

    @Test func v2ArmingIgnoresTheInstrumentEntirely() throws {
        let loaded = try battery(v2Header, choiceItem)
        let hostile = loaded.resolveArming(
            promptMode: .rawCompletion, systemPrompt: Self.hostile,
            qwenThinkingEnabled: true)
        let clean = loaded.resolveArming()
        #expect(hostile == clean)
        #expect(hostile.systemPrompt == nil)
        #expect(hostile.maxTokens == 8)
        #expect(hostile.promptMode == .chatAssistant)
        #expect(hostile.isolated)
        #expect(loaded.contaminationAdvisory(hostile) == nil)
    }

    @Test func aCleanLegacyReadingEarnsNoAdvisory() throws {
        let legacy = try battery(#"{"prompt":"2+2?","answer":"4"}"#)
        let arming = legacy.resolveArming(promptMode: .chatAssistant)
        #expect(legacy.contaminationAdvisory(arming) == nil)
    }

    @Test func v2ScoringIsInvariantToTheSurroundingInstrument() async throws {
        // THE property. Same battery, wildly different surrounding
        // instruments — identical per-item results, and the CHOICE reader is
        // what ran (nothing was generated, so nothing could be graded on
        // format compliance).
        let loaded = try battery(
            #"{"batteryFormat":2,"maxTokens":4}"#, choiceItem,
            #"{"prompt":"What is 17 + 26?","answer":"43","options":["41","42","43"]}"#)
        let instruments: [(ExperimentManifest.PromptMode?, String?, Bool)] = [
            (nil, nil, false),
            (.rawCompletion, Self.hostile, true),
            (.chatAssistant, "Write 500 words of poetry, always.", false),
        ]
        var results: [[Bool]] = []
        for (mode, system, thinking) in instruments {
            let recorder = Recorder()
            let back = backends(seen: recorder)
            let arming = loaded.resolveArming(
                promptMode: mode, systemPrompt: system,
                qwenThinkingEnabled: thinking)
            var row: [Bool] = []
            for item in loaded.items {
                row.append(
                    try await loaded.scoreItem(
                        item, arming: arming,
                        generate: back.generate, choice: back.choice
                    ).correct)
            }
            results.append(row)
            #expect(await recorder.kinds == ["choice", "choice"])
            #expect(
                await recorder.armings.allSatisfy {
                    $0.systemPrompt == nil && $0.maxTokens == 4
                })
        }
        #expect(results[0] == results[1])
        #expect(results[1] == results[2])
        #expect(results[0] == [true, true])
    }

    @Test func theLegacyReadingStaysInstrumentDependent() async throws {
        // The legacy defect, reproduced at unit scale and DELIBERATELY
        // preserved: identical battery, identical model, two accuracies —
        // 1.00 clean, 0.00 under the study's system prompt.
        let legacy = try battery(
            #"{"prompt":"What is the capital of France?","answer":"paris","grading":"token_exact"}"#,
            #"{"prompt":"What is 17 + 26?","answer":"43","grading":"exact_number"}"#)
        let back = backends()
        var scored: [Bool] = []
        for system in [nil, Self.hostile] as [String?] {
            let arming = legacy.resolveArming(systemPrompt: system)
            for item in legacy.items {
                scored.append(
                    try await legacy.scoreItem(
                        item, arming: arming,
                        generate: back.generate, choice: back.choice
                    ).correct)
            }
        }
        #expect(scored == [true, true, false, false])
    }

    // MARK: - Scoring

    @Test func aChoiceItemRecordsTheDistributionAndScoresByArgmax() async throws {
        let loaded = try battery(v2Header, choiceItem)
        let back = backends()
        let score = try await loaded.scoreItem(
            loaded.items[0], arming: loaded.resolveArming(),
            generate: back.generate, choice: back.choice)
        #expect(score.correct)
        #expect(score.scoring == .choiceProbability)
        #expect(score.selected == "Paris")
        #expect(score.output == "Paris")
        #expect(score.options == ["Paris", "Rome", "Berlin"])
        #expect(score.choiceProbability?["Paris"] == 0.8)
    }

    @Test func aWrongSelectionScoresIncorrectWithoutGenerating() async throws {
        let loaded = try battery(v2Header, choiceItem)
        let score = try await loaded.scoreItem(
            loaded.items[0], arming: loaded.resolveArming(),
            generate: { _, _ in
                Issue.record("a choice item must not generate")
                return ""
            },
            choice: { _, options, _ in
                ("Rome", Dictionary(
                    uniqueKeysWithValues: options.map {
                        ($0, $0 == "Rome" ? 0.6 : 0.2)
                    }))
            })
        #expect(!score.correct)
        #expect(score.selected == "Rome")
    }

    @Test func aV2GeneratedItemStillTextMatchesUnderItsDeclaredGrading()
        async throws
    {
        let loaded = try battery(
            #"{"batteryFormat":2,"scoring":"generatedText"}"#,
            #"{"id":"cap","prompt":"What is the capital of France?","answer":"paris","grading":"token_exact"}"#)
        let back = backends()
        let score = try await loaded.scoreItem(
            loaded.items[0], arming: loaded.resolveArming(),
            generate: back.generate, choice: back.choice)
        #expect(score.scoring == .generatedText)
        #expect(score.output == "Paris")
        #expect(score.correct)
        #expect(score.options == nil && score.selected == nil)
        #expect(score.choiceProbability == nil)
    }

    // MARK: - Record shape (cross-engine keys)

    private func record(
        _ battery: CapabilityBattery, _ score: BatteryItemScore
    ) throws -> [String: Any] {
        let record = ExperimentTasks.batteryRecord(
            condition: "steered", promptIndex: 1, item: battery.items[0],
            score: score, battery: battery,
            arming: battery.resolveArming(
                promptMode: .chatAssistant, systemPrompt: Self.hostile))
        let data = try JSONEncoder().encode(record)
        return try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func aLegacyRecordKeepsExactlyItsHistoricalKeys() throws {
        let legacy = try battery(#"{"prompt":"2+2?","answer":"4"}"#)
        let json = try record(
            legacy,
            BatteryItemScore(
                scoring: .generatedText, output: "4", correct: true))
        #expect(
            Set(json.keys) == [
                "condition", "promptIndex", "prompt", "expected", "output",
                "correct",
            ])
    }

    @Test func aV2RecordCarriesTheServersKeyNames() throws {
        let loaded = try battery(v2Header, choiceItem)
        let json = try record(
            loaded,
            BatteryItemScore(
                scoring: .choiceProbability,
                options: ["Paris", "Rome", "Berlin"],
                choiceProbability: ["Paris": 0.8, "Rome": 0.1, "Berlin": 0.1],
                selected: "Paris", output: "Paris", correct: true))
        // Server `battery.score_item` + `BatteryArming.as_record_fields`.
        #expect(
            Set(json.keys) == [
                "condition", "promptIndex", "prompt", "expected", "output",
                "correct", "batteryFormat", "scoring", "options",
                "choiceProbability", "selected", "armingIsolated",
                "armingPromptMode", "armingSystemPrompt", "armingMaxTokens",
            ])
        #expect(json["scoring"] as? String == "choiceProbability")
        #expect(json["selected"] as? String == "Paris")
        #expect(json["batteryFormat"] as? Int == 2)
        // Arming provenance is the BATTERY's, not the caller's hostile
        // instrument, and only the system prompt's PRESENCE is stamped.
        #expect(json["armingIsolated"] as? Bool == true)
        #expect(json["armingPromptMode"] as? String == "chatAssistant")
        #expect(json["armingSystemPrompt"] as? Bool == false)
        #expect(json["armingMaxTokens"] as? Int == 8)
    }

    @Test func theSummaryIsTheMeanOfTheReadings() throws {
        let scores = [
            BatteryItemScore(scoring: .choiceProbability, output: "a", correct: true),
            BatteryItemScore(scoring: .choiceProbability, output: "b", correct: false),
            BatteryItemScore(scoring: .choiceProbability, output: "c", correct: true),
            BatteryItemScore(scoring: .choiceProbability, output: "d", correct: true),
        ]
        let summary = ExperimentTasks.batterySummary(scores, batteryHash: "feedface")
        #expect(summary.itemCount == 4)
        #expect(summary.accuracy == 0.75)
        #expect(summary.batteryHash == "feedface")
    }

    @Test func theRecordLayerIsPureAndOrdersRowsWithTheItems() throws {
        let loaded = try battery(
            v2Header, choiceItem,
            #"{"prompt":"What is 17 + 26?","answer":"43","options":["41","42","43"]}"#)
        let arming = loaded.resolveArming()
        let scored = ExperimentTasks.scoreBatteryReadings(
            condition: "steered", battery: loaded, arming: arming,
            scores: [
                BatteryItemScore(
                    scoring: .choiceProbability, options: ["Paris", "Rome", "Berlin"],
                    choiceProbability: ["Paris": 0.8], selected: "Paris",
                    output: "Paris", correct: true),
                BatteryItemScore(
                    scoring: .choiceProbability, options: ["41", "42", "43"],
                    choiceProbability: ["41": 0.5], selected: "41",
                    output: "41", correct: false),
            ],
            batteryHash: "abc")
        #expect(scored.records.map(\.promptIndex) == [1, 2])
        #expect(scored.records.map(\.expected) == ["Paris", "43"])
        #expect(scored.records.map(\.correct) == [true, false])
        #expect(scored.summary.accuracy == 0.5)
        #expect(scored.summary.batteryHash == "abc")
    }
}
