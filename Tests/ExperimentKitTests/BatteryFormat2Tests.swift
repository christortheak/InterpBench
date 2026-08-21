import Foundation
import Testing

@testable import ExperimentKit

/// Battery format 2 on this engine: PINNABLE (the Mac is the authoring
/// surface) and, since 2026-08-19, RUNNABLE — the local runner parses the
/// header, arms the reading from the FILE, and scores choice items through
/// the answer-token logprob instrument. These tests pin the pin-time shape
/// rules; `BatteryTwinTests` pins the loading, arming and scoring behaviour.
@Suite struct BatteryFormat2Tests {

    private let header =
        #"{"batteryFormat":2,"scoring":"choiceProbability","maxTokens":8,"#
        + #""promptMode":"chatAssistant","qwenThinkingEnabled":false}"#

    private func file(_ lines: String...) -> Data {
        Data(lines.joined(separator: "\n").utf8)
    }

    @Test func aValidV2BatteryPins() {
        let problem = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                header,
                #"{"id":"cap-1","prompt":"Capital of France?","answer":"Paris","options":["Paris","Rome","Berlin"]}"#,
                #"{"id":"cap-2","prompt":"2+2?","answer":"4","options":["3","4"]}"#
            ),
            file: "prompts/batteries/repaired.jsonl")
        #expect(problem == nil, Comment(rawValue: problem ?? ""))
    }

    @Test func aV2FileHashesLikeAnyOtherPinnedInput() {
        // Byte-hashing is format-agnostic by construction; asserted so the
        // "v2 pins stably" claim is mechanical rather than assumed.
        let bytes = file(header, #"{"prompt":"p","answer":"a","options":["a","b"]}"#)
        #expect(
            ExperimentStore.sha256Hex(bytes) == ExperimentStore.sha256Hex(bytes))
        #expect(ExperimentStore.sha256Hex(bytes).count == 64)
    }

    @Test func aMalformedHeaderRefuses() {
        // Declaring format 1 in a header is exactly the case that would
        // change what an existing legacy hash means.
        let one = PinShapeValidation.capabilityBatteryShapeProblem(
            file(#"{"batteryFormat":1}"#, #"{"prompt":"p","answer":"a"}"#),
            file: "b.jsonl")
        #expect(one?.contains("batteryFormat") == true)

        let unknownScoring = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                #"{"batteryFormat":2,"scoring":"vibes"}"#,
                #"{"prompt":"p","answer":"a","options":["a","b"]}"#),
            file: "b.jsonl")
        #expect(unknownScoring?.contains("scoring") == true)

        let badTokens = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                #"{"batteryFormat":2,"maxTokens":0}"#,
                #"{"prompt":"p","answer":"a","options":["a","b"]}"#),
            file: "b.jsonl")
        #expect(badTokens?.contains("maxTokens") == true)

        let headerOnly = PinShapeValidation.capabilityBatteryShapeProblem(
            file(header), file: "b.jsonl")
        #expect(headerOnly?.contains("no items") == true)
    }

    @Test func aChoiceItemWithoutOptionsRefuses() {
        let problem = PinShapeValidation.capabilityBatteryShapeProblem(
            file(header, #"{"prompt":"Capital of France?","answer":"Paris"}"#),
            file: "b.jsonl")
        #expect(problem?.contains("options") == true)
        #expect(problem?.contains("line 2") == true)
    }

    @Test func anAnswerOutsideItsOptionsRefuses() {
        // The item could never be scored correct, so the capability CONTROL
        // would read a depressed accuracy under every condition.
        let problem = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                header,
                #"{"prompt":"Capital of France?","answer":"Paris","options":["Rome","Berlin"]}"#),
            file: "b.jsonl")
        #expect(problem?.contains("never be scored correct") == true)
    }

    @Test func aGeneratedTextItemMustDeclareItsGrading() {
        let missing = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                #"{"batteryFormat":2,"scoring":"generatedText"}"#,
                #"{"prompt":"p","answer":"4"}"#),
            file: "b.jsonl")
        #expect(missing?.contains("grading") == true)

        let declared = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                #"{"batteryFormat":2,"scoring":"generatedText"}"#,
                #"{"prompt":"p","answer":"4","grading":"exact_number"}"#),
            file: "b.jsonl")
        #expect(declared == nil, Comment(rawValue: declared ?? ""))
    }

    @Test func legacyBatteriesPinExactlyAsBefore() {
        let ok = PinShapeValidation.capabilityBatteryShapeProblem(
            file(
                #"{"prompt":"Capital of France?","answer":"Paris"}"#,
                #"{"prompt":"2+2?","answer":"4","grading":"exact_number"}"#),
            file: "prompts/batteries/basic.jsonl")
        #expect(ok == nil, Comment(rawValue: ok ?? ""))

        let bad = PinShapeValidation.capabilityBatteryShapeProblem(
            file(#"{"prompt":"p"}"#), file: "b.jsonl")
        #expect(bad?.contains("line 1") == true)
    }

    /// Was `theLocalRunnerRefusesAV2BatteryByName` (flipped 2026-08-19, the
    /// twin's whole point): the local runner now LOADS a v2 battery and
    /// scores it by choice probability — the answer decides correctness, and
    /// the reading is armed by the file, not by whoever loaded it.
    @Test func theLocalRunnerScoresAV2Battery() async throws {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "battery-\(UUID().uuidString).jsonl")
        try file(
            header,
            #"{"prompt":"Capital of France?","answer":"Paris","options":["Paris","Rome","Berlin"]}"#
        ).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let battery = try CapabilityBattery(url: url)
        #expect(battery.formatVersion == 2)
        #expect(battery.isolated)
        #expect(battery.maxTokens == 8)
        #expect(battery.scoring(for: battery.items[0]) == .choiceProbability)

        // Armed by the FILE even under a hostile surrounding instrument.
        let arming = battery.resolveArming(
            promptMode: .rawCompletion,
            systemPrompt: "Respond only in JSON.",
            qwenThinkingEnabled: true)
        #expect(arming.isolated)
        #expect(arming.systemPrompt == nil)
        #expect(arming.maxTokens == 8)

        let score = try await battery.scoreItem(
            battery.items[0], arming: arming,
            generate: { _, _ in
                Issue.record("a choice item must not generate")
                return ""
            },
            choice: { _, options, _ in
                ("Paris", Dictionary(
                    uniqueKeysWithValues: options.map {
                        ($0, $0 == "Paris" ? 0.8 : 0.1)
                    }))
            })
        #expect(score.correct)
        #expect(score.selected == "Paris")
        #expect(score.output == "Paris")
        #expect(score.choiceProbability?["Paris"] == 0.8)
    }

    @Test func theLocalRunnerStillLoadsLegacyBatteries() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "battery-\(UUID().uuidString).jsonl")
        try file(
            #"{"prompt":"Capital of France?","answer":"Paris"}"#,
            #"{"prompt":"2+2?","answer":"4","grading":"exact_number"}"#
        ).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let battery = try CapabilityBattery(url: url)
        #expect(battery.items.count == 2)
        #expect(battery.items[1].grading == .exactNumber)
    }
}
