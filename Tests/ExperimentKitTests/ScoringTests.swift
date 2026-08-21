import Foundation
import Testing
@testable import ExperimentKit

@Suite struct ScoringTests {

    @Test func markerRubricCountsAndDensity() {
        let rubric = MarkerRubric(words: ["food", "hungry"], characters: ["é"])
        let text = "I was so hungry that the café food vanished."
        #expect(rubric.count(in: text) == 3)  // hungry + food + é
        #expect(rubric.density(in: text) > 0)
        #expect(rubric.count(in: "Nothing relevant here.") == 0)
    }

    @Test func markerRubricLoadsFromJSON() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "rubric-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"words": ["alpha", "beta"], "characters": "ß"}"#.write(
            to: dir.appending(component: "markers.json"), atomically: true, encoding: .utf8)

        let rubric = MarkerRubric(directory: dir)
        #expect(rubric != nil)
        #expect(rubric?.count(in: "Alpha meets beta straße.") == 3)
        #expect(MarkerRubric(directory: dir.appending(component: "missing")) == nil)
    }

    @Test func batteryScoring() {
        #expect(CapabilityBattery.isCorrect(response: "The answer is 43.", answer: "43"))
        #expect(!CapabilityBattery.isCorrect(response: "I am not sure.", answer: "43"))
        #expect(CapabilityBattery.isCorrect(response: "PARIS, of course", answer: "paris"))
        #expect(!CapabilityBattery.isCorrect(response: "It took 20 minutes.", answer: "2"))
        #expect(!CapabilityBattery.isCorrect(response: "another possibility", answer: "no"))
        #expect(CapabilityBattery.isCorrect(response: "No.", answer: "no"))
        #expect(CapabilityBattery.isCorrect(response: "0.70", answer: "0.7"))
        #expect(!CapabilityBattery.isCorrect(response: "0.07", answer: "0.7"))
        #expect(!CapabilityBattery.isCorrect(response: "72? No, it's 81.", answer: "72"))
        #expect(CapabilityBattery.isCorrect(response: "17 + 26 = 43.", answer: "43"))
        #expect(!CapabilityBattery.isCorrect(response: "43, then drifted to 19", answer: "43"))
        #expect(CapabilityBattery.isCorrect(response: "Final answer: 43. Confidence 0.8.", answer: "43"))
    }

    /// Cross-engine grading vocabulary pin (2026-07-20, mirrored as
    /// `SWIFT_GRADING_MODES` in the server's `test_battery_in_run.py`):
    /// the server's battery loader accepts EXACTLY the raw values this
    /// enum decodes, so a change on either side must be a deliberate
    /// contract change. The exhaustive switch (no default) makes an added
    /// case a compile error here — the pin can't silently drift.
    @Test func gradingVocabularyIsPinnedCrossEngine() {
        let vocabulary = [
            "exact_number", "yes_no", "token_exact", "exact_normalized",
            "regex",
        ]
        let modes: [CapabilityBattery.GradingMode] = [
            .exactNumber, .yesNo, .tokenExact, .exactNormalized, .regex,
        ]
        #expect(modes.map(\.rawValue) == vocabulary)
        for raw in vocabulary {
            #expect(CapabilityBattery.GradingMode(rawValue: raw) != nil)
        }
        #expect(CapabilityBattery.GradingMode(rawValue: "made_up") == nil)
        for mode in modes {
            switch mode {  // exhaustive on purpose — see doc comment
            case .exactNumber, .yesNo, .tokenExact, .exactNormalized,
                .regex:
                break
            }
        }
        // The junk record the server round reproduced refuses to decode
        // here too (string prompt/answer + known grading are the
        // contract on BOTH engines).
        let decoder = JSONDecoder()
        #expect(
            (try? decoder.decode(
                CapabilityBattery.Item.self,
                from: Data(
                    #"{"prompt":123,"answer":7,"grading":"made_up"}"#.utf8)))
                == nil)
        #expect(
            (try? decoder.decode(
                CapabilityBattery.Item.self,
                from: Data(
                    #"{"prompt":"2+2?","answer":"4","grading":"made_up"}"#
                        .utf8)))
                == nil)
    }

    @Test func batteryScoringModes() {
        #expect(
            CapabilityBattery.isCorrect(
                response: "final answer: B", answer: #"\bB\b"#, grading: .regex))
        #expect(
            !CapabilityBattery.isCorrect(
                response: "final answer: boat", answer: #"\bB\b"#, grading: .regex))
        #expect(
            CapabilityBattery.isCorrect(
                response: "The Eiffel Tower", answer: "the eiffel tower",
                grading: .exactNormalized))
        #expect(
            !CapabilityBattery.isCorrect(
                response: "The Eiffel Tower is in Paris", answer: "the eiffel tower",
                grading: .exactNormalized))
    }

    @Test func distinctBigramsDetectDegeneration() {
        let healthy = "The garden needs sun water and patience to grow well each season"
        let degenerate = "je suis je suis je suis je suis je suis je suis je suis"
        #expect(distinctBigramRatio(healthy) > 0.9)
        #expect(distinctBigramRatio(degenerate) < 0.3)
        #expect(distinctBigramRatio("Yes.") == 0)
    }

    @Test func robustnessPresetFilesLoad() throws {
        struct PromptRow: Decodable {
            let text: String?
            let prompt: String?
        }

        // Anchor to the bundled SEED root: the preset files are repo-shipped
        // seed data, and this suite runs in PARALLEL with serialized suites
        // that temporarily override the workspace root — projectRoot can
        // point at a half-built temp workspace mid-test (observed flake),
        // while the seed root is stable under xcodebuild's /private/tmp cwd
        // and immune to workspace overrides.
        let root = VectorCatalog.bundledSeedRoot
        for preset in VariantRobustness.presets {
            let battery = try CapabilityBattery(url: root.appending(path: preset.batteryFile))
            #expect(!battery.items.isEmpty)
            let lines = String(
                decoding: try Data(contentsOf: root.appending(path: preset.coherencePromptsFile)),
                as: UTF8.self
            )
            .split(separator: "\n", omittingEmptySubsequences: true)
            let prompts = try lines.map {
                try JSONDecoder().decode(PromptRow.self, from: Data($0.utf8))
            }
            #expect(prompts.contains { ($0.text ?? $0.prompt ?? "").isEmpty == false })
            #expect(preset.maxCoherencePrompts > 0)
            #expect(preset.maxTokens > 0)
        }
    }
}
