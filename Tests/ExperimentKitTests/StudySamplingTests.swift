import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

struct StudySamplingTests {
    @Test func measuredRecordCarriesOperativeSeedAndPairingIndex() throws {
        var manifest = ExperimentManifest(name: "sampling", description: "", modelID: "test/model")
        manifest.temperature = 0.7
        manifest.samplesPerItem = 3
        let record = ExperimentTasks.sampledGenerationRecord(manifest: manifest,
            experimentHash: "hash", taskPromptsFile: "items.jsonl", taskPromptsHash: "inputs",
            promptMode: .chatAssistant, systemPrompt: nil, qwenThinkingEnabled: false,
            condition: "baseline", seed: UInt64.max, sampleIndex: 2, promptTokenCount: 12,
            promptIndex: 1, prompt: .init(id: "item", text: "Prompt", options: nil, target: nil,
                anchorMonths: nil, severity: nil, arm: nil, caseID: nil), output: "Answer",
            row: .init(condition: "baseline", seed: UInt64.max, promptIndex: 1,
                promptID: "item", wordCount: 1, distinct2: 0, markerDensity: [:]))
        let value = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(record)) as? [String: Any])
        #expect(value["seedInert"] as? Bool == false)
        #expect(value["sampleIndex"] as? Int == 2)
        #expect(value["promptTokenCount"] as? Int == 12)
        #expect(value["seedPolicy"] as? String == "derivedSHA256")
        #expect((value["seed"] as? NSNumber)?.stringValue == String(UInt64.max))
    }

    @Test func savedAgentThinkingRemainsIndependentOfTheBaselineEffort() {
        var manifest = ExperimentManifest(name: "sampling", description: "", modelID: "test/model")
        manifest.reasoningEffort = "off"
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: nil) == .off)
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: true) == .xhigh)
        manifest.reasoningEffort = "on"
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: false) == .off)
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: nil) == .on)
    }

    @Test func identitiesMatchThePythonContractAndPreserveLiteralSeeds() {
        #expect(StudySampling.deriveSeed(experimentHash: "abc", condition: "baseline",
            promptID: "item", sampleIndex: 0) == 4727079337508148899)
        #expect(StudySampling.deriveSeed(experimentHash: "abc", condition: "",
            promptID: "turn", sampleIndex: 3) == 5849915954937287103)
        #expect(StudySampling.deriveSeed(experimentHash: "é", condition: "条件",
            promptID: "項目🧪", sampleIndex: 3) == 4540562418375410840)
        var manifest = ExperimentManifest(name: "sampling", description: "", modelID: "test/model")
        manifest.temperature = 0.7
        manifest.seeds = [UInt64.max, 9007199254740993]
        #expect(StudySampling.policy(manifest) == "manifestSeeds")
        #expect(StudySampling.seed(manifest, experimentHash: "abc", condition: "baseline",
            promptID: "item", sampleIndex: 0) == UInt64.max)
        manifest.samplesPerItem = 3
        #expect(StudySampling.count(manifest) == 3)
        #expect(StudySampling.policy(manifest) == "derivedSHA256")
        let full = (0..<3).map { StudySampling.seed(manifest, experimentHash: "abc",
            condition: "baseline", promptID: "item", sampleIndex: $0) }
        #expect(full[0] == 4727079337508148899)
        // A resumed record is addressed by index, not a previously advanced RNG.
        #expect(full[2] == StudySampling.seed(manifest, experimentHash: "abc",
            condition: "baseline", promptID: "item", sampleIndex: 2))
        manifest.temperature = 0
        #expect(StudySampling.count(manifest) == 2)
        #expect(StudySampling.policy(manifest) == "manifestSeeds")
    }

    // MARK: - The shared seed-rule fixture

    private struct SeedFixture: Decodable {
        let experimentHash: String
        let prompts: [Prompt]
        let conditions: [String]
        let cases: [Case]

        struct Prompt: Decodable {
            let id: String
            let prompt: String
        }

        struct Case: Decodable {
            let label: String
            let manifest: Fields
            let derivesSeeds: Bool
            let seedPolicy: String
            let sampleCount: Int
            let records: [Record]
            let advisories: [String]
        }

        /// The manifest's sampling fields; an absent key is undeclared.
        struct Fields: Decodable {
            let temperature: Double
            let samplesPerItem: Int?
            let seedPolicy: String?
            let seeds: [UInt64]
        }

        struct Record: Decodable, Equatable {
            let condition: String
            let promptID: String
            let sampleIndex: Int
            let seed: UInt64
        }
    }

    /// `Tests/Fixtures/cross-engine/study-sampling-seeds.json`, produced by
    /// the Python engine's seed rule (`sampling.record_seeds`). Server twin:
    /// `Server/tests/test_study_sampling_seeds.py`.
    private func seedFixture() throws -> SeedFixture {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine", "study-sampling-seeds.json")
        return try JSONDecoder().decode(SeedFixture.self, from: Data(contentsOf: url))
    }

    private func manifest(_ fields: SeedFixture.Fields) -> ExperimentManifest {
        var manifest = ExperimentManifest(name: "seeds", description: "", modelID: "test/model")
        manifest.temperature = fields.temperature
        manifest.samplesPerItem = fields.samplesPerItem
        manifest.seedPolicy = fields.seedPolicy
        manifest.seeds = fields.seeds
        return manifest
    }

    /// Every case draws the seeds, the count, and the advisories the Python
    /// engine gives — including the one this rule exists for: a declared
    /// derivedSHA256 with one sample per item derives per-record seeds, so two
    /// items with the same text and different IDs draw different streams.
    @Test func seedRuleMatchesThePythonEngineOnTheSharedFixture() throws {
        let fixture = try seedFixture()
        #expect(fixture.cases.count >= 7)
        #expect(Set(fixture.prompts.map(\.prompt)).count == 1)
        for value in fixture.cases {
            let study = manifest(value.manifest)
            #expect(StudySampling.derivesSeeds(study) == value.derivesSeeds, "\(value.label)")
            #expect(StudySampling.policy(study) == value.seedPolicy, "\(value.label)")
            #expect(StudySampling.count(study) == value.sampleCount, "\(value.label)")
            var records: [SeedFixture.Record] = []
            for condition in fixture.conditions {
                for prompt in fixture.prompts {
                    for index in 0..<StudySampling.count(study) {
                        records.append(.init(
                            condition: condition, promptID: prompt.id, sampleIndex: index,
                            seed: StudySampling.seed(
                                study, experimentHash: fixture.experimentHash,
                                condition: condition, promptID: prompt.id,
                                sampleIndex: index)))
                    }
                }
            }
            #expect(records == value.records, "\(value.label)")
            #expect(StudySampling.unusedSeedSettings(study) == value.advisories, "\(value.label)")
        }
    }

    @Test func aDeclaredDerivedPolicyDerivesSeedsWithOneSamplePerItem() throws {
        let fixture = try seedFixture()
        let value = try #require(fixture.cases.first {
            $0.label == "derivedSHA256, one sample per item, warm, seed list"
        })
        let study = manifest(value.manifest)
        #expect(StudySampling.policy(study) == "derivedSHA256")
        #expect(StudySampling.count(study) == 1)
        let seeds = fixture.prompts.map {
            StudySampling.seed(study, experimentHash: fixture.experimentHash,
                condition: "baseline", promptID: $0.id, sampleIndex: 0)
        }
        #expect(Set(seeds).count == fixture.prompts.count)
        #expect(Set(seeds).isDisjoint(with: value.manifest.seeds))
    }

    /// A panel's turns take seeds by their own rule, so neither advisory is
    /// true of one.
    @Test func aPanelGetsNoSeedAdvisory() {
        var manifest = ExperimentManifest(name: "panel", description: "", modelID: "test/model")
        manifest.studyKind = .multiAgent
        manifest.seedPolicy = "derivedSHA256"
        manifest.seeds = [1, 2]
        #expect(StudySampling.unusedSeedSettings(manifest).isEmpty)
        manifest.temperature = 0.7
        #expect(StudySampling.unusedSeedSettings(manifest).isEmpty)
    }
}
