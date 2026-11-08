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
            /// Whether the study generates sampled text: only sampled text
            /// loops over the seeds.
            let generatesSampledText: Bool
            let records: [Record]
            let advisories: [String]
            /// The run-start, submission, and draft-verify refusal, or nil.
            let refusal: String?
            /// `verify`'s samplesPerItem rules, where the run reads the count.
            let samplesPerItemProblems: [String]
            /// The advisory a study whose instruments never sample gets.
            let inertSamplingAdvisory: String?
            /// `verify`'s refusedWhenRun advisory for the same study frozen
            /// under an older release, as this command line spells it.
            let frozenAdvisory: String?
        }

        /// The manifest's sampling fields; an absent key is undeclared.
        struct Fields: Decodable {
            let temperature: Double
            let samplesPerItem: Int?
            let seedPolicy: String?
            let seeds: [UInt64]
            let outcomeInstruments: [String]?
            let pipeline: JSONValue?
        }

        /// A sampled record (`sampleIndex`, `seed`), or, for a study of
        /// readouts alone, the one readout per item it writes (`readout`).
        struct Record: Decodable, Equatable {
            let condition: String
            let promptID: String
            var sampleIndex: Int? = nil
            var seed: UInt64? = nil
            var readout: Bool? = nil
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
        manifest.outcomeInstruments = fields.outcomeInstruments
        manifest.pipeline = fields.pipeline
        return manifest
    }

    /// Every case draws the seeds, the count, and the advisories the Python
    /// engine gives — including the one this rule exists for: a declared
    /// derivedSHA256 with one sample per item derives per-record seeds, so two
    /// items with the same text and different IDs draw different streams.
    @Test func seedRuleMatchesThePythonEngineOnTheSharedFixture() throws {
        let fixture = try seedFixture()
        #expect(fixture.cases.count >= 16)
        #expect(Set(fixture.prompts.map(\.prompt)).count == 1)
        for value in fixture.cases {
            let study = manifest(value.manifest)
            let generates = ExecutionPlan.resolve(instruments: study.outcomeInstruments)
                .generatesSampledText
            #expect(StudySampling.derivesSeeds(study) == value.derivesSeeds, "\(value.label)")
            #expect(StudySampling.policy(study) == value.seedPolicy, "\(value.label)")
            #expect(
                (generates ? StudySampling.count(study) : 0) == value.sampleCount,
                "\(value.label)")
            var records: [SeedFixture.Record] = []
            for condition in fixture.conditions {
                for prompt in fixture.prompts {
                    if !generates {
                        // One readout per (condition, item), no sampled record.
                        records.append(.init(
                            condition: condition, promptID: prompt.id, readout: true))
                        continue
                    }
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
            #expect(StudySampling.greedySeedListProblem(study) == value.refusal, "\(value.label)")
            #expect(generates == value.generatesSampledText, "\(value.label)")
            #expect(
                StudySampling.samplesPerItemProblems(study) == value.samplesPerItemProblems,
                "\(value.label)")
            #expect(
                ExecutionPlan.inertSamplingAdvisory(
                    instruments: study.outcomeInstruments, temperature: study.temperature,
                    samplesPerItem: study.samplesPerItem) == value.inertSamplingAdvisory,
                "\(value.label)")
            var frozen = study
            frozen.status = .frozen
            #expect(
                StudySampling.frozenGreedySeedListAdvisory(frozen) == value.frozenAdvisory,
                "\(value.label)")
        }
    }

    /// Temperature 0 with several seeds (public issue #9) is refused under
    /// every seed policy, and no advisory calls the list one with no effect.
    @Test func aGreedyStudyWithSeveralSeedsIsRefusedAndNeverAdvised() throws {
        let refused = try seedFixture().cases.filter { $0.refusal != nil }
        #expect(Set(refused.map(\.label)) == [
            "manifestSeeds, greedy, seed list", "derivedSHA256, greedy, seed list",
            "undeclared, greedy, seed list", "readout and sampled text, greedy, seed list",
            "undeclared, greedy, seed list, chain without run",
        ])
        // Readouts alone read each item once: never refused, and told the
        // list is not used.
        let readouts = try seedFixture().cases.filter {
            !$0.generatesSampledText && $0.manifest.seeds.count > 1
        }
        #expect(readouts.count == 2)
        for value in readouts {
            #expect(value.refusal == nil)
            #expect(value.advisories.count == 1)
            #expect(value.advisories.first?.hasPrefix("the seed list (3 seeds) is not used: "
                + "this study records answer-token readouts only") == true)
        }
        for value in refused {
            #expect(value.advisories.isEmpty)
            #expect(value.refusal?.hasPrefix(StudySampling.greedySeedListPrefix) == true)
        }
        var panel = ExperimentManifest(name: "panel", description: "", modelID: "test/model")
        panel.studyKind = .multiAgent
        panel.seeds = [1, 2, 3]
        #expect(StudySampling.greedySeedListProblem(panel) == nil)
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
