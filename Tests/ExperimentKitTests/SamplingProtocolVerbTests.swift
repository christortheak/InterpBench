import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// `experiment set-sampling` + `experiment set-exclusions` — the writers for
/// the six manifest protocol fields that had none on either CLI
/// (`temperature`, `maxTokens`, `promptMode`, `samplesPerItem`, `seedPolicy`,
/// `exclusionRules`).
///
/// The gap was field-discovered: a stochastic replication arm (25 samples ×
/// T=0.7 × 1024 tokens) could not be authored headlessly — the fields were
/// writable only from the Study Setup panel and the SwiftUI exclusion-rules
/// editor — and was cut from a study design. The server twin is
/// `experiment_store.set_protocol`, whose vocabulary gained
/// `samplesPerItem`/`seedPolicy` in the same change; the refusal sentences
/// asserted here are the cross-engine contract.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct SamplingProtocolVerbTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "sampling-verb-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func draft(_ name: String = "d") throws -> ExperimentManifest {
        try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding)
            .run(namespace: "experiment", args)
    }

    func lifecycleGate(_ error: any Error) -> LifecycleGate? {
        (error as? ExperimentError)?.lifecycleRefusal?.gate
    }

    // MARK: - setSamplingProtocol: merge semantics

    /// nil = leave the field as it stands. A merge setter that cleared
    /// undeclared fields would make `--temperature 0.7` alone silently reset
    /// the token budget nobody typed.
    @Test func onlyTheParametersGivenMove() async throws {
        try await withTempRoot { _ in
            try draft()
            let before = try ExperimentStore.load(name: "d")
            let first = try ExperimentStore.setSamplingProtocol(
                temperature: 0.7, experimentName: "d")
            #expect(first.temperature == 0.7)
            #expect(first.maxTokens == before.maxTokens)
            #expect(first.samplesPerItem == before.samplesPerItem)
            let second = try ExperimentStore.setSamplingProtocol(
                samplesPerItem: 25, seedPolicy: "derivedSHA256",
                experimentName: "d")
            #expect(second.temperature == 0.7)  // the earlier write survives
            #expect(second.samplesPerItem == 25)
            #expect(second.seedPolicy == "derivedSHA256")
        }
    }

    /// "" clears promptMode/seedPolicy to ABSENT; samplesPerItem 1 clears to
    /// the deterministic default — the same content-hash hygiene every
    /// science-manifest setter keeps.
    @Test func emptyAndUnitValuesClearToAbsent() async throws {
        try await withTempRoot { _ in
            try draft()
            try ExperimentStore.setSamplingProtocol(
                promptMode: "rawCompletion", samplesPerItem: 3,
                seedPolicy: "manifestSeeds", experimentName: "d")
            var loaded = try ExperimentStore.load(name: "d")
            #expect(loaded.promptMode == .rawCompletion)
            #expect(loaded.samplesPerItem == 3)
            #expect(loaded.seedPolicy == "manifestSeeds")
            try ExperimentStore.setSamplingProtocol(
                promptMode: "", samplesPerItem: 1, seedPolicy: "",
                experimentName: "d")
            loaded = try ExperimentStore.load(name: "d")
            #expect(loaded.promptMode == nil)
            #expect(loaded.samplesPerItem == nil)
            #expect(loaded.seedPolicy == nil)
        }
    }

    // MARK: - setSamplingProtocol: value refusals (malformed, not gates)

    @Test func outOfVocabularyValuesAreMalformedAndWriteNothing() async throws {
        try await withTempRoot { _ in
            try draft()
            let attempts: [() throws -> ExperimentManifest] = [
                {
                    try ExperimentStore.setSamplingProtocol(
                        temperature: -0.5, experimentName: "d")
                },
                {
                    try ExperimentStore.setSamplingProtocol(
                        maxTokens: 0, experimentName: "d")
                },
                {
                    try ExperimentStore.setSamplingProtocol(
                        promptMode: "freestyle", experimentName: "d")
                },
                {
                    try ExperimentStore.setSamplingProtocol(
                        samplesPerItem: 0, experimentName: "d")
                },
                {
                    try ExperimentStore.setSamplingProtocol(
                        seedPolicy: "diceRoll", experimentName: "d")
                },
            ]
            for attempt in attempts {
                #expect { try attempt() } throws: { error in
                    (error as? ExperimentError)?.malformedInvocation != nil
                }
            }
            // The refusal sentences carry the vocabulary (help and refusal
            // read the same constants, so they cannot disagree).
            #expect {
                try ExperimentStore.setSamplingProtocol(
                    promptMode: "freestyle", experimentName: "d")
            } throws: { error in
                let reason = (error as? ExperimentError)?.reason ?? ""
                return reason
                    == "unknown promptMode 'freestyle' — known: "
                    + "chatAssistant, rawCompletion"
            }
            #expect {
                try ExperimentStore.setSamplingProtocol(
                    seedPolicy: "diceRoll", experimentName: "d")
            } throws: { error in
                let reason = (error as? ExperimentError)?.reason ?? ""
                return reason
                    == "unknown seedPolicy 'diceRoll' — known: "
                    + "manifestSeeds, derivedSHA256"
            }
            // Nothing moved.
            let loaded = try ExperimentStore.load(name: "d")
            #expect(loaded.temperature == 0)
            #expect(loaded.samplesPerItem == nil)
            #expect(loaded.seedPolicy == nil)
        }
    }

    /// The full-state panel setter now speaks the same classification: its
    /// two historical refusals were plain errors (`verbFailed`/70 territory),
    /// upgraded to malformed with the byte-identical reason.
    @Test func theFullStatePanelSetterRefusesTheSameWay() async throws {
        try await withTempRoot { _ in
            try draft()
            #expect {
                try ExperimentStore.setSamplingPolicy(
                    samplesPerItem: 0, seedPolicy: nil, experimentName: "d")
            } throws: { error in
                let typed = error as? ExperimentError
                return typed?.malformedInvocation != nil
                    && typed?.reason == "samplesPerItem must be ≥ 1 — got 0"
            }
        }
    }

    @Test func aFrozenStudyRefusesWithTheImmutabilityGate() async throws {
        try await withTempRoot { root in
            try draft("sealed")
            let concepts = root.appending(components: "prompts", "concepts")
            try FileManager.default.createDirectory(
                at: concepts, withIntermediateDirectories: true)
            try FileManager.default.copyItem(
                at: CodeResources.compiledCheckoutPath.appending(
                    components: "prompts", "concepts", "french"),
                to: concepts.appending(component: "french"))
            await invoke(["attach", "sealed", "french"])
            #expect(await invoke(["freeze", "sealed", "--force"]).exitCode == 0)
            #expect {
                try ExperimentStore.setSamplingProtocol(
                    temperature: 0.7, experimentName: "sealed")
            } throws: { error in
                self.lifecycleGate(error) == .statusImmutable
            }
            #expect {
                try ExperimentStore.setExclusionRules(
                    [ExclusionRule(rule: "unparseableEndpoint")],
                    experimentName: "sealed")
            } throws: { error in
                self.lifecycleGate(error) == .statusImmutable
            }
        }
    }

    // MARK: - The set-sampling verb

    /// The motivating arm, end-to-end: 25 samples × T=0.7 × 1024 tokens ×
    /// derivedSHA256 — declared in one invocation, echoed from the stored
    /// manifest, with `verify` named as the next action because the joint
    /// stochastic rules live there.
    @Test func theStochasticReplicationArmIsAuthorable() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            let outcome = await invoke([
                "set-sampling", "demo", "--temperature", "0.7",
                "--max-tokens", "1024", "--samples-per-item", "25",
                "--seed-policy", "derivedSHA256",
            ])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["temperature"] == .number(0.7))
            #expect(outcome.envelope.result?["maxTokens"] == .number(1024))
            #expect(outcome.envelope.result?["samplesPerItem"] == .number(25))
            #expect(
                outcome.envelope.result?["seedPolicy"]
                    == .string("derivedSHA256"))
            #expect(
                outcome.envelope.nextAction?.verb
                    == "experiment verify demo")
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(loaded.temperature == 0.7)
            #expect(loaded.maxTokens == 1024)
            #expect(loaded.samplesPerItem == 25)
            #expect(loaded.seedPolicy == "derivedSHA256")
        }
    }

    /// No flags = nothing to write = a malformed invocation, never a
    /// `changed: true` over a manifest that did not move.
    @Test func noFlagsWouldWriteNothingAndIsMalformed() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            let outcome = await invoke(["set-sampling", "demo"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(outcome.envelope.error?.gate == nil)
        }
    }

    @Test func anUnknownSeedPolicyIsAMalformedInvocation() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            let outcome = await invoke(
                ["set-sampling", "demo", "--seed-policy", "diceRoll"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            let repair = try #require(outcome.envelope.error?.repairAction)
            for policy in ExperimentStore.knownSeedPolicies {
                #expect(repair.contains(policy), "repair omits \(policy)")
            }
            #expect(
                try ExperimentStore.load(name: "demo").seedPolicy == nil)
        }
    }

    /// A declared temperature nothing will read (deterministic-only
    /// instruments) advises at the moment of declaration — the same advisory
    /// `set-instruments` emits from the other side of the same join.
    @Test func anInertTemperatureAdvisesAtDeclaration() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            await invoke(["set-instruments", "demo", "choiceProbability"])
            let outcome = await invoke(
                ["set-sampling", "demo", "--temperature", "0.7"])
            #expect(outcome.envelope.exitCode == 0)
            let advisories = outcome.envelope.advisories ?? []
            #expect(
                advisories.contains {
                    $0.code == CLIAdvisory.choiceItemsWithoutInstrument.rawValue
                },
                "no inert-sampling advisory on a deterministic-only study")
        }
    }

    // MARK: - The set-exclusions verb

    @Test func rulesAreDeclaredWithTheirBoundsAndEchoed() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            let outcome = await invoke([
                "set-exclusions", "demo", "unparseableEndpoint,outOfRange",
                "--min", "0", "--max", "600",
            ])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["ruleCount"] == .number(2))
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(
                loaded.exclusionRules == [
                    ExclusionRule(rule: "unparseableEndpoint"),
                    ExclusionRule(rule: "outOfRange", min: 0, max: 600),
                ])
        }
    }

    @Test func theEmptyStringClearsTheDeclaration() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            await invoke(
                ["set-exclusions", "demo", "unparseableEndpoint"])
            let outcome = await invoke(["set-exclusions", "demo", ""])
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["ruleCount"] == .number(0))
            #expect(
                try ExperimentStore.load(name: "demo").exclusionRules == nil)
        }
    }

    /// An unknown rule id refuses with the exclusion ENGINE's own sentence —
    /// one vocabulary, one wording, both engines — classified as malformed.
    @Test func anUnknownRuleRefusesWithTheEngineWording() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            let outcome = await invoke(
                ["set-exclusions", "demo", "outOfRnge"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(reason.contains("'outOfRnge' is not recognized"))
            #expect(
                try ExperimentStore.load(name: "demo").exclusionRules == nil)
        }
    }

    /// A bound aimed at no rule that takes one is a malformed invocation,
    /// refused before the store — never silently attached to nothing.
    @Test func boundsWithoutTheOutOfRangeRuleAreMalformed() async throws {
        try await withTempRoot { _ in
            try draft("demo")
            let outcome = await invoke([
                "set-exclusions", "demo", "unparseableEndpoint",
                "--min", "0",
            ])
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(
                try ExperimentStore.load(name: "demo").exclusionRules == nil)
        }
    }
}
