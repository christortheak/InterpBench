import Foundation
import Testing

@testable import ExperimentKit

/// E1 — one resolver for what a study will actually do.
///
/// The rule already existed inside the run loop (`wantsSampled`); everything
/// else re-derived it or guessed. The visible consequence: a logprob-only
/// study was pinned to the server by a temperature its instrument ignores.
struct ExecutionPlanTests {

    // MARK: resolution

    @Test func absentInstrumentsResolveToSampledText() {
        // The engine default both run loops already applied.
        let plan = ExecutionPlan.resolve(instruments: nil)
        #expect(plan.generatesSampledText)
        #expect(!plan.scoresDirectly)
        #expect(plan.instruments == ["sampledText"])
        #expect(ExecutionPlan.resolve(instruments: []).generatesSampledText)
    }

    @Test func deterministicInstrumentsDoNotGenerate() {
        for instrument in ["answerTokenLogprob", "choiceProbability", "ordinalScale"] {
            let plan = ExecutionPlan.resolve(instruments: [instrument])
            #expect(!plan.generatesSampledText, "\(instrument) should not generate")
            #expect(plan.scoresDirectly)
            #expect(!plan.samplingIsOperative)
        }
    }

    @Test func readerScoringRequiresGeneration() {
        // repeReaderScore reads a reader over the model's OUTPUT, so it needs
        // text even though it is not itself a sampled-text metric.
        let plan = ExecutionPlan.resolve(instruments: ["repeReaderScore"])
        #expect(plan.generatesSampledText)
        #expect(plan.samplingIsOperative)
    }

    @Test func bothModesCanBeDeclaredTogether() {
        let plan = ExecutionPlan.resolve(
            instruments: ["sampledText", "answerTokenLogprob"])
        #expect(plan.generatesSampledText)
        #expect(plan.scoresDirectly)
        #expect(plan.summary.contains("AND"))
    }

    // MARK: THE bug

    @Test func aLogprobOnlyStudyIsNotPinnedToTheServerByAnInertTemperature() {
        // The failure this closes: LogprobInstrument scores through a stepped
        // KV cache and never samples, so temperature decides nothing for it —
        // yet a declared temperature made the study server-only.
        #expect(
            !SubstrateRouting.isStochastic(
                temperature: 0.7, samplesPerItem: 4,
                outcomeInstruments: ["answerTokenLogprob"]))
        let decision = SubstrateRouting.decide(
            .init(
                temperature: 0.7, samplesPerItem: 4,
                outcomeInstruments: ["answerTokenLogprob"],
                activeWorkspaceIsServer: false, siteRegistered: true,
                serverConnected: true))
        #expect(decision.selection == .thisMac)
        #expect(!decision.pinnedToServer)
    }

    @Test func aGeneratingStudyIsStillPinnedByRealStochasticity() {
        // The rule that was right stays right.
        #expect(
            SubstrateRouting.isStochastic(
                temperature: 0.7, samplesPerItem: 1,
                outcomeInstruments: ["sampledText"]))
        #expect(
            SubstrateRouting.isStochastic(
                temperature: 0, samplesPerItem: 4,
                outcomeInstruments: ["repeReaderScore"]))
        let decision = SubstrateRouting.decide(
            .init(
                temperature: 0.7, outcomeInstruments: ["sampledText"],
                siteRegistered: true, serverConnected: true))
        #expect(decision.selection == .server)
        #expect(decision.pinnedToServer)
    }

    /// Callers that pass no instruments must behave exactly as before.
    @Test func theHistoricalAnswerIsPreservedWhenInstrumentsAreUnknown() {
        #expect(
            SubstrateRouting.isStochastic(temperature: 0.7, samplesPerItem: nil))
        #expect(
            SubstrateRouting.isStochastic(temperature: 0, samplesPerItem: 3))
        #expect(
            !SubstrateRouting.isStochastic(temperature: 0, samplesPerItem: 1))
    }

    // MARK: the inert-value advisory

    @Test func anInertTemperatureIsSaidOutLoudRatherThanIgnored() throws {
        // A temperature that decides nothing is a design mistake — just not
        // a reason to move the study to another substrate.
        let advisory = try #require(
            ExecutionPlan.inertSamplingAdvisory(
                instruments: ["answerTokenLogprob"], temperature: 0.7,
                samplesPerItem: 1))
        #expect(advisory.contains("inert"))
        #expect(advisory.contains("temperature 0.7"))
        #expect(advisory.contains("never sample"))
    }

    @Test func inertSamplesPerItemIsAlsoNamed() throws {
        let advisory = try #require(
            ExecutionPlan.inertSamplingAdvisory(
                instruments: ["ordinalScale"], temperature: 0, samplesPerItem: 8))
        #expect(advisory.contains("samplesPerItem 8"))
    }

    @Test func nothingIsSaidWhenTheSettingIsOperativeOrAbsent() {
        // Operative: the value does something, so there is nothing to warn of.
        #expect(
            ExecutionPlan.inertSamplingAdvisory(
                instruments: ["sampledText"], temperature: 0.7,
                samplesPerItem: 4) == nil)
        // Deterministic study with greedy defaults: nothing inert either.
        #expect(
            ExecutionPlan.inertSamplingAdvisory(
                instruments: ["answerTokenLogprob"], temperature: 0,
                samplesPerItem: 1) == nil)
    }

    // MARK: the engine gate

    @Test func aLocalDeterministicStudyIsNotRefusedOverATemperatureNothingReads() throws {
        // The greedy requirement exists because the MLX generator has no
        // per-run sampling seed. A study that never samples is unaffected.
        var manifest = ExperimentManifest(
            name: "e1", description: "", modelID: "test/model")
        manifest.temperature = 0.7
        manifest.outcomeInstruments = ["answerTokenLogprob"]
        try ExperimentTasks.requireGreedyLocalDesign(manifest)

        // ...but a generating study still is.
        manifest.outcomeInstruments = ["sampledText"]
        #expect(throws: (any Error).self) {
            try ExperimentTasks.requireGreedyLocalDesign(manifest)
        }
    }
}
