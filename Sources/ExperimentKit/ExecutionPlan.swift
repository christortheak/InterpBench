import Foundation

/// What a study will ACTUALLY do when it runs — one resolver, used by the run
/// loops, the routing rules and the UI (E1).
///
/// The rule "which instruments imply generation" already existed, but only
/// inside the run loop (`wantsSampled` / `wants_sampled`). Everything else
/// re-derived its own answer or guessed. The visible consequence: a
/// logprob-only study was **pinned to the server by a temperature its
/// instrument ignores**. `LogprobInstrument` scores options through a stepped
/// KV cache and never samples, so temperature decides nothing for it — yet a
/// declared `temperature > 0` made the study server-only, because the routing
/// rule looked at the number without asking whether anything read it.
///
/// So routing depends on the PLAN, not on a field that may be inert. And
/// where a declared value is inert, that is said out loud rather than
/// silently ignored — a temperature that decides nothing is a study-design
/// mistake worth surfacing, just not a reason to move the study to another
/// substrate.
///
/// Cross-engine twin: `Server/steerlab_server/experiment/execution_plan.py`.
public enum ExecutionPlan {

    /// Instruments that require sampled generation.
    ///
    /// `repeReaderScore` reads a reader over the model's OUTPUT, so it needs
    /// text even though it is not itself a sampled-text metric.
    public static let generationImplyingInstruments: Set<String> = [
        "sampledText", "repeReaderScore",
    ]

    /// Instruments that read option logprobs directly, with no generation.
    public static let directScoringInstruments: Set<String> = [
        "answerTokenLogprob", "choiceProbability", "ordinalScale",
    ]

    public struct Plan: Sendable, Equatable {
        /// The study samples text. Temperature and seeds are operative only
        /// when this is true.
        public var generatesSampledText: Bool
        /// The study scores declared options deterministically.
        public var scoresDirectly: Bool
        /// The resolved instrument list (empty input resolves to sampled
        /// text, the engine default).
        public var instruments: [String]

        /// Sampling settings are operative for this plan.
        public var samplingIsOperative: Bool { generatesSampledText }

        public var summary: String {
            switch (generatesSampledText, scoresDirectly) {
            case (true, true):
                "generates sampled text AND scores options deterministically"
            case (true, false):
                "generates sampled text"
            case (false, true):
                "scores options deterministically — no sampling"
            case (false, false):
                "nothing to run"
            }
        }
    }

    /// Resolve what the declared instruments mean.
    ///
    /// Absent or empty resolves to sampled text — the engine default both run
    /// loops already apply (`instruments.isEmpty || contains("sampledText")`).
    public static func resolve(instruments: [String]?) -> Plan {
        let declared = instruments ?? []
        guard !declared.isEmpty else {
            return Plan(
                generatesSampledText: true, scoresDirectly: false,
                instruments: ["sampledText"])
        }
        let set = Set(declared)
        return Plan(
            generatesSampledText: !set.isDisjoint(with: generationImplyingInstruments),
            scoresDirectly: !set.isDisjoint(with: directScoringInstruments),
            instruments: declared)
    }

    /// A declared sampling setting that this plan will not read.
    ///
    /// Advisory, never a refusal: declaring a temperature on a
    /// deterministic-only study is a design mistake, but the RUN is
    /// well-defined and its result is unaffected.
    public static func inertSamplingAdvisory(
        instruments: [String]?, temperature: Double?, samplesPerItem: Int?
    ) -> String? {
        let plan = resolve(instruments: instruments)
        guard !plan.samplingIsOperative else { return nil }
        var inert: [String] = []
        if (temperature ?? 0) > 0 {
            inert.append(
                "temperature "
                    + (temperature ?? 0).formatted(
                        .number.precision(.fractionLength(0 ... 2))))
        }
        if (samplesPerItem ?? 1) > 1 {
            inert.append("samplesPerItem \(samplesPerItem ?? 1)")
        }
        guard !inert.isEmpty else { return nil }
        return "this study \(plan.summary), so \(inert.joined(separator: " and ")) "
            + "\(inert.count == 1 ? "is" : "are") inert — the deterministic "
            + "instruments never sample, so the declared value changes "
            + "nothing. Remove it, or add a generating instrument if sampled "
            + "output was intended."
    }
}
