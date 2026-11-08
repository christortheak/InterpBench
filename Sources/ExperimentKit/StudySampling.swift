import CryptoKit
import Foundation
import SteeringKit

/// Exact twin of experiment/sampling.py and condition_execution.py.
public enum StudySampling {
    /// Saved agents retain their own thinking toggle; the baseline reads the
    /// study's explicit effort. Matches effective_variant_condition in Python.
    public static func reasoningEffort(
        _ manifest: ExperimentManifest, variantThinking: Bool?
    ) -> ReasoningEffort {
        if let variantThinking {
            return ReasoningEffort.resolve(nil, qwenThinkingEnabled: variantThinking)
        }
        return manifest.resolvedReasoningEffort
    }

    public static func deriveSeed(
        experimentHash: String, condition: String, promptID: String, sampleIndex: Int
    ) -> UInt64 {
        let bytes = Data("\(experimentHash)|\(condition)|\(promptID)|\(sampleIndex)".utf8)
        return SHA256.hash(data: bytes).prefix(8).reduce(UInt64(0)) {
            ($0 << 8) | UInt64($1)
        } & 0x7fff_ffff_ffff_ffff
    }

    /// The seed rule of an ordinary study's sampled records — the ONE place
    /// it is decided (Python twin `sampling.derives_seeds`). Records derive
    /// per-record seeds when decoding is stochastic and the study either
    /// declares `derivedSHA256` or samples an item more than once;
    /// `samplesPerItem` then sets the per-item count, at any value including
    /// 1. Otherwise they enumerate the fixed seed list, one generation per
    /// listed seed — every undeclared and `manifestSeeds` study, unchanged.
    /// Before 2026-10-31 only `samplesPerItem > 1` selected derived seeds,
    /// so a declared `derivedSHA256` with one sample per item reused the
    /// fixed list (public issue #3).
    public static func derivesSeeds(_ manifest: ExperimentManifest) -> Bool {
        manifest.temperature > 0
            && ((manifest.samplesPerItem ?? 1) > 1 || manifest.seedPolicy == "derivedSHA256")
    }

    /// Whether an engine from before issue #3 would seed this study's sampled
    /// records differently: a declared derivedSHA256 at one sample per item
    /// above temperature 0. Panels never read the policy. Python twin:
    /// `sampling.declared_policy_changes_seeds`.
    public static func declaredPolicyChangesSeeds(_ manifest: ExperimentManifest) -> Bool {
        manifest.studyKind != .multiAgent && manifest.seedPolicy == "derivedSHA256"
            && (manifest.samplesPerItem ?? 1) <= 1 && manifest.temperature > 0
    }

    /// Generations per (condition, prompt) under that rule.
    public static func count(_ manifest: ExperimentManifest) -> Int {
        derivesSeeds(manifest) ? max(1, manifest.samplesPerItem ?? 1) : manifest.seeds.count
    }

    public static func policy(_ manifest: ExperimentManifest) -> String {
        derivesSeeds(manifest) ? "derivedSHA256" : "manifestSeeds"
    }

    public static func seed(
        _ manifest: ExperimentManifest, experimentHash: String, condition: String,
        promptID: String, sampleIndex: Int
    ) -> UInt64 {
        if derivesSeeds(manifest) {
            return deriveSeed(experimentHash: experimentHash, condition: condition,
                promptID: promptID, sampleIndex: sampleIndex)
        }
        return manifest.seeds[sampleIndex]
    }

    /// How `greedySeedListProblem`'s sentence opens, on both engines.
    /// `VerificationRefusal` matches it to name the repair verb, and it says
    /// none of the words a drift sentence uses. Python twin:
    /// `sampling.GREEDY_SEED_LIST_PREFIX`.
    public static let greedySeedListPrefix =
        "at temperature 0 decoding is greedy, so every seed gives the same response"

    /// Why an ordinary study at temperature 0 that generates sampled text
    /// must list exactly one seed (public issue #9), or nil. Under the fixed
    /// seed list each item's sampled text runs once per listed seed; at
    /// temperature 0 every seed gives the same greedy response, so N seeds
    /// would write N identical records per item (each stamped `seedInert`)
    /// and analysis would count them as N observations. Only sampled text
    /// loops over the seeds: an answer-token readout reads each item once, so
    /// a study that measures readouts alone (`ExecutionPlan`) is never
    /// refused here, and its unused list is an advisory
    /// (`unusedSeedSettings`). A panel never reads the list. One sentence
    /// everywhere: a `verify` violation on a draft (so `freeze` refuses), the
    /// engine's submission preflight, both engines' run start, and both
    /// clients before they submit. Byte-identical to the Python twin
    /// `sampling.greedy_seed_list_problem`; the shared fixture
    /// `Tests/Fixtures/cross-engine/study-sampling-seeds.json` pins both.
    public static func greedySeedListProblem(_ manifest: ExperimentManifest) -> String? {
        guard manifest.studyKind != .multiAgent, manifest.temperature <= 0 else {
            return nil
        }
        let count = manifest.seeds.count
        guard count > 1,
            ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
                .generatesSampledText
        else { return nil }
        return "\(greedySeedListPrefix), and the study's \(count) seeds would "
            + "write \(count) identical records per item. Set one seed, or "
            + "raise the temperature above 0 to sample"
    }

    /// Whether `samplesPerItem` decides anything a run writes: a panel's
    /// replicates always do, and an ordinary study's sampled text does. An
    /// answer-token readout reads each item once whatever the count, so a
    /// readout-only study's `samplesPerItem` is inert (`ExecutionPlan.
    /// inertSamplingAdvisory` says so). Python twin: `sampling._samples_are_read`.
    static func samplesAreRead(_ manifest: ExperimentManifest) -> Bool {
        manifest.studyKind == .multiAgent
            || ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
                .generatesSampledText
    }

    /// `verify`'s joint rules for several samples per item, in order: a
    /// temperature above 0 and the derived seed policy. Neither applies to a
    /// study whose run does not read the count (review round 3 of public
    /// issue #9). Byte-identical to the Python twin
    /// `sampling.samples_per_item_problems`; the shared fixture pins both.
    public static func samplesPerItemProblems(_ manifest: ExperimentManifest) -> [String] {
        guard let samples = manifest.samplesPerItem, samples > 1, samplesAreRead(manifest)
        else { return [] }
        var problems: [String] = []
        if manifest.temperature <= 0 {
            problems.append(
                "samplesPerItem > 1 requires temperature > 0 "
                    + "(greedy decoding makes every sample identical)")
        }
        if manifest.seedPolicy != "derivedSHA256" {
            problems.append(
                "samplesPerItem > 1 requires seedPolicy 'derivedSHA256' "
                    + "(per-record seeds derived from condition/prompt/sampleIndex)")
        }
        return problems
    }

    /// What `verify` says of a study frozen under an older release that the
    /// seed rule refuses, or nil: it still verifies (its runs stay
    /// readable), so the reader learns here, not at submission, that running
    /// it needs a duplicate with one seed. Advisory code `refusedWhenRun`.
    /// Python twin: `sampling.frozen_greedy_seed_list_advisory`, as the Mac
    /// command line spells the repair.
    public static func frozenGreedySeedListAdvisory(_ manifest: ExperimentManifest) -> String? {
        guard manifest.status != .draft else { return nil }
        // Never for a study whose declared chain has no run stage.
        if let block = manifest.pipeline, block != .null,
            !(PipelineDraft.parse(block)?.stages.contains("run") ?? false)
        {
            return nil
        }
        guard let problem = greedySeedListProblem(manifest) else { return nil }
        return "this study will be refused when its run is submitted or "
            + "started, because \(problem). To run it: "
            + greedySeedListRepair(name: manifest.name, frozen: true)
    }

    /// The run-start repair for `greedySeedListProblem`: one seed with
    /// `set-sampling --seeds`, on a duplicate when the study is frozen.
    /// Python twin: `sampling.greedy_seed_list_repair`, as the Mac command
    /// line spells it.
    public static func greedySeedListRepair(name: String, frozen: Bool) -> String {
        let target = frozen ? "\(name)-v2" : name
        return (frozen ? "steerlab-cli experiment duplicate \(name) \(target) && " : "")
            + "steerlab-cli experiment set-sampling \(target) --seeds <n>"
            + "  (one seed; or raise the temperature above 0 to sample)"
    }

    /// Declared seed settings an ordinary study's run will not read — said
    /// at `verify` and at `set-sampling`, never refused (advisory code
    /// `seedSettingNotUsed`). A panel says nothing here: its turns take seeds
    /// by their own rule. Nor does a temperature-0 study with several seeds
    /// that generates sampled text: it is refused (`greedySeedListProblem`),
    /// and a setting a study is refused for is never called one with "no
    /// effect". A study that measures answer-token readouts alone is told
    /// its list of several seeds is not used: a readout reads each item once,
    /// at any temperature. Byte-identical to the Python twin
    /// `sampling.unused_seed_settings`; the shared fixture
    /// `Tests/Fixtures/cross-engine/study-sampling-seeds.json` pins both.
    public static func unusedSeedSettings(_ manifest: ExperimentManifest) -> [String] {
        guard manifest.studyKind != .multiAgent else { return [] }
        let count = manifest.seeds.count
        let plan = ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
        if !plan.generatesSampledText, plan.scoresDirectly, count > 1 {
            return [
                "the seed list (\(count) seeds) is not used: this study "
                    + "records answer-token readouts only, and a readout reads each "
                    + "item once and draws no seed at any temperature. To generate "
                    + "responses that use the seeds, add sampledText to "
                    + "outcomeInstruments."
            ]
        }
        guard manifest.seedPolicy == "derivedSHA256" else { return [] }
        guard manifest.temperature > 0 else {
            if greedySeedListProblem(manifest) != nil { return [] }
            return [
                "seedPolicy derivedSHA256 has no effect at temperature 0: "
                    + "decoding is greedy and no seed is used. Seeds are derived "
                    + "per record only when the temperature is above 0."
            ]
        }
        guard plan.generatesSampledText, count > 1 else { return [] }
        let runs = max(1, manifest.samplesPerItem ?? 1)
        return [
            "the seed list (\(count) seeds) is not used under seedPolicy "
                + "derivedSHA256: each item runs samplesPerItem (\(runs)) "
                + "time\(runs == 1 ? "" : "s"), each with its own seed "
                + "derived from the study hash, condition, prompt ID, and "
                + "sample index. To generate more responses per item, raise "
                + "samplesPerItem."
        ]
    }
}
