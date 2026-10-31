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

    /// Declared seed settings an ordinary study's run will not read — said
    /// at `verify` and at `set-sampling`, never refused (advisory code
    /// `seedSettingNotUsed`). A panel says nothing here: its turns take seeds
    /// by their own rule. Byte-identical to the Python twin
    /// `sampling.unused_seed_settings`; the shared fixture
    /// `Tests/Fixtures/cross-engine/study-sampling-seeds.json` pins both.
    public static func unusedSeedSettings(_ manifest: ExperimentManifest) -> [String] {
        guard manifest.seedPolicy == "derivedSHA256", manifest.studyKind != .multiAgent else {
            return []
        }
        guard manifest.temperature > 0 else {
            return [
                "seedPolicy derivedSHA256 has no effect at temperature 0: "
                    + "decoding is greedy and no seed is used. Seeds are derived "
                    + "per record only when the temperature is above 0."
            ]
        }
        let count = manifest.seeds.count
        guard count > 1 else { return [] }
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
