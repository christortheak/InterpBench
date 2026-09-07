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

    public static func count(_ manifest: ExperimentManifest) -> Int {
        manifest.temperature > 0 && (manifest.samplesPerItem ?? 1) > 1
            ? manifest.samplesPerItem! : manifest.seeds.count
    }

    public static func policy(_ manifest: ExperimentManifest) -> String {
        manifest.temperature > 0 && (manifest.samplesPerItem ?? 1) > 1
            ? "derivedSHA256" : "manifestSeeds"
    }

    public static func seed(
        _ manifest: ExperimentManifest, experimentHash: String, condition: String,
        promptID: String, sampleIndex: Int
    ) -> UInt64 {
        if policy(manifest) == "derivedSHA256" {
            return deriveSeed(experimentHash: experimentHash, condition: condition,
                promptID: promptID, sampleIndex: sampleIndex)
        }
        return manifest.seeds[sampleIndex]
    }
}
