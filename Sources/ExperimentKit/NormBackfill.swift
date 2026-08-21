import CryptoKit
import Foundation
import MLXLMCommon
import SteeringKit

/// Residual-norm BACKFILL: measure the per-layer typical residual-stream
/// norms for an EXISTING vector artifact that lacks them (legacy vectors,
/// Gemma Scope SAE imports, RepE reader-derived steering vectors) on a
/// pinned neutral corpus, and write a NEW artifact — same vector bytes,
/// sidecar completed with `residualNormPerLayer` + provenance — into a fresh
/// immutable run directory. The original run is never touched (CLAUDE.md ›
/// Data & reproducibility: never overwrite or mutate a run). Swift twin of
/// the server's `POST /api/vectors/backfill-norms` job.
///
/// The measurement reuses the exact extraction code path
/// (`ConceptExtractor.neutralCorpusResidualNorms`, the neutral-corpus branch
/// of `extract`/`extractGrandMean`) at the artifact's stamped reading
/// position, so backfilled denominators are the ones a fresh extraction
/// against the same corpus would have stamped.
public enum NormBackfill {

    public struct BackfillError: Error, CustomStringConvertible, Equatable {
        public let reason: String
        public var description: String { reason }

        public init(reason: String) {
            self.reason = reason
        }
    }

    /// The exact `residualNormSource` string extraction stamps for
    /// neutral-corpus norms (`ConceptBuilder.saveConceptAndExtract` /
    /// `ExperimentTasks.saveVectorSidecar`): "neutral-corpus <hash12>".
    public static func residualNormSource(corpusHash: String) -> String {
        "neutral-corpus \(corpusHash.prefix(12))"
    }

    /// Pure preconditions (unit-tested without a model):
    /// - backfill never overwrites — an artifact that already records
    ///   `residualNormPerLayer` is refused;
    /// - HARD model guard — norms are a per-model measurement, so the loaded
    ///   model must be the sidecar's model (same style as the RepE reader's
    ///   scoring guard).
    public static func validate(
        sidecar: SteeringVectorSidecar, loadedModelID: String,
        redenominate: Bool = false
    ) throws {
        if sidecar.residualNormPerLayer != nil {
            let source = sidecar.residualNormSource ?? "unrecorded"
            // Loud, never silent (the freeze --force pattern): redenominate
            // re-measures a STIMULUS-denominated artifact on the neutral
            // corpus into a NEW artifact; already-neutral artifacts refuse.
            guard redenominate else {
                throw BackfillError(
                    reason: "artifact already records residualNormPerLayer "
                        + "(source: \(source)) — backfill never overwrites; "
                        + "pass --redenominate to write a NEW neutral-corpus-"
                        + "denominated artifact")
            }
            guard !source.hasPrefix("neutral-corpus") else {
                throw BackfillError(
                    reason: "artifact is already neutral-corpus denominated "
                        + "(\(source)) — nothing to redenominate")
            }
        }
        guard sidecar.modelID == loadedModelID else {
            throw BackfillError(
                reason: "vector was extracted on model '\(sidecar.modelID)'; the "
                    + "loaded model is '\(loadedModelID)' — residual norms are a "
                    + "per-model measurement, load the artifact's model to backfill")
        }
    }

    /// The artifact's stamped reading position. Artifacts predating the
    /// options schema (nil label) read at the last token; an unparseable
    /// label is an error, never a guess.
    static func readingPosition(for sidecar: SteeringVectorSidecar) throws -> ReadingPosition {
        guard let label = sidecar.readingPosition else { return .lastToken }
        guard let position = ReadingPosition(label: label) else {
            throw BackfillError(
                reason: "unrecognized readingPosition '\(label)' in sidecar — "
                    + "cannot re-measure at the stamped position")
        }
        return position
    }

    /// Align measured norms (one per model layer) with the artifact's vector
    /// layers. Derived artifacts (reader-/SAE-derived, zeros below the
    /// injection layer) carry fewer vector layers than the model has blocks,
    /// so a measurement covering MORE layers takes the aligned prefix (block
    /// indices coincide; the model guard already pins the model). A
    /// measurement covering FEWER layers than the vectors is a hard mismatch.
    public static func alignedNorms(_ measured: [Float], layerCount: Int) throws -> [Float] {
        // A non-finite norm ANYWHERE in the measurement means the forward
        // pass overflowed (fp16 activation blow-up, the Gemma-3 failure
        // mode) — the whole measurement is untrustworthy as a denominator,
        // even at layers before the overflow. Same rule as the server twin.
        guard measured.allSatisfy(\.isFinite) else {
            throw BackfillError(
                reason: "measured residual norms contain non-finite values — "
                    + "activation overflow; re-measure with a float32-capable "
                    + "dtype (Gemma 3 needs float32 on MPS)")
        }
        guard measured.count >= layerCount else {
            throw BackfillError(
                reason: "measured residual norms for \(measured.count) layers but "
                    + "the artifact has \(layerCount) vector layers — layer counts "
                    + "must match the vectors")
        }
        return Array(measured.prefix(layerCount))
    }

    /// Pure sidecar assembly: every original field is preserved (including
    /// `source`/`readerID`/`readerHash`/`controlMode` on reader-derived
    /// vectors); only the norm fields are filled and the `normBackfill`
    /// provenance stamped (pinned cross-engine JSON shape).
    public static func backfilledSidecar(
        from original: SteeringVectorSidecar,
        residualNormPerLayer: [Float],
        corpusHash: String,
        sourceArtifact: String,
        sourceVectorsHash: String,
        date: Date = Date()
    ) -> SteeringVectorSidecar {
        // Redenomination provenance: record what the new denominator replaced.
        let replacedSource: String? =
            original.residualNormPerLayer != nil ? original.residualNormSource : nil
        var sidecar = original
        sidecar.residualNormPerLayer = residualNormPerLayer
        sidecar.residualNormSource = residualNormSource(corpusHash: corpusHash)
        // Backfill IS the opt-in migration to the current denominator
        // convention: these norms were measured just now, by this code, under
        // `ResidualNormConvention.current`. Legacy artifacts are never
        // stamped in place — running backfill is how a researcher chooses to
        // move one onto the stamped convention.
        sidecar.residualNormConvention = ResidualNormConvention.current
        sidecar.neutralCorpusHash = corpusHash
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(identifier: "UTC")
        sidecar.normBackfill = SteeringVectorSidecar.NormBackfillProvenance(
            sourceArtifact: sourceArtifact,
            sourceVectorsHash: sourceVectorsHash,
            date: formatter.string(from: date),
            replacedNormSource: replacedSource)
        return sidecar
    }

    /// Measure and write. `artifact` is the source artifact's base path
    /// (`<runDir>/<name>`, no extension — `VectorArtifact.id`); the new
    /// artifact keeps the same name inside `runDirectory`. The `.safetensors`
    /// bytes are copied byte-for-byte (never re-encoded), so
    /// `sourceVectorsHash` is also the new file's hash. Returns the new
    /// artifact's base URL.
    @discardableResult
    public static func backfillNorms(
        container: ModelContainer,
        modelID: String,
        artifact: URL,
        corpusURL: URL,
        runDirectory: URL,
        redenominate: Bool = false
    ) async throws -> URL {
        let name = artifact.lastPathComponent
        let sidecarData = try Data(contentsOf: artifact.appendingPathExtension("json"))
        let sidecar = try JSONDecoder().decode(SteeringVectorSidecar.self, from: sidecarData)
        try validate(sidecar: sidecar, loadedModelID: modelID, redenominate: redenominate)

        let vectorBytes = try Data(contentsOf: artifact.appendingPathExtension("safetensors"))
        // `loadTexts` hashes the corpus FILE bytes (stimulus-set convention) —
        // the neutralCorpusHash pin, and the hash embedded in the source string.
        let corpus = try StimulusSet.loadTexts(url: corpusURL)
        let position = try readingPosition(for: sidecar)
        let measured = try await ConceptExtractor.neutralCorpusResidualNorms(
            container: container, texts: corpus.texts, position: position)
        let norms = try alignedNorms(
            measured.residualNormPerLayer, layerCount: sidecar.layerCount)

        let updated = backfilledSidecar(
            from: sidecar,
            residualNormPerLayer: norms,
            corpusHash: corpus.hash,
            sourceArtifact: artifact.path,
            sourceVectorsHash: SHA256.hash(data: vectorBytes)
                .map { String(format: "%02x", $0) }.joined())

        try FileManager.default.createDirectory(
            at: runDirectory, withIntermediateDirectories: true)
        try RunMetadata.write(
            runType: "norm-backfill", to: runDirectory,
            modelID: sidecar.modelID, revision: sidecar.revision)
        try vectorBytes.write(to: runDirectory.appending(component: "\(name).safetensors"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(updated).write(
            to: runDirectory.appending(component: "\(name).json"))
        return runDirectory.appending(component: name)
    }
}
