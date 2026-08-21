import Foundation
import SteeringKit

public struct GemmaScopeReportVector: Codable, Sendable {
    public let concept: String
    public let modelID: String
    public let layer: Int
    public let hiddenSize: Int
    public let norm: Float
}

public struct GemmaScopeFeatureRow: Codable, Sendable, Identifiable {
    public var id: Int { feature }
    public let feature: Int
    public let cosine: Float
    public let sparsity: Float?
    public let decoderValues: [Float]?
}

public struct GemmaScopeFeatureReport: Codable, Sendable {
    public let jobFile: String
    public let vector: GemmaScopeReportVector
    public let gemmaScope: GemmaScopeInfo
    public let artifactSidecar: SteeringVectorSidecar
    public let decoderShape: [Int]
    public let topPositive: [GemmaScopeFeatureRow]
    public let topNegative: [GemmaScopeFeatureRow]
    public let topAbsolute: [GemmaScopeFeatureRow]
}

public struct GemmaScopeReportArtifact: Identifiable, Sendable {
    public let url: URL
    public let report: GemmaScopeFeatureReport

    public var id: String { url.path }

    public var label: String {
        "\(report.vector.concept) · L\(report.vector.layer) · \(report.gemmaScope.recommendedSAEID)"
    }
}

public enum GemmaScopeReportImportError: Error, CustomStringConvertible {
    case missingDecoderValues(Int)
    case dimensionMismatch(feature: Int, expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .missingDecoderValues(let feature):
            "feature \(feature) report predates decoder-vector export; rerun Gemma Scope analysis"
        case .dimensionMismatch(let feature, let expected, let actual):
            "feature \(feature) decoder has dimension \(actual), expected \(expected)"
        }
    }
}

public enum GemmaScopeReportCatalog {
    /// The decided cross-engine Gemma Scope import convention (WS7.2) — this
    /// app's historical behavior, now shared with the Python server
    /// (`gemma_scope.IMPORT_CONVENTION`, same string): the SAE decoder row is
    /// rescaled AT IMPORT so its L2 norm equals the analyzed concept vector's
    /// L2 norm at the report layer (norm-matched, like LAT is norm-matched to
    /// the mean difference, so α produces the same perturbation magnitude for
    /// the SAE feature as for the concept vector it was ranked against). The
    /// sidecar stamps the convention name plus `rawDecoderNorm` /
    /// `gemmascopeTargetNorm`, making the transform fully recoverable; a
    /// Gemma-Scope-sourced sidecar WITHOUT the stamp is a pre-convention
    /// import (surfaced by `artifacts audit`; the server warns at load).
    public static let importConvention = "analyzed-vector-norm-match"

    public static func scan(runsDirectory: URL = VectorCatalog.runsDirectory)
        -> [GemmaScopeReportArtifact]
    {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return [] }

        var reports: [GemmaScopeReportArtifact] = []
        for case let url as URL in enumerator where url.lastPathComponent == "gemmascope-report.json" {
            guard
                let data = try? Data(contentsOf: url),
                let report = try? JSONDecoder().decode(GemmaScopeFeatureReport.self, from: data)
            else { continue }
            reports.append(GemmaScopeReportArtifact(url: url, report: report))
        }
        return reports.sorted { $0.url.path > $1.url.path }
    }

    /// Pure importer core (no I/O, unit-testable without MLX): applies
    /// `importConvention` to the feature's decoder row and builds the full
    /// artifact — the scaled row at the report layer of a full-depth zero
    /// vector, carrying the ANALYZED artifact's model identity and
    /// residual-norm calibration so norm-unit alphas keep meaning.
    public static func deriveFeatureArtifact(
        report: GemmaScopeFeatureReport,
        row: GemmaScopeFeatureRow,
        source: String
    ) throws -> (vectors: ConceptVectors, sidecar: SteeringVectorSidecar) {
        guard let values = row.decoderValues else {
            throw GemmaScopeReportImportError.missingDecoderValues(row.feature)
        }
        let sidecar = report.artifactSidecar
        guard values.count == sidecar.hiddenSize else {
            throw GemmaScopeReportImportError.dimensionMismatch(
                feature: row.feature, expected: sidecar.hiddenSize, actual: values.count)
        }
        let rawDecoderNorm = sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
        let scaledValues = scale(values, toNorm: report.vector.norm)

        var perLayer = Array(
            repeating: Array(repeating: Float(0), count: sidecar.hiddenSize),
            count: sidecar.layerCount)
        perLayer[report.vector.layer] = scaledValues
        let vectors = ConceptVectors(perLayer: perLayer)
        var featureSidecar = SteeringVectorSidecar(
            modelID: sidecar.modelID,
            revision: sidecar.revision,
            concept:
                "sae:\(report.vector.concept):L\(report.vector.layer):F\(row.feature)",
            stimulusSetHash:
                "gemmascope:\(report.gemmaScope.recommendedRelease):\(report.gemmaScope.recommendedSAEID):\(row.feature)",
            vectors: vectors,
            options: nil,
            residualNormPerLayer: sidecar.residualNormPerLayer,
            residualNormSource: sidecar.residualNormSource,
            // The norms are the ANALYZED artifact's, copied wholesale — so
            // its convention stamp travels with them (absent stays absent;
            // nothing is invented for a legacy donor).
            residualNormConvention: sidecar.residualNormConvention,
            recipeMethod: nil,
            recipeHash:
                "\(report.gemmaScope.recommendedRelease)|\(report.gemmaScope.recommendedSAEID)|feature:\(row.feature)",
            recipeName:
                "Gemma Scope SAE feature \(row.feature) from \(source), scaled to analyzed vector norm")
        // Same method label as the server importer, and the WS7.2 convention
        // stamp: name + the pre-transform decoder norm + the target norm.
        featureSidecar.extractionMethod = "gemmaScopeSAE"
        featureSidecar.gemmascopeConvention = importConvention
        featureSidecar.rawDecoderNorm = rawDecoderNorm
        featureSidecar.gemmascopeTargetNorm = report.vector.norm
        return (vectors, featureSidecar)
    }

    @discardableResult
    public static func importFeature(
        report artifact: GemmaScopeReportArtifact,
        row: GemmaScopeFeatureRow,
        source: String
    ) throws -> VectorArtifact {
        let (vectors, featureSidecar) = try deriveFeatureArtifact(
            report: artifact.report, row: row, source: source)
        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug:
                "sae-feature-\(slug(artifact.report.vector.concept))-l\(artifact.report.vector.layer)-f\(row.feature)"
        )
        let name = "sae-feature-\(row.feature)"
        try RunMetadata.write(
            runType: "sae-feature-import", to: directory,
            modelID: featureSidecar.modelID, revision: featureSidecar.revision)
        try SteeringVectorStore.save(vectors: vectors, sidecar: featureSidecar, to: directory, name: name)
        return VectorArtifact(directory: directory, name: name, sidecar: featureSidecar)
    }

    /// The convention's transform, float32 math (the server mirrors this
    /// EXACTLY in `gemma_scope._convention_rescale`, including the guard: a
    /// zero-norm row or non-positive target is returned unscaled — the
    /// stamped `rawDecoderNorm` says why).
    private static func scale(_ values: [Float], toNorm targetNorm: Float) -> [Float] {
        let norm = sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
        guard norm > 0, targetNorm > 0 else { return values }
        let factor = targetNorm / norm
        return values.map { $0 * factor }
    }

    private static func slug(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(scalars)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
            .lowercased()
        return collapsed.isEmpty ? "vector" : collapsed
    }
}
