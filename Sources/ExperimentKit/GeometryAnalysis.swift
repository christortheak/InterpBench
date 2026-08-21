import Foundation
import SteeringKit

public struct GeometryLayerMatrix: Sendable, Identifiable {
    public var id: Int { layer }
    public let layer: Int
    public let labels: [String]
    public let values: [[Float]]

    public init(layer: Int, labels: [String], values: [[Float]]) {
        self.layer = layer
        self.labels = labels
        self.values = values
    }
}

public struct GeometryAnalysisResult: Sendable {
    public let artifacts: [VectorArtifact]
    public let matrices: [GeometryLayerMatrix]
    public let rsa: [[Float]]

    public init(
        artifacts: [VectorArtifact],
        matrices: [GeometryLayerMatrix],
        rsa: [[Float]]
    ) {
        self.artifacts = artifacts
        self.matrices = matrices
        self.rsa = rsa
    }
}

public enum GeometryAnalysisError: Error, CustomStringConvertible {
    case tooFewVectors
    case noCommonLayers
    case dimensionMismatch(String)

    public var description: String {
        switch self {
        case .tooFewVectors:
            "select at least two vectors"
        case .noCommonLayers:
            "selected vectors have no common layers"
        case .dimensionMismatch(let label):
            "vector dimensions do not match for \(label)"
        }
    }
}

public enum GeometryAnalysis {
    public static func analyze(
        artifacts: [VectorArtifact]
    ) throws -> GeometryAnalysisResult {
        guard artifacts.count >= 2 else { throw GeometryAnalysisError.tooFewVectors }

        let loaded = try artifacts.map { artifact in
            let vectors = try SteeringVectorStore.load(
                from: artifact.directory, name: artifact.name
            ).vectors
            return (artifact, vectors)
        }

        guard let layerCount = loaded.map({ $0.1.layerCount }).min(), layerCount > 0 else {
            throw GeometryAnalysisError.noCommonLayers
        }

        let labels = artifacts.map(\.label)
        var matrices: [GeometryLayerMatrix] = []
        matrices.reserveCapacity(layerCount)

        for layer in 0 ..< layerCount {
            let vectors = loaded.map { $0.1.perLayer[layer] }
            guard let dimension = vectors.first?.count,
                vectors.allSatisfy({ $0.count == dimension })
            else {
                throw GeometryAnalysisError.dimensionMismatch("layer \(layer)")
            }

            var values = Array(
                repeating: Array(repeating: Float(0), count: vectors.count),
                count: vectors.count)
            for i in vectors.indices {
                for j in vectors.indices {
                    values[i][j] =
                        (try? SteeringVectorMath.cosineSimilarity(vectors[i], vectors[j])) ?? 0
                }
            }
            matrices.append(GeometryLayerMatrix(layer: layer, labels: labels, values: values))
        }

        let triangles = matrices.map { upperTriangle($0.values) }
        var rsa = Array(
            repeating: Array(repeating: Float(0), count: matrices.count),
            count: matrices.count)
        for i in matrices.indices {
            for j in matrices.indices {
                rsa[i][j] = pearson(triangles[i], triangles[j])
            }
        }

        return GeometryAnalysisResult(
            artifacts: artifacts, matrices: matrices, rsa: rsa)
    }

    public static func csv(matrix: GeometryLayerMatrix) -> String {
        var rows = ["concept," + matrix.labels.map(escape).joined(separator: ",")]
        for (index, label) in matrix.labels.enumerated() {
            let values = matrix.values[index].map { String(format: "%.6f", $0) }
            rows.append(escape(label) + "," + values.joined(separator: ","))
        }
        return rows.joined(separator: "\n") + "\n"
    }

    public static func csvRSA(_ result: GeometryAnalysisResult) -> String {
        let labels = result.matrices.map { "L\($0.layer)" }
        var rows = ["layer," + labels.joined(separator: ",")]
        for (index, label) in labels.enumerated() {
            let values = result.rsa[index].map { String(format: "%.6f", $0) }
            rows.append(label + "," + values.joined(separator: ","))
        }
        return rows.joined(separator: "\n") + "\n"
    }

    private static func upperTriangle(_ matrix: [[Float]]) -> [Float] {
        guard matrix.count >= 2 else { return [] }
        var values: [Float] = []
        for i in 0 ..< matrix.count {
            for j in (i + 1) ..< matrix.count {
                values.append(matrix[i][j])
            }
        }
        return values
    }

    private static func pearson(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, a.count >= 2 else { return 0 }
        let meanA = a.reduce(0, +) / Float(a.count)
        let meanB = b.reduce(0, +) / Float(b.count)
        var numerator: Float = 0
        var denomA: Float = 0
        var denomB: Float = 0
        for (x, y) in zip(a, b) {
            let dx = x - meanA
            let dy = y - meanB
            numerator += dx * dy
            denomA += dx * dx
            denomB += dy * dy
        }
        guard denomA > 0, denomB > 0 else { return 0 }
        return numerator / sqrt(denomA * denomB)
    }

    private static func escape(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
            return value
        }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
