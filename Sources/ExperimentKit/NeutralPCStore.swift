import Foundation
import SteeringKit

/// Stored neutral-corpus principal components. These are not concept vectors:
/// they are model-specific nuisance directions estimated from a shared neutral
/// corpus and optionally projected out of active steering vectors at runtime.
public struct NeutralPCBasis: Codable, Sendable {
    public var schemaVersion: Int
    public var modelID: String
    public var modelRevision: String?
    public var corpusKind: NeutralCorpusKind?
    public var corpusName: String?
    public var corpusHash: String
    public var corpusPath: String
    public var readingPosition: String
    public var selectionDescription: String
    public var layers: [Int]
    public var componentsByLayer: [[[Float]]]
    public var residualNormPerLayer: [Float]
    public var screening: ExtractionScreening
    public var tokenRowCount: Int
    /// Token-bank downsampling provenance (nil on pre-downsampling bases):
    /// rows per layer before the deterministic cap, rows actually used for
    /// PCA, and the corpus-hash-derived seed of the draw.
    public var sourceRowsPerLayer: Int?
    public var usedRowsPerLayer: Int?
    public var downsampleSeed: UInt64?
    /// The per-layer row cap that bounded ingestion (nil on bases built
    /// before the cap was enforced during capture).
    public var rowCapPerLayer: Int?
    /// The variance target and the hard component ceiling that produced
    /// `componentsByLayer`. Both are stamped because a target alone does not
    /// determine the component count — the cap can bind.
    public var minimumExplainedVariance: Double?
    public var maximumComponentCount: Int?
    /// Which layers were CAPTURED, in words ("middle-third band 16–31 of 48").
    /// `layers` lists them; this says what rule chose them.
    public var layerSelectionDescription: String?
    public var createdAt: String

    public init(
        modelID: String,
        modelRevision: String? = nil,
        corpusKind: NeutralCorpusKind = .normCalibration,
        corpusName: String = "norm-calibration",
        corpusHash: String,
        corpusPath: String,
        readingPosition: ReadingPosition,
        selectionDescription: String,
        layers: [Int],
        componentsByLayer: [[[Float]]],
        residualNormPerLayer: [Float],
        screening: ExtractionScreening,
        tokenRowCount: Int,
        sourceRowsPerLayer: Int? = nil,
        usedRowsPerLayer: Int? = nil,
        downsampleSeed: UInt64? = nil,
        rowCapPerLayer: Int? = nil,
        minimumExplainedVariance: Double? = nil,
        maximumComponentCount: Int? = nil,
        layerSelectionDescription: String? = nil,
        createdAt: Date = Date()
    ) {
        self.schemaVersion = 1
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.corpusKind = corpusKind
        self.corpusName = corpusName
        self.corpusHash = corpusHash
        self.corpusPath = corpusPath
        self.readingPosition = readingPosition.label
        self.selectionDescription = selectionDescription
        self.layers = layers
        self.componentsByLayer = componentsByLayer
        self.residualNormPerLayer = residualNormPerLayer
        self.screening = screening
        self.tokenRowCount = tokenRowCount
        self.sourceRowsPerLayer = sourceRowsPerLayer
        self.usedRowsPerLayer = usedRowsPerLayer
        self.downsampleSeed = downsampleSeed
        self.rowCapPerLayer = rowCapPerLayer
        self.minimumExplainedVariance = minimumExplainedVariance
        self.maximumComponentCount = maximumComponentCount
        self.layerSelectionDescription = layerSelectionDescription
        self.createdAt = ISO8601DateFormatter().string(from: createdAt)
    }

    public var totalComponentCount: Int {
        componentsByLayer.reduce(0) { $0 + $1.count }
    }

    public func components(for layer: Int) -> [[Float]] {
        guard let index = layers.firstIndex(of: layer),
            componentsByLayer.indices.contains(index)
        else { return [] }
        return componentsByLayer[index]
    }
}

public struct NeutralPCArtifactRecord: Identifiable, Sendable {
    public let url: URL
    public let basis: NeutralPCBasis

    public var id: String { url.path }

    public var label: String {
        let created = basis.createdAt.replacingOccurrences(of: "T", with: " ").prefix(16)
        let perLayer = basis.componentsByLayer.map(\.count)
        let range: String
        if let min = perLayer.min(), let max = perLayer.max() {
            range = min == max ? "\(min)/layer" : "\(min)-\(max)/layer"
        } else {
            range = "0/layer"
        }
        return "\(basis.selectionDescription) · \(range) · \(created)"
    }
}

public enum NeutralPCStore {
    public static var directory: URL {
        VectorCatalog.runsDirectory.appending(components: "neutral-pcs")
    }

    public static func scan(directory root: URL = directory) -> [NeutralPCArtifactRecord] {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
        else { return [] }

        var records: [NeutralPCArtifactRecord] = []
        for case let url as URL in enumerator where url.lastPathComponent == "neutral-pcs.json" {
            guard
                let data = try? Data(contentsOf: url),
                let basis = try? JSONDecoder().decode(NeutralPCBasis.self, from: data)
            else { continue }
            records.append(NeutralPCArtifactRecord(url: url, basis: basis))
        }
        return records.sorted {
            if $0.basis.createdAt == $1.basis.createdAt {
                return $0.url.path < $1.url.path
            }
            return $0.basis.createdAt > $1.basis.createdAt
        }
    }

    public static func load(path: String) throws -> NeutralPCArtifactRecord {
        let url: URL
        if path.hasPrefix("/") {
            url = URL(filePath: path)
        } else {
            url = VectorCatalog.projectRoot.appending(path: path)
        }
        let data = try Data(contentsOf: url)
        let basis = try JSONDecoder().decode(NeutralPCBasis.self, from: data)
        return NeutralPCArtifactRecord(url: url, basis: basis)
    }

    public static func relativePath(for record: NeutralPCArtifactRecord) -> String {
        let root = VectorCatalog.projectRoot.path
        let path = record.url.path
        if path.hasPrefix(root + "/") {
            return String(path.dropFirst(root.count + 1))
        }
        return path
    }

    @discardableResult
    public static func save(_ basis: NeutralPCBasis) throws -> NeutralPCArtifactRecord {
        let slug = "neutral-pcs-\(slugify(basis.modelID))-\(basis.corpusHash.prefix(10))"
        let run = try VectorCatalog.makeUniqueRunDirectory(slug: slug, under: directory)
        try RunMetadata.write(
            runType: "neutral-pcs", to: run,
            modelID: basis.modelID, revision: basis.modelRevision)
        let url = run.appending(component: "neutral-pcs.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(basis).write(to: url, options: .atomic)
        return NeutralPCArtifactRecord(url: url, basis: basis)
    }

    private static func slugify(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = value.unicodeScalars.map { scalar -> Character in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        return String(scalars)
            .replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .lowercased()
    }
}
