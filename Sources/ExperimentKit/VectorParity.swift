import Foundation
import SteeringKit

/// Cross-engine vector-artifact parity harness (WS7.3; `steerlab-cli vectors
/// compare`). Compares two per-layer steering-vector artifacts
/// (`<name>.safetensors` + `<name>.json` sidecar — the cross-engine on-disk
/// contract of `SteeringVectorStore`) layer by layer: cosine similarity, norm
/// ratio, and a min/mean summary with a threshold gate for CI. The Python
/// twin (`steerlab_server/steering/vector_parity.py`, `steerlab-server
/// vectors compare`) emits a KEY-IDENTICAL JSON report computed with the
/// same double-precision sequential arithmetic, so both test suites assert
/// the same committed golden files (`Tests/ExperimentKitTests/Fixtures/
/// parity/` / `Server/tests/fixtures/parity/`).
///
/// Deliberately PURE CPU: the safetensors reader below parses the format
/// directly (8-byte little-endian header length, JSON header, raw F32 data)
/// instead of `MLX.loadArrays`, so the compare verb and its tests run
/// without Metal — parity checks must work on any box, including CI without
/// a GPU and plain `swift test` without the metallib bundle.
///
/// Pinned semantics (mirror any change in BOTH engines):
/// - Layer-count mismatch is tolerated: the intersection
///   `0..<min(countA, countB)` is compared and `layerCountMismatch` says so.
/// - A layer where either vector has zero norm has no defined cosine: its
///   `cosine` is null, it is EXCLUDED from min/mean cosine, and counted in
///   `summary.skippedZeroNormLayers`. `normRatio` is `normB / normA` (B in
///   units of A), null when `normA` is zero.
/// - `pass` is true iff a min cosine exists and is ≥ `threshold` — an
///   empty/fully-skipped comparison can never pass.
/// - Hidden-size mismatch is an error, not a report: it is the CLI's THIRD
///   outcome (could-not-compare), never a comparison that failed. Layer-count
///   mismatch stays a report — those artifacts ARE comparable.
public enum VectorParity {

    public static let defaultThreshold = 0.98

    public struct ParityError: Error, CustomStringConvertible {

        /// Which COULD-NOT-COMPARE class this is. Both answer the CLI's third
        /// outcome (`notFound`/66 in JSON, exit 2 in human mode — see
        /// `ExperimentCLIRunner.runVectorsCommand`), but they have different
        /// repairs, and a repair that names the wrong thing sends a caller in
        /// a circle. Python twin: `OSError` vs `ValueError` out of
        /// `vector_parity.compare_paths`.
        public enum Kind: String, Sendable {
            /// The artifact, its sidecar, or a layer the sidecar declares
            /// could not be read. Repair: fix the path, or extract.
            case unreadableArtifact
            /// Both artifacts read cleanly and are still not comparable —
            /// hidden-size mismatch, i.e. two different models. Repair:
            /// compare artifacts from the same model.
            case incomparableArtifacts
        }

        public let reason: String
        public var kind: Kind = .unreadableArtifact
        public var description: String { reason }
    }

    public struct ArtifactSummary: Sendable, Equatable {
        public let name: String
        public let layerCount: Int
        public let hiddenSize: Int
    }

    public struct LayerComparison: Sendable, Equatable {
        public let layer: Int
        public let normA: Double
        public let normB: Double
        /// nil when either norm is zero (cosine undefined).
        public let cosine: Double?
        /// normB / normA; nil when normA is zero.
        public let normRatio: Double?
    }

    public struct Report: Sendable {
        public let artifactA: ArtifactSummary
        public let artifactB: ArtifactSummary
        public let perLayer: [LayerComparison]
        public let threshold: Double

        public var comparedLayerCount: Int { perLayer.count }
        public var layerCountMismatch: Bool {
            artifactA.layerCount != artifactB.layerCount
        }
        public var cosines: [Double] { perLayer.compactMap(\.cosine) }
        public var minCosine: Double? { cosines.min() }
        public var meanCosine: Double? {
            cosines.isEmpty ? nil : cosines.reduce(0, +) / Double(cosines.count)
        }
        public var meanNormRatio: Double? {
            let ratios = perLayer.compactMap(\.normRatio)
            return ratios.isEmpty ? nil : ratios.reduce(0, +) / Double(ratios.count)
        }
        public var skippedZeroNormLayers: Int {
            perLayer.count { $0.cosine == nil }
        }
        public var passed: Bool {
            guard let minCosine else { return false }
            return minCosine >= threshold
        }

        /// The pinned cross-engine JSON report — the KEY SET must stay
        /// identical to the Python `ParityReport.to_dict()` (the committed
        /// goldens assert it on both engines), and the FORMATTING matches
        /// Python's `json.dumps(indent=2, sort_keys=True)` so a cross-engine
        /// `diff` of two `--json` files is trivial.
        public func jsonText() -> String {
            func summary(_ a: ArtifactSummary) -> PythonStyleJSON {
                .object([
                    "hiddenSize": .int(a.hiddenSize),
                    "layerCount": .int(a.layerCount),
                    "name": .string(a.name),
                ])
            }
            let value = PythonStyleJSON.object([
                "artifactA": summary(artifactA),
                "artifactB": summary(artifactB),
                "comparedLayerCount": .int(comparedLayerCount),
                "layerCountMismatch": .bool(layerCountMismatch),
                "pass": .bool(passed),
                "perLayer": .array(
                    perLayer.map { row in
                        .object([
                            "cosine": row.cosine.map { .double($0) } ?? .null,
                            "layer": .int(row.layer),
                            "normA": .double(row.normA),
                            "normB": .double(row.normB),
                            "normRatio": row.normRatio.map { .double($0) } ?? .null,
                        ])
                    }),
                "summary": .object([
                    "meanCosine": meanCosine.map { .double($0) } ?? .null,
                    "meanNormRatio": meanNormRatio.map { .double($0) } ?? .null,
                    "minCosine": minCosine.map { .double($0) } ?? .null,
                    "skippedZeroNormLayers": .int(skippedZeroNormLayers),
                ]),
                "threshold": .double(threshold),
            ])
            return value.rendered(indent: 0) + "\n"
        }
    }

    // MARK: - Comparison

    /// Pure comparison over in-memory per-layer vectors (unit-test surface).
    public static func compare(
        nameA: String, perLayerA: [[Float]],
        nameB: String, perLayerB: [[Float]],
        threshold: Double = defaultThreshold
    ) throws -> Report {
        let hiddenA = perLayerA.first?.count ?? 0
        let hiddenB = perLayerB.first?.count ?? 0
        guard hiddenA == hiddenB else {
            throw ParityError(
                reason: "hidden-size mismatch: \(nameA) is \(hiddenA)-dim, "
                    + "\(nameB) is \(hiddenB)-dim — these artifacts are not comparable",
                kind: .incomparableArtifacts)
        }
        let compared = min(perLayerA.count, perLayerB.count)
        var rows: [LayerComparison] = []
        for layer in 0 ..< compared {
            let va = perLayerA[layer]
            let vb = perLayerB[layer]
            guard va.count == vb.count else {
                throw ParityError(
                    reason: "hidden-size mismatch at layer_\(layer): "
                        + "\(va.count) vs \(vb.count)",
                    kind: .incomparableArtifacts)
            }
            let na = norm(va)
            let nb = norm(vb)
            let cosine: Double? = (na > 0 && nb > 0) ? dot(va, vb) / (na * nb) : nil
            let ratio: Double? = na > 0 ? nb / na : nil
            rows.append(
                LayerComparison(
                    layer: layer, normA: na, normB: nb, cosine: cosine, normRatio: ratio))
        }
        return Report(
            artifactA: ArtifactSummary(
                name: nameA, layerCount: perLayerA.count, hiddenSize: hiddenA),
            artifactB: ArtifactSummary(
                name: nameB, layerCount: perLayerB.count, hiddenSize: hiddenB),
            perLayer: rows,
            threshold: threshold)
    }

    /// Loads and compares two artifacts by path (`…/name.safetensors`,
    /// `…/name.json`, or the extension-less artifact base path).
    public static func compareArtifacts(
        pathA: URL, pathB: URL, threshold: Double = defaultThreshold
    ) throws -> Report {
        let a = try loadArtifact(at: pathA)
        let b = try loadArtifact(at: pathB)
        return try compare(
            nameA: a.name, perLayerA: a.perLayer,
            nameB: b.name, perLayerB: b.perLayer,
            threshold: threshold)
    }

    // Double-precision SEQUENTIAL accumulation — the Python twin uses the
    // identical loop order, so tiny fixtures agree far below the 1e-6 bar.
    private static func norm(_ values: [Float]) -> Double {
        var total = 0.0
        for x in values { total += Double(x) * Double(x) }
        return total.squareRoot()
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Double {
        var total = 0.0
        for index in 0 ..< a.count { total += Double(a[index]) * Double(b[index]) }
        return total
    }

    // MARK: - Artifact loading (pure CPU, no MLX)

    /// Reads `<base>.safetensors` (+ required `<base>.json` sidecar for the
    /// layer count — the same contract as `vector_store.load` on the server).
    public static func loadArtifact(at path: URL) throws
        -> (name: String, perLayer: [[Float]])
    {
        var base = path
        if ["safetensors", "json"].contains(base.pathExtension) {
            base = base.deletingPathExtension()
        }
        let name = base.lastPathComponent
        let sidecarURL = base.appendingPathExtension("json")
        guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
            throw ParityError(
                reason: "no vector sidecar at \(sidecarURL.path) — pass the "
                    + "artifact's .safetensors (or base) path with its sidecar beside it")
        }
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(contentsOf: sidecarURL))
        let tensors = try readSafetensorsF32(
            url: base.appendingPathExtension("safetensors"))
        let perLayer: [[Float]] = try (0 ..< sidecar.layerCount).map { index in
            guard let values = tensors["layer_\(index)"] else {
                throw ParityError(
                    reason: "\(name).safetensors is missing layer_\(index) "
                        + "(sidecar declares \(sidecar.layerCount) layers)")
            }
            return values
        }
        return (name, perLayer)
    }

    /// Minimal safetensors parser for float32 tensors: 8-byte little-endian
    /// header length, JSON header `{tensor: {dtype, shape, data_offsets}}`
    /// (offsets relative to the end of the header), raw little-endian data.
    static func readSafetensorsF32(url: URL) throws -> [String: [Float]] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ParityError(reason: "cannot read \(url.path): \(error)")
        }
        guard data.count >= 8 else {
            throw ParityError(reason: "\(url.path): not a safetensors file (< 8 bytes)")
        }
        let headerLength = data.prefix(8).withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let dataStart = 8 + Int(headerLength)
        guard headerLength > 0, dataStart <= data.count else {
            throw ParityError(reason: "\(url.path): corrupt safetensors header length")
        }
        let headerObject: [String: Any]
        do {
            headerObject =
                try JSONSerialization.jsonObject(
                    with: data.subdata(in: 8 ..< dataStart)) as? [String: Any] ?? [:]
        } catch {
            throw ParityError(reason: "\(url.path): unreadable safetensors header: \(error)")
        }
        var tensors: [String: [Float]] = [:]
        for (key, rawSpec) in headerObject where key != "__metadata__" {
            guard
                let spec = rawSpec as? [String: Any],
                let dtype = spec["dtype"] as? String,
                let offsets = spec["data_offsets"] as? [Any],
                offsets.count == 2,
                let begin = (offsets[0] as? NSNumber)?.intValue,
                let end = (offsets[1] as? NSNumber)?.intValue
            else {
                throw ParityError(reason: "\(url.path): malformed entry for tensor '\(key)'")
            }
            guard dtype == "F32" else {
                throw ParityError(
                    reason: "\(url.path): tensor '\(key)' is \(dtype), expected F32 "
                        + "(the steering-vector artifact contract)")
            }
            let byteCount = end - begin
            guard
                begin >= 0, end >= begin, byteCount % 4 == 0,
                dataStart + end <= data.count
            else {
                throw ParityError(reason: "\(url.path): tensor '\(key)' offsets out of range")
            }
            let bytes = data.subdata(in: (dataStart + begin) ..< (dataStart + end))
            let values: [Float] = bytes.withUnsafeBytes { raw in
                (0 ..< byteCount / 4).map { index in
                    Float(
                        bitPattern: UInt32(
                            littleEndian: raw.loadUnaligned(
                                fromByteOffset: index * 4, as: UInt32.self)))
                }
            }
            tensors[key] = values
        }
        return tensors
    }
}

/// JSON emitter matching Python's `json.dumps(value, indent=2,
/// sort_keys=True)` byte for byte (sorted keys, two-space indent, `": "`
/// separators, `ensure_ascii` escaping, shortest round-trip floats — Swift's
/// `Double` description and Python's `repr` both emit shortest-round-trip
/// decimals). Internal to the parity report; not a general JSON library.
enum PythonStyleJSON {
    case object([String: PythonStyleJSON])
    case array([PythonStyleJSON])
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null

    func rendered(indent: Int) -> String {
        let pad = String(repeating: " ", count: indent)
        let childPad = String(repeating: " ", count: indent + 2)
        switch self {
        case .object(let entries):
            guard !entries.isEmpty else { return "{}" }
            let body = entries.keys.sorted().map { key in
                "\(childPad)\(Self.escaped(key)): "
                    + (entries[key] ?? .null).rendered(indent: indent + 2)
            }.joined(separator: ",\n")
            return "{\n\(body)\n\(pad)}"
        case .array(let items):
            guard !items.isEmpty else { return "[]" }
            let body = items.map { "\(childPad)\($0.rendered(indent: indent + 2))" }
                .joined(separator: ",\n")
            return "[\n\(body)\n\(pad)]"
        case .string(let value):
            return Self.escaped(value)
        case .int(let value):
            return String(value)
        case .double(let value):
            return Self.pythonRepr(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .null:
            return "null"
        }
    }

    /// Python `repr` float formatting for the values JSON can carry.
    private static func pythonRepr(_ value: Double) -> String {
        // Whole-number doubles print with a trailing ".0" in both languages
        // ("1.0"); Swift's default description already does this, so the
        // shortest-round-trip forms agree across the report's value range.
        "\(value)"
    }

    private static func escaped(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 || scalar.value > 0x7E {
                    // ensure_ascii: non-ASCII (and control) escaped as \uXXXX
                    // (surrogate pairs for astral scalars, like Python).
                    for unit in String(scalar).utf16 {
                        out += String(format: "\\u%04x", unit)
                    }
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
