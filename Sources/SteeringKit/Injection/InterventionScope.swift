import Foundation

/// A description of the configured intervention, separate from study identity.
/// Prose is shared with the Python owner; numbers come from the actual object.
public struct InterventionScope: Codable, Sendable, Equatable {
    public var path: String
    public var site: String
    public var layers: [Int]
    public var positions: String
    public var prefill: String
    public var decode: String
    public var centering: String
    public var doseUnits: String
    public var control: String
    public var claimLimits: String
    public var detail: [String: Value]

    /// Scope details contain exact integer counts, scalar doses and nested maps.
    public indirect enum Value: Codable, Sendable, Equatable {
        case string(String), integer(Int), number(Double), object([String: Value]), null

        public init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { self = .null }
            else if let v = try? c.decode(Int.self) { self = .integer(v) }
            else if let v = try? c.decode(Double.self) { self = .number(v) }
            else if let v = try? c.decode(String.self) { self = .string(v) }
            else { self = .object(try c.decode([String: Value].self)) }
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .null: try c.encodeNil()
            case .string(let v): try c.encode(v)
            case .integer(let v): try c.encode(v)
            case .number(let v): try c.encode(v)
            case .object(let v): try c.encode(v)
            }
        }
    }

    public static func centeringSummary(_ values: [String]) -> String {
        let distinct = Set(values).sorted()
        if distinct.isEmpty { return InterventionScopeVocabulary.CENTERING_NONE }
        if distinct.count == 1 { return distinct[0] }
        return "mixed(" + distinct.joined(separator: ",") + ")"
    }
}
