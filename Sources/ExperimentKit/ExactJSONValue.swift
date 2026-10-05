import Foundation

/// A JSON value kept exactly as it was read, for the manifest keys this build
/// does not model (`ExperimentManifest.unknownTopLevelFields`).
///
/// `JSONValue` holds every number as a `Double`, which cannot represent an
/// integer above 2^53 exactly: a 64-bit seed of 9007199254740993 written by a
/// newer client came back from an older app's draft save as 9007199254740992.
/// Here an integer stays an integer: `Int64` when it fits, `UInt64` above that,
/// and only a number with a fraction or an exponent, or an integer beyond
/// 64 bits, is a `Double`. Values are preserved, not their spelling: key order
/// and whitespace follow the manifest writer, and `3.0` may be written `3`.
public enum ExactJSONValue: Codable, Sendable, Equatable {
    case string(String)
    case integer(Int64)
    case unsigned(UInt64)
    case number(Double)
    case bool(Bool)
    case object([String: ExactJSONValue])
    case array([ExactJSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(UInt64.self) {
            self = .unsigned(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([ExactJSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: ExactJSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .unsigned(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}
