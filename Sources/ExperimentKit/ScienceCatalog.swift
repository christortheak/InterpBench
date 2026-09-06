import Foundation
import CryptoKit

/// Immutable shipped references. A catalog record never grants execution authority.
public enum ScienceCatalog {
    public struct Method: Codable, Identifiable, Sendable {
        public let id: String
        public let title: String
        public let purpose: String
        public let guide: String
    }
    public struct Operation: Codable, Identifiable, Sendable {
        public let id: String
        public let method: String
        public let title: String
        public let engineCLI: String?
        public let mac: String
        public let http: String?
        public let compute: String
        public let outputs: [String]
        public let restriction: String

        enum CodingKeys: String, CodingKey { case id, method, title, engineCLI, mac, http, compute, outputs, restriction }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(id, forKey: .id); try c.encode(method, forKey: .method)
            try c.encode(title, forKey: .title)
            if let engineCLI { try c.encode(engineCLI, forKey: .engineCLI) } else { try c.encodeNil(forKey: .engineCLI) }
            try c.encode(mac, forKey: .mac); try c.encode(compute, forKey: .compute)
            try c.encode(outputs, forKey: .outputs); try c.encode(restriction, forKey: .restriction)
            if let http { try c.encode(http, forKey: .http) } else { try c.encodeNil(forKey: .http) }
        }
    }
    public struct Catalog: Codable, Sendable {
        public let schemaVersion: Int
        public let methods: [Method]
        public let operations: [Operation]
        public let scope: String
        public var catalogSHA256: String?
    }
    public struct Guide: Codable, Sendable {
        public let method: Method
        public let text: String
        public let guideSHA256: String
    }
    static func resource(_ name: String) throws -> Data {
        let files = try JSONDecoder().decode([String: String].self, from: Data(ScienceResourceText.filesJSON.utf8))
        guard let text = files[name] else { throw malformed("The installed scientific guide is missing. Rebuild this installation.") }
        return Data(text.utf8)
    }
    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func malformed(_ reason: String) -> ExperimentError {
        .malformed(reason, repair: "Use science list to choose a shipped method or operation. These commands read guidance; use the listed interface to execute.")
    }
    public static func catalog() throws -> Catalog {
        let data = try resource("catalog.json")
        var result = try JSONDecoder().decode(Catalog.self, from: data)
        result.catalogSHA256 = digest(data)
        return result
    }
    public static func guide(_ id: String) throws -> Guide {
        guard let method = try catalog().methods.first(where: { $0.id == id }) else { throw malformed("Unknown scientific method.") }
        let data = try resource(method.guide)
        return Guide(method: method, text: String(decoding: data, as: UTF8.self), guideSHA256: digest(data))
    }
    public static func operation(_ id: String) throws -> Operation {
        guard let item = try catalog().operations.first(where: { $0.id == id }) else { throw malformed("Unknown scientific operation.") }
        return item
    }
    static func payload(_ value: some Encodable) throws -> [String: JSONValue] {
        try JSONDecoder().decode([String: JSONValue].self, from: JSONEncoder().encode(value))
    }
    static func run(_ invocation: ExperimentCLIInvocation, sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        guard args.count == (args.first == "list" ? 1 : 2) else { throw malformed("Supply exactly the declared arguments.") }
        let result: [String: JSONValue]
        switch args[0] {
        case "list": result = try payload(catalog())
        case "guide":
            let guide = try guide(args[1])
            sink.out(guide.text)
            return .init(message: "Scientific method guide read; no execution performed.", payload: try payload(guide))
        case "operation": result = try payload(operation(args[1]))
        default: throw malformed("Unknown scientific reference verb.")
        }
        sink.out(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
        return .init(message: "Scientific workflow reference read; no execution performed.", payload: result)
    }
    static func http(kind: String, id: String?) -> StudyAuthoringHTTP.Response {
        do {
            switch kind {
            case "catalog": return .json(try catalog())
            case "guide": return .json(try guide(id ?? ""))
            case "operation": return .json(try operation(id ?? ""))
            default: throw malformed("Unknown scientific reference path.")
            }
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
