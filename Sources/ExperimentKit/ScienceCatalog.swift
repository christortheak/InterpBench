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
    public struct Action: Codable, Identifiable, Sendable {
        public let id: String
        public let method: String
        public let path: String
        public let serviceRole: String
        public let authorityReason: String
    }
    public struct Access: Codable, Sendable {
        public let status: String
        public let client: String?
        public let macCLI: String?
        public let restriction: String
        enum CodingKeys: String, CodingKey { case status, client, macCLI, restriction }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(status, forKey: .status); try c.encode(restriction, forKey: .restriction)
            if let client { try c.encode(client, forKey: .client) } else { try c.encodeNil(forKey: .client) }
            if let macCLI { try c.encode(macCLI, forKey: .macCLI) } else { try c.encodeNil(forKey: .macCLI) }
        }
    }
    /// Where one operation runs: its status on each numerical backend, and
    /// a short phrase for the index. Generated into the catalog from
    /// `docs/substrate-capabilities.json`; guidance, never admission.
    public struct ExecutionProfile: Codable, Sendable, Equatable {
        public struct Backend: Codable, Sendable, Equatable {
            public let status: String
            public let label: String
        }
        public let profile: String
        public let runs: String
        /// Keyed by `cuda`, `mps` and `mlx`.
        public let backends: [String: Backend]
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
        public let actions: [Action]
        public let access: Access
        public let executionProfile: ExecutionProfile

        enum CodingKeys: String, CodingKey { case id, method, title, engineCLI, mac, http, compute, outputs, restriction, actions, access, executionProfile }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(id, forKey: .id); try c.encode(method, forKey: .method)
            try c.encode(title, forKey: .title)
            if let engineCLI { try c.encode(engineCLI, forKey: .engineCLI) } else { try c.encodeNil(forKey: .engineCLI) }
            try c.encode(mac, forKey: .mac); try c.encode(compute, forKey: .compute)
            try c.encode(outputs, forKey: .outputs); try c.encode(restriction, forKey: .restriction)
            try c.encode(actions, forKey: .actions); try c.encode(access, forKey: .access)
            if let http { try c.encode(http, forKey: .http) } else { try c.encodeNil(forKey: .http) }
            try c.encode(executionProfile, forKey: .executionProfile)
        }
    }
    /// What each compute choice can run, from the same inventory as each
    /// operation's profile: the choices, the study declarations a run must
    /// be able to carry out, and the rows of the app's What Runs Where
    /// table. Every yes and no here is derived by the generator from the
    /// per-backend statuses; nothing is set by hand.
    public struct WhereItRuns: Codable, Sendable, Equatable {
        public struct Choice: Codable, Sendable, Equatable, Identifiable {
            public let id: String
            public let title: String
            public let shortTitle: String
            public let engine: String
            public let backends: [String]
            public let computeSubstrate: String
            public let computeLocation: String?
        }
        public struct StudyFeature: Codable, Sendable, Equatable, Identifiable {
            public let id: String
            public let label: String
            public let phrase: String
            public let plural: Bool
            public let manifestKey: String
            public let backends: [String: String]
            /// Keyed by compute choice id.
            public let runsOn: [String: Bool]
            /// The sentence for each choice that cannot run it.
            public let advisories: [String: String]
        }
        public struct Activity: Codable, Sendable, Equatable, Identifiable {
            public let id: String
            public let activity: String
            public let runsOn: [String: Bool]
        }
        public let source: String
        public let statuses: [String: String]
        public let computeChoices: [Choice]
        public let studyFeatures: [StudyFeature]
        public let activities: [Activity]
        public let qualifiedAnywhere: Bool
    }
    public struct Catalog: Codable, Sendable {
        public let schemaVersion: Int
        public let methods: [Method]
        public let operations: [Operation]
        public let scope: String
        public let whereItRuns: WhereItRuns
        public var catalogSHA256: String?
    }
    public struct Guide: Codable, Sendable {
        public let method: Method
        public let text: String
        public let guideSHA256: String
    }
    public struct WorkflowField: Codable, Identifiable, Sendable {
        public let id: String
        public let label: String
        public let kind: String
        public let required: Bool
        public let help: String
        public let `default`: String?
        public let example: String?
    }
    public struct Workflow: Codable, Identifiable, Sendable {
        public let id: String
        public let title: String
        public let purpose: String
        public let claimBoundary: String
        public let fields: [WorkflowField]
    }
    public static func workflows() throws -> [Workflow] {
        struct Document: Decodable { let operations: [Workflow] }
        return try JSONDecoder().decode(Document.self, from: resource("workflows.json")).operations
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
    /// The catalog as a short index, for a caller choosing where to read next.
    public struct Brief: Codable, Sendable {
        public struct Method: Codable, Sendable {
            public let id: String
            public let title: String
            public let purpose: String
        }
        public struct Operation: Codable, Sendable {
            public let id: String
            public let method: String
            public let title: String
            public let purpose: String
            /// Where it runs, in a few words: its execution profile's phrase.
            public let runs: String
        }
        public let schemaVersion: Int
        public let brief: Bool
        public let catalogSHA256: String
        public let methods: [Method]
        public let operations: [Operation]
    }
    /// The text up to its first sentence break, as one line. Python twin:
    /// `science_catalog.first_sentence` — the same literal ". " split, so the
    /// two brief catalogs stay equal.
    static func firstSentence(_ text: String) -> String {
        let head = (text.range(of: ". ").map { String(text[..<$0.lowerBound]) } ?? text)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return head.hasSuffix(".") ? head : head + "."
    }
    /// Method and operation ids, titles, and one line of purpose each, plus
    /// the hash of the FULL catalog this was read from. Nothing here is new
    /// text: an operation's line is the first sentence of its guided
    /// workflow's purpose when it has one, and its method's purpose when it
    /// does not; its `runs` phrase is its execution profile's.
    /// `catalog()` is unchanged. Python twin: `science_catalog.brief`.
    public static func brief() throws -> Brief {
        let full = try catalog()
        let guided = Dictionary(try workflows().map { ($0.id, $0.purpose) }, uniquingKeysWith: { first, _ in first })
        let methods = Dictionary(full.methods.map { ($0.id, $0.purpose) }, uniquingKeysWith: { first, _ in first })
        return Brief(
            schemaVersion: full.schemaVersion, brief: true, catalogSHA256: full.catalogSHA256 ?? "",
            methods: full.methods.map { .init(id: $0.id, title: $0.title, purpose: $0.purpose) },
            operations: full.operations.map {
                .init(id: $0.id, method: $0.method, title: $0.title,
                      purpose: firstSentence(guided[$0.id] ?? methods[$0.method] ?? $0.title),
                      runs: $0.executionProfile.runs)
            })
    }
    /// Where a caller goes after the short index. Python twins:
    /// `science_commands.BRIEF_NEXT_VERB` and `BRIEF_NEXT_DETAIL`.
    static let briefNextVerb = "science guide <method>"
    static let briefNextDetail =
        "Read one method with science guide <method>, or one operation with science operation <operation>. "
        + "science list without --brief is the full catalog."
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
        // `--brief` is declared on `list` alone, so the strict parser has
        // already refused it anywhere else.
        let brief = invocation.args.contains("--brief")
        let args = invocation.args.filter { $0 != "--brief" }
        guard args.count == (args.first == "list" ? 1 : 2) else { throw malformed("Supply exactly the declared arguments.") }
        // Under --json the sink's stdout is stderr, and the document on stdout
        // already carries the whole result. Echoing the body again doubled what
        // a caller reading both streams paid for the catalog, so one line says
        // what was read and where it is. Python twin: `science_commands.summary`.
        let result: [String: JSONValue]
        let summary: String
        switch args[0] {
        case "list":
            let full = try catalog()
            result = try brief ? payload(self.brief()) : payload(full)
            summary = "science list: \(full.methods.count) methods and \(full.operations.count) operations "
                + "(catalog \((full.catalogSHA256 ?? "").prefix(12))…); the document is on stdout"
        case "guide":
            let guide = try guide(args[1])
            sink.out(invocation.json
                ? "science guide \(guide.method.id): \(guide.text.unicodeScalars.count) characters "
                    + "(guide \(guide.guideSHA256.prefix(12))…); the document is on stdout"
                : guide.text)
            return .init(message: "Scientific method guide read; no execution performed.", payload: try payload(guide))
        case "operation":
            let operation = try operation(args[1])
            result = try payload(operation)
            summary = "science operation \(operation.id): the document is on stdout"
        default: throw malformed("Unknown scientific reference verb.")
        }
        sink.out(invocation.json ? summary : String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
        return .init(message: "Scientific workflow reference read; no execution performed.", payload: result,
                     nextAction: brief ? .init(verb: briefNextVerb, detail: briefNextDetail) : nil)
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
