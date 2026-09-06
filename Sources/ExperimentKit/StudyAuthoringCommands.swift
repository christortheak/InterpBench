import Foundation

/// Public transport adapters over the panel and pipeline authoring owners.
enum StudyAuthoringCommands {
    static func run(_ invocation: ExperimentCLIInvocation, root: URL, sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        func flag(_ name: String) throws -> String {
            guard let index = args.firstIndex(of: name), args.indices.contains(index + 1) else {
                throw StudyPanelAuthoring.malformed("Supply \(name) once.")
            }
            return args[index + 1]
        }
        guard args.count >= 2 else { throw StudyPanelAuthoring.malformed("Name the study or panel input.") }
        if invocation.namespace == "experiment" {
            let reviewed = try DraftAuthoringSnapshot.review(name: args[1], workspaceRoot: root, expectedFileSHA256: flag("--manifest-sha256"))
            let block = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: URL(fileURLWithPath: flag("--file"))))
            let saved = try StudyPipelineAuthoring.saveBlock(block == .null ? nil : block, reviewed: reviewed)
            return try result(StudyAuthoringHTTP.Document(saved), changed: saved.file.sha256 != reviewed.file.sha256, sink: sink)
        }
        if args[0] == "inspect" {
            return try result(StudyPanelAuthoring.inspect(path: args[1], root: root), changed: false, sink: sink)
        }
        if args[0] == "compile" {
            guard !["--seat", "--model", "--temperature", "--max-tokens", "--file-slug"].contains(where: args.contains) else {
                throw StudyPanelAuthoring.malformed("The reviewed casting-file form does not accept direct seat or model-setting flags; edit the study first.")
            }
            let reviewed = try DraftAuthoringSnapshot.review(name: flag("--experiment"), workspaceRoot: root, expectedFileSHA256: flag("--manifest-sha256"))
            let saved = try StudyPanelAuthoring.compile(path: args[1], expectedPanel: flag("--file-sha256"),
                casting: Data(contentsOf: URL(fileURLWithPath: flag("--casting"))), reviewed: reviewed)
            return try result(StudyAuthoringHTTP.Document(saved), changed: saved.file.sha256 != reviewed.file.sha256, sink: sink)
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: args[1]))
        let panel = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        try StudyPanelAuthoring.validate(panel)
        if args[0] == "check" {
            return try result(Review(document: panel, fileSHA256: MultiAgentScenarioStore.hash(data)), changed: false, sink: sink)
        }
        guard MultiAgentScenarioStore.hash(data) == (try flag("--file-sha256")) else {
            throw StudyPanelAuthoring.malformed("The proposed panel changed since review.")
        }
        let saved = try StudyPanelAuthoring.publish(panel, root: root)
        let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
        return try result(StudyPanelAuthoring.inspect(path: String(saved.record.url.path.dropFirst(prefix.count)), root: root), changed: saved.changed, sink: sink)
    }

    struct Review: Encodable { let document: MultiAgentScenario; let fileSHA256: String; let valid = true }

    static func result(_ value: some Encodable, changed: Bool, sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let data = try JSONEncoder().encode(value)
        sink.out(String(decoding: data, as: UTF8.self))
        return .init(message: "Authoring review complete.", changed: changed,
                     payload: try JSONDecoder().decode([String: JSONValue].self, from: data))
    }
}

enum StudyAuthoringOperationsHTTP {
    static func perform(operation: String, body: Data, root: URL) -> StudyAuthoringHTTP.Response {
        do {
            let allowed: Set<String>
            switch operation {
            case "pipeline": allowed = ["workspaceRoot", "name", "manifestFileSHA256", "document"]
            case "compile": allowed = ["workspaceRoot", "name", "manifestFileSHA256", "path", "fileSHA256", "casting"]
            case "inspect": allowed = ["workspaceRoot", "path"]
            case "check", "import": allowed = ["workspaceRoot", "source", "fileSHA256"]
            default: throw StudyPanelAuthoring.malformed("Unknown authoring operation.")
            }
            guard let fields = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                Set(fields.keys).isSubset(of: allowed),
                let selectedRoot = fields["workspaceRoot"] as? String,
                selectedRoot.hasPrefix("/"),
                try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: selectedRoot)) == ManifestFileTransaction.canonicalPath(root) else {
                throw StudyPanelAuthoring.malformed("Supply the intended workspaceRoot and only the operation's declared fields.")
            }
            func string(_ key: String) throws -> String {
                guard let value = fields[key] as? String else { throw StudyPanelAuthoring.malformed("Supply \(key).") }
                return value
            }
            if operation == "check" || operation == "import" {
                let data = Data(try string("source").utf8)
                let panel = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
                try StudyPanelAuthoring.validate(panel)
                if operation == "check" { return .json(StudyAuthoringCommands.Review(document: panel, fileSHA256: MultiAgentScenarioStore.hash(data))) }
                guard MultiAgentScenarioStore.hash(data) == (try string("fileSHA256")) else { throw StudyPanelAuthoring.malformed("Panel input changed after review.") }
                let record = try StudyPanelAuthoring.publish(panel, root: root).record
                let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
                return .json(try StudyPanelAuthoring.inspect(path: String(record.url.path.dropFirst(prefix.count)), root: root))
            }
            if operation == "inspect" { return .json(try StudyPanelAuthoring.inspect(path: string("path"), root: root)) }
            if operation == "pipeline" || operation == "compile" {
                let reviewed = try DraftAuthoringSnapshot.review(name: string("name"), workspaceRoot: root, expectedFileSHA256: string("manifestFileSHA256"))
                let saved: DraftAuthoringSnapshot
                if operation == "pipeline" {
                    guard fields.keys.contains("document") else { throw StudyPanelAuthoring.malformed("Supply document; null explicitly clears the pipeline.") }
                    let data = try JSONSerialization.data(withJSONObject: fields["document"]!, options: .fragmentsAllowed)
                    let block = try JSONDecoder().decode(JSONValue.self, from: data)
                    saved = try StudyPipelineAuthoring.saveBlock(block == .null ? nil : block, reviewed: reviewed)
                } else {
                    guard let casting = fields["casting"] as? [String: Any] else { throw StudyPanelAuthoring.malformed("Supply casting.") }
                    saved = try StudyPanelAuthoring.compile(path: string("path"), expectedPanel: string("fileSHA256"), casting: JSONSerialization.data(withJSONObject: casting), reviewed: reviewed)
                }
                return .json(try StudyAuthoringHTTP.Document(saved))
            }
            throw StudyPanelAuthoring.malformed("Unknown authoring operation.")
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
