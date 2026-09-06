import Foundation
import SteeringKit

/// Swift HTTP adapter for explicit, reviewed protocol edits. No panel or editor state.
enum StudyProtocolHTTP {
    /// The closed protocol-body vocabulary — the keys
    /// `POST /api/experiment/protocol` may carry, in its Body's declaration
    /// order. A default `JSONDecoder` silently ignores keys the Body does not
    /// declare, so an out-of-vocabulary key used to write nothing while the
    /// route answered ok — the same silent loss the Python engine's
    /// `set_protocol` refuses (`experiment_store.PROTOCOL_FIELDS`; that
    /// vocabulary is the manifest's spellings, this one is the panel's).
    /// The adapter and its tests use the same vocabulary; no live editor is needed.
    static let bodyKeys: [String] = [
        "name", "workspaceRoot", "manifestFileSHA256",
        "description", "task", "outcomes", "judgeModel", "judgePrompt",
        "taskPromptsFile", "promptMode", "systemPrompt", "qwenThinkingEnabled",
        "reasoningEffort", "reasoningMaxTokens",
        "temperature", "maxTokens", "samplesPerItem", "seedPolicy",
        "exclusionRules",
    ]

    /// The body's top-level keys outside the vocabulary, sorted. A body that
    /// is not a JSON object answers `[]` — the typed decode already refuses
    /// those as "bad body".
    static func unknownBodyKeys(in body: Data) -> [String] {
        guard let object = (try? JSONSerialization.jsonObject(with: body))
            as? [String: Any] else { return [] }
        return object.keys.filter { !bodyKeys.contains($0) }.sorted()
    }

    struct Response {
        let status: String
        let body: Data
        var succeeded: Bool { status == "200 OK" }

        static func json(_ object: some Encodable, status: String = "200 OK") -> Self {
            do { return .init(status: status, body: try JSONEncoder().encode(object)) }
            catch {
                return .init(status: "500 Internal Server Error",
                             body: Data(#"{"ok":false,"code":"encodingFailed","error":"Could not encode the operation result."}"#.utf8))
            }
        }

        static func failure(_ code: String, _ reason: String, repair: String,
                            status: String = "400 Bad Request") -> Self {
            struct Failure: Encodable {
                let ok = false
                let code: String
                let error: String
                let repairAction: String
            }
            return .json(Failure(code: code, error: reason, repairAction: repair), status: status)
        }
    }

    private struct Document: Encodable {
        let ok = true
        let name: String
        let workspaceRoot: String
        let manifestFileSHA256: String
        let document: JSONValue
        var advisories: [String] = []

        init(_ snapshot: DraftAuthoringSnapshot, advisories: [String] = []) throws {
            name = snapshot.manifest.name
            workspaceRoot = snapshot.workspaceRoot.path
            manifestFileSHA256 = snapshot.file.sha256
            document = try JSONDecoder().decode(JSONValue.self, from: snapshot.file.data)
            self.advisories = advisories
        }
    }

    static func read(name: String?, workspaceRoot: URL) -> Response {
        guard let name, !name.isEmpty else {
            return .failure("targetRequired", "Name the study to read.",
                            repair: "GET /api/experiment/manifest?name=<study>")
        }
        do {
            return .json(try Document(DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: name)))
        } catch { return failure(error) }
    }

    private struct Body: Decodable {
        let name: String?
        let workspaceRoot: String?
        let manifestFileSHA256: String?
        let description: String?
        let task: String?
        let outcomes: String?
        let judgeModel: String?
        let judgePrompt: String?
        let taskPromptsFile: String?
        let promptMode: ExperimentManifest.PromptMode?
        let systemPrompt: String?
        let qwenThinkingEnabled: Bool?
        let reasoningEffort: String?
        let reasoningMaxTokens: Int?
        let temperature: Double?
        let maxTokens: Int?
        let samplesPerItem: Int?
        let seedPolicy: String?
        let exclusionRules: [ExclusionRule]?
    }

    static func apply(body: Data, workspaceRoot: URL) -> Response {
        let unknown = unknownBodyKeys(in: body)
        guard unknown.isEmpty else {
            return .failure("unknownProtocolField", "unknown protocol field(s) "
                + unknown.map { "'\($0)'" }.joined(separator: ", ") + "; nothing was written",
                repair: "Use only: " + bodyKeys.joined(separator: ", "))
        }
        guard let request = try? JSONDecoder().decode(Body.self, from: body) else {
            return .failure("invalidProtocolRequest", "Protocol request must be a JSON object with typed fields.",
                            repair: "Read the protocol request schema in docs/DRAFT-AUTHORING-PRECONDITIONS.md.")
        }
        guard let name = request.name, !name.isEmpty,
              let expectedRoot = request.workspaceRoot, expectedRoot.hasPrefix("/") else {
            return .failure("targetRequired", "Name the study and workspace returned by the manifest read.",
                            repair: "GET /api/experiment/manifest?name=<study>; review the document before saving.")
        }
        guard let expected = request.manifestFileSHA256 else {
            return .failure("manifest_precondition_required", "Protocol edits require the reviewed file digest.",
                            repair: "Read and review the named manifest, then supply its manifestFileSHA256.",
                            status: "428 Precondition Required")
        }
        guard expected.count == 64, expected.allSatisfy({ "0123456789abcdef".contains($0) }) else {
            return .failure("invalid_manifest_precondition", "manifestFileSHA256 must be a lowercase SHA-256 digest.",
                            repair: "Use the manifestFileSHA256 returned by the manifest read.")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: expectedRoot))
                    == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                throw ExperimentError.refusing(.staleManifest, "The HTTP workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace, read its named manifest, review and apply the edit again.")
            }
            // Reading current bytes here is safe only because their digest must
            // equal the caller's previously reviewed digest. Publication checks
            // again under the shared lock, closing the read-to-write race.
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: name)
            guard reviewed.file.sha256 == expected else {
                throw ExperimentError.refusing(.staleManifest, "The study changed since the supplied manifest read.",
                    repair: "Read and review the named manifest again, then reconstruct the intended edit.")
            }
            var fields = StudyProtocolFields(manifest: reviewed.manifest)
            if let value = request.description { fields.protocolDescription = value }
            if let value = request.task { fields.taskDescription = value }
            if let value = request.outcomes { fields.outcomeMeasures = value }
            if let value = request.judgeModel {
                let model = value.trimmingCharacters(in: .whitespacesAndNewlines)
                fields.inlineJudgeModel = model.isEmpty ? reviewed.manifest.modelID : model
            }
            if let value = request.judgePrompt { fields.evaluationPrompt = value }
            if let value = request.taskPromptsFile { fields.taskPromptsFile = value }
            if let value = request.promptMode { fields.promptMode = value }
            if let value = request.systemPrompt { fields.systemPrompt = value }
            if let value = request.qwenThinkingEnabled {
                fields.reasoningEffort = ReasoningEffort.legacy(qwenThinkingEnabled: value).rawValue
                if !value { fields.reasoningMaxTokens = nil }
            }
            if let value = request.reasoningEffort { fields.reasoningEffort = value }
            if let value = request.reasoningMaxTokens {
                fields.reasoningMaxTokens = value > 0 ? value : nil
            }
            if let value = request.temperature { fields.temperature = value }
            if let value = request.maxTokens { fields.maxTokens = value }
            if let value = request.samplesPerItem { fields.samplesPerItem = value }
            if let value = request.seedPolicy { fields.seedPolicy = value }
            switch try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: fields, exclusionRules: request.exclusionRules)
            {
            case .saved(let saved, _, let advisories):
                return .json(try Document(saved, advisories: advisories))
            case .requiresScenario:
                return .failure("missingPrerequisite", "Select and pin a scenario before saving this multi-agent setup.",
                                repair: "Use the panel authoring/casting operation, then read and review the study again.",
                                status: "409 Conflict")
            }
        } catch { return failure(error) }
    }

    private static func failure(_ error: Error) -> Response {
        if let error = error as? ExperimentError {
            if let refusal = error.lifecycleRefusal {
                return .failure(refusal.gate.rawValue, refusal.reason, repair: refusal.repairAction,
                                status: refusal.gate == .staleManifest ? "412 Precondition Failed" : "409 Conflict")
            }
            return .failure(error.malformedInvocation == nil ? "authoringFailed" : "usage", error.reason,
                            repair: error.malformedInvocation?.repairAction ?? "Check the named study and its input files, then read and review it again.",
                            status: error.malformedInvocation == nil ? "409 Conflict" : "400 Bad Request")
        }
        if let error = error as? CocoaError, error.code == .fileReadNoSuchFile {
            return .failure("notFound", "The named study or input file does not exist.",
                            repair: "Check the workspace and study name before retrying.", status: "404 Not Found")
        }
        return .failure("authoringFailed", String(describing: error),
                        repair: "Check workspace and input-file access; read and review the document before retrying.",
                        status: "500 Internal Server Error")
    }
}
