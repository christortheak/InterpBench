import Foundation
import SteeringKit

/// Common transport results for explicit study-authoring operations.
enum StudyAuthoringHTTP {
    struct Response: Sendable {
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

    struct Document: Encodable {
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

    static func failure(_ error: Error) -> Response {
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
