import Foundation

enum LocalModelPreparationHTTP {
    enum Operation: String { case plan, install, status, cancel }

    @MainActor static func perform(_ operation: Operation, body: Data, installer: LocalModelInstaller) -> StudyAuthoringHTTP.Response {
        if operation == .status { return .json(LocalModelPreparation.Status(installer)) }
        let allowed: Set<String> = operation == .cancel ? ["requestID"] : ["modelID", "revision"]
        guard let fields = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any], Set(fields.keys).isSubset(of: allowed) else {
            return .failure("invalidModelInstallRequest", "Use only the named operation's fields.", repair: "Allowed fields: " + allowed.sorted().joined(separator: ", "))
        }
        do {
            if operation == .cancel {
                guard let id = fields["requestID"] as? String, let requestID = UUID(uuidString: id) else {
                    return .failure("modelInstallPreconditionRequired", "Cancellation requires the installation request ID that was observed.",
                        repair: "Read local model status, review the request and supply requestID.", status: "428 Precondition Required")
                }
                try installer.cancel(requestID: requestID)
                return .json(LocalModelPreparation.Status(installer))
            }
            guard let model = fields["modelID"] as? String,
                fields["revision"] == nil || fields["revision"] is String else {
                return .failure("invalidModelInstallRequest", "Name modelID and an optional revision string.", repair: "Use the intended owner/repo and commit, branch or tag.")
            }
            let revision = fields["revision"] as? String
            if operation == .plan { return .json(try LocalModelPreparation.plan(modelID: model, revision: revision)) }
            return .json(try LocalModelPreparation.start(modelID: model, revision: revision, installer: installer), status: "202 Accepted")
        } catch let error as LocalModelPreparationError {
            return .failure(error.code, error.reason, repair: error.repairAction,
                status: error.code == "invalidModelInstallRequest" ? "400 Bad Request" : "409 Conflict")
        } catch {
            return .failure("modelPreparationFailed", error.localizedDescription,
                repair: "Inspect local installation status before retrying.", status: "500 Internal Server Error")
        }
    }
}
