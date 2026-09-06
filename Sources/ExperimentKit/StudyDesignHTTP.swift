import Foundation

/// Explicit workbench adapters over design authorship, independent of selection.
enum StudyDesignHTTP {
    enum Operation: String, Sendable { case list, inspect, describe }
    private struct Request: Decodable {
        let workspaceRoot: String
        let name: String?
        let description: String?
        let designFileSHA256: String?
    }

    static func perform(_ operation: Operation, body: Data, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        var allowed: Set<String> = ["workspaceRoot"]
        if operation != .list { allowed.insert("name") }
        if operation == .describe { allowed.formUnion(["description", "designFileSHA256"]) }
        guard let fields = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            Set(fields.keys).isSubset(of: allowed),
            let request = try? JSONDecoder().decode(Request.self, from: body),
            request.workspaceRoot.hasPrefix("/"),
            operation == .list || request.name != nil,
            operation != .describe || request.description != nil else {
            return .failure("invalidDesignRequest", "Name the workspace and the design operation's fields explicitly.",
                repair: "Use only: " + allowed.sorted().joined(separator: ", "))
        }
        if operation == .describe, request.designFileSHA256 == nil {
            return .failure("design_precondition_required", "A description edit requires the reviewed design file digest.",
                repair: "Inspect the named design and supply its designFileSHA256 after reviewing it.", status: "428 Precondition Required")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                return .failure("designWorkspaceChanged", "The workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace and inspect its design again.", status: "409 Conflict")
            }
            if operation == .list {
                struct Listed: Encodable { let ok = true; let catalog: StudyDesignCatalog }
                return .json(Listed(catalog: try StudyDesignAuthoring.list(workspaceRoot: workspaceRoot)))
            }
            struct Inspected: Encodable { let ok = true; let changed: Bool; let design: StudyDesignDocument }
            let name = request.name!
            if operation == .inspect {
                let snapshot = try StudyDesignSnapshot(workspaceRoot: workspaceRoot, name: name)
                return .json(Inspected(changed: false, design: try StudyDesignDocument(snapshot)))
            }
            let reviewed = try StudyDesignAuthoring.review(name: name, workspaceRoot: workspaceRoot,
                expectedFileSHA256: request.designFileSHA256!)
            let saved = try StudyDesignAuthoring.updateDescription(request.description!, reviewed: reviewed)
            return .json(Inspected(changed: saved.file.sha256 != reviewed.file.sha256, design: try StudyDesignDocument(saved)))
        } catch let error as StudyDesignAuthoringError {
            let status = error.code == "designChanged" ? "412 Precondition Failed"
                : ["invalidDesignName", "invalidDesignPrecondition"].contains(error.code) ? "400 Bad Request" : "409 Conflict"
            return .failure(error.code, error.reason, repair: error.repairAction, status: status)
        } catch CocoaError.fileReadNoSuchFile {
            return .failure("designNotFound", "The named design does not exist in this workspace.",
                repair: "List the designs and inspect a name from that library.", status: "404 Not Found")
        } catch {
            return .failure("designUnreadable", error.localizedDescription,
                repair: "Check access to the named design and inspect its stored document before retrying.", status: "409 Conflict")
        }
    }
}
