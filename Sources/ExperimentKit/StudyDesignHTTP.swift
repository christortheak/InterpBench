import Foundation

/// Explicit workbench adapters over design authorship, independent of selection.
enum StudyDesignHTTP {
    enum Operation: String, Sendable { case list, inspect, describe, instantiate, save, update }
    private struct Request: Decodable {
        let workspaceRoot: String
        let name: String?
        let description: String?
        let designFileSHA256: String?
        let casting: JSONValue?
        let studyName: String?
        let sourceStudy: String?
        let manifestFileSHA256: String?
    }

    static func perform(_ operation: Operation, body: Data, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        var allowed: Set<String> = ["workspaceRoot"]
        if operation != .list { allowed.insert("name") }
        if operation == .describe { allowed.formUnion(["description", "designFileSHA256"]) }
        if operation == .instantiate { allowed.formUnion(["casting", "studyName", "designFileSHA256"]) }
        if operation == .save { allowed.formUnion(["sourceStudy", "manifestFileSHA256", "description"]) }
        if operation == .update { allowed.formUnion(["sourceStudy", "manifestFileSHA256", "designFileSHA256"]) }
        guard let fields = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            Set(fields.keys).isSubset(of: allowed),
            let request = try? JSONDecoder().decode(Request.self, from: body),
            request.workspaceRoot.hasPrefix("/"),
            operation == .list || operation == .save || request.name != nil,
            (operation != .save && operation != .update) || request.sourceStudy != nil,
            operation != .describe || request.description != nil,
            operation != .instantiate || request.casting != nil else {
            return .failure("invalidDesignRequest", "Name the workspace and the design operation's fields explicitly.",
                repair: "Use only: " + allowed.sorted().joined(separator: ", "))
        }
        if (operation == .save || operation == .update), request.manifestFileSHA256 == nil {
            return .failure("source_precondition_required", "Saving a design requires the reviewed source study file digest.",
                repair: "Inspect the source study and supply its manifestFileSHA256 after reviewing it.", status: "428 Precondition Required")
        }
        if (operation == .describe || operation == .instantiate || operation == .update), request.designFileSHA256 == nil {
            return .failure("design_precondition_required", "This design operation requires the reviewed design file digest.",
                repair: "Inspect the named design and supply its designFileSHA256 after reviewing it.", status: "428 Precondition Required")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                return .failure("designWorkspaceChanged", "The workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace and inspect its design again.", status: "409 Conflict")
            }
            if operation == .save || operation == .update {
                let source = try StudyDesignSourceReview(study: DraftAuthoringSnapshot.review(name: request.sourceStudy!,
                    workspaceRoot: workspaceRoot, expectedFileSHA256: request.manifestFileSHA256!))
                let result: StudyDesignSaveResult
                if operation == .save {
                    result = try StudyDesignSaving.create(from: source, name: request.name, description: request.description)
                } else {
                    let design = try StudyDesignAuthoring.review(name: request.name!, workspaceRoot: workspaceRoot,
                        expectedFileSHA256: request.designFileSHA256!)
                    result = try StudyDesignSaving.update(from: source, reviewed: design)
                }
                return .json(try StudyDesignSavingDocument(result, source: source))
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
            if operation == .instantiate {
                let casting = try StudyDesignCastingInput.resolve(JSONEncoder().encode(request.casting!), reviewed: reviewed)
                let saved = try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: casting, studyName: request.studyName)
                return .json(try StudyAuthoringHTTP.Document(saved))
            }
            let saved = try StudyDesignAuthoring.updateDescription(request.description!, reviewed: reviewed)
            return .json(Inspected(changed: saved.file.sha256 != reviewed.file.sha256, design: try StudyDesignDocument(saved)))
        } catch let error as StudyDesignAuthoringError {
            let status = error.code == "designChanged" ? "412 Precondition Failed"
                : ["invalidDesignName", "invalidDesignPrecondition"].contains(error.code) ? "400 Bad Request" : "409 Conflict"
            return .failure(error.code, error.reason, repair: error.repairAction, status: status)
        } catch let error as ExperimentError {
            return StudyAuthoringHTTP.failure(error)
        } catch CocoaError.fileReadNoSuchFile {
            return .failure("designNotFound", "The named design or casting input does not exist in this workspace.",
                repair: "List the designs and inspect a name from that library.", status: "404 Not Found")
        } catch {
            return .failure("designUnreadable", error.localizedDescription,
                repair: "Check access to the named design and inspect its stored document before retrying.", status: "409 Conflict")
        }
    }
}
