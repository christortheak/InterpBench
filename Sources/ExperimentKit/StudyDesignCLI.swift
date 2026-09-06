import Foundation

/// Thin local CLI adapters over reviewed design authorship. No panel selection,
/// server connection or unreviewed file-edit fallback is used.
enum StudyDesignCLI {
    static func run(_ invocation: ExperimentCLIInvocation, workspaceRoot: URL,
                    sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.indices.contains(index + 1) else { return nil }
            return args[index + 1]
        }
        do {
            switch args.first {
            case "list":
                guard args.count == 1 else { throw usage() }
                let catalog = try StudyDesignAuthoring.list(workspaceRoot: workspaceRoot)
                for item in catalog.entries { sink.out("\(item.name)  \(item.description)") }
                for issue in catalog.issues { sink.err(issue + "\n") }
                let payload = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(catalog))
                return ExperimentCLIResult(message: "\(catalog.entries.count) design(s)", payload: ["catalog": payload])
            case "inspect":
                guard args.count == 2 else { throw usage() }
                let read = try StudyDesignSnapshot(workspaceRoot: workspaceRoot, name: args[1])
                return try result(read, changed: false, sink: sink)
            case "instantiate":
                guard args.count >= 2, let expected = flag("--file-sha256"), let castingPath = flag("--casting") else { throw usage() }
                let reviewed = try StudyDesignAuthoring.review(name: args[1], workspaceRoot: workspaceRoot, expectedFileSHA256: expected)
                let data = try Data(contentsOf: URL(fileURLWithPath: castingPath))
                let casting = try StudyDesignCastingInput.resolve(data, reviewed: reviewed)
                let saved = try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: casting, studyName: flag("--study-name"))
                let encoded = try JSONEncoder().encode(StudyAuthoringHTTP.Document(saved))
                sink.out(String(decoding: encoded, as: UTF8.self))
                return ExperimentCLIResult(message: "Draft created from reviewed design.", changed: true,
                    payload: try JSONDecoder().decode([String: JSONValue].self, from: encoded))
            case "describe":
                guard args.count >= 2, let description = flag("--description"), let expected = flag("--file-sha256") else { throw usage() }
                let reviewed = try StudyDesignAuthoring.review(name: args[1], workspaceRoot: workspaceRoot, expectedFileSHA256: expected)
                let saved = try StudyDesignAuthoring.updateDescription(description, reviewed: reviewed)
                return try result(saved, changed: saved.file.sha256 != reviewed.file.sha256, sink: sink)
            default: throw usage()
            }
        } catch let error as StudyDesignAuthoringError {
            let malformed = ["invalidDesignName", "invalidDesignPrecondition"].contains(error.code)
            throw ExperimentCLIStop(exitCode: malformed ? 64 : 65, state: malformed ? .blocked : .refused,
                code: error.code, reason: error.reason, repairAction: error.repairAction)
        } catch CocoaError.fileReadNoSuchFile {
            throw ExperimentCLIStop(exitCode: 66, state: .notFound, code: "designNotFound",
                reason: "The named design or casting file does not exist.", repairAction: "Use design list --json and inspect a design from that library.")
        }
    }

    private static func result(_ snapshot: StudyDesignSnapshot, changed: Bool,
                               sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let value = try StudyDesignDocument(snapshot)
        let data = try JSONEncoder().encode(value)
        let payload = try JSONDecoder().decode([String: JSONValue].self, from: data)
        sink.out(String(decoding: data, as: UTF8.self))
        return ExperimentCLIResult(message: changed ? "Design description saved." : "Design inspected.", changed: changed, payload: payload)
    }

    private static func usage() -> ExperimentError {
        .malformed("Use design list, inspect <name>, describe <name> --description <text> --file-sha256 <digest>, or instantiate <name> --casting <file> --file-sha256 <digest>.",
            repair: "steerlab-cli design --help")
    }
}
