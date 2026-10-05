import Foundation

/// `experiment rename|delete`, `design rename|delete`, and `agent delete`.
///
/// Every verb previews by default. A preview reads the current state, applies
/// every rule, and prints what would change together with the reviewed file
/// digest; nothing is written. The change happens only when the same command
/// is repeated with that digest and `--yes`, and it is refused if the file
/// changed in between. The rules and the moves live in
/// `WorkspaceHousekeeping`, which the app's buttons also use.
enum WorkspaceHousekeepingCLI {

    static func run(
        _ invocation: ExperimentCLIInvocation, workspaceRoot: URL, sink: ExperimentCLISink
    ) throws -> ExperimentCLIResult {
        let family = invocation.namespace
        guard let verb = invocation.args.first, ["rename", "delete"].contains(verb) else {
            throw usage(family: family, verb: invocation.args.first ?? "")
        }
        let digestFlag = Self.digestFlag(family)
        let (positionals, digest, confirmed) = try parse(invocation.args.dropFirst(), digestFlag: digestFlag)
        let expected = positionals.count == (verb == "rename" ? 2 : 1)
        guard expected else { throw usage(family: family, verb: verb) }
        if confirmed && digest == nil {
            throw ExperimentError.malformed(
                "--yes needs the reviewed \(digestFlag) from the preview.",
                repair: "\(ExperimentCLIHelp.program) \(family) \(verb) "
                    + positionals.joined(separator: " ")
                    + "  (preview it, then repeat with \(digestFlag) <digest> --yes)")
        }
        let apply = confirmed
        let review: WorkspaceHousekeeping.Review
        do {
            switch (family, verb) {
            case ("experiment", "rename"):
                if let digest, apply {
                    review = try WorkspaceHousekeeping.renameStudy(
                        name: positionals[0], to: positionals[1],
                        expectedFileSHA256: digest, workspaceRoot: workspaceRoot)
                } else {
                    review = try reviewed(
                        WorkspaceHousekeeping.reviewStudyRename(
                            name: positionals[0], to: positionals[1], workspaceRoot: workspaceRoot),
                        digest: digest, current: \.manifestFileSHA256)
                }
            case ("experiment", "delete"):
                if let digest, apply {
                    review = try WorkspaceHousekeeping.deleteStudy(
                        name: positionals[0], expectedFileSHA256: digest, workspaceRoot: workspaceRoot)
                } else {
                    review = try reviewed(
                        WorkspaceHousekeeping.reviewStudyDelete(
                            name: positionals[0], workspaceRoot: workspaceRoot),
                        digest: digest, current: \.manifestFileSHA256)
                }
            case ("design", "rename"):
                if let digest, apply {
                    review = try WorkspaceHousekeeping.renameTemplate(
                        name: positionals[0], to: positionals[1],
                        expectedFileSHA256: digest, workspaceRoot: workspaceRoot)
                } else {
                    review = try reviewed(
                        WorkspaceHousekeeping.reviewTemplateRename(
                            name: positionals[0], to: positionals[1], workspaceRoot: workspaceRoot),
                        digest: digest, current: \.designFileSHA256)
                }
            case ("design", "delete"):
                if let digest, apply {
                    review = try WorkspaceHousekeeping.deleteTemplate(
                        name: positionals[0], expectedFileSHA256: digest, workspaceRoot: workspaceRoot)
                } else {
                    review = try reviewed(
                        WorkspaceHousekeeping.reviewTemplateDelete(
                            name: positionals[0], workspaceRoot: workspaceRoot),
                        digest: digest, current: \.designFileSHA256)
                }
            case ("agent", "delete"):
                if let digest, apply {
                    review = try WorkspaceHousekeeping.deleteAgent(
                        path: positionals[0], expectedFileSHA256: digest, workspaceRoot: workspaceRoot)
                } else {
                    review = try reviewed(
                        WorkspaceHousekeeping.reviewAgentDelete(
                            path: positionals[0], workspaceRoot: workspaceRoot),
                        digest: digest, current: \.artifactFileSHA256)
                }
            default:
                throw usage(family: family, verb: verb)
            }
        } catch let error as StudyDesignAuthoringError {
            let malformed = ["invalidDesignName", "invalidDesignPrecondition"].contains(error.code)
            throw ExperimentCLIStop(
                exitCode: malformed ? 64 : 65, state: malformed ? .blocked : .refused,
                code: error.code, reason: error.reason, repairAction: error.repairAction)
        } catch let error as WorkspaceHousekeeping.Refusal {
            var payload: [String: JSONValue] = ["path": .string(error.path)]
            if error.code == WorkspaceHousekeeping.agentInUseCode {
                payload["usedBy"] = .array(error.usedBy.map { .string($0) })
            }
            throw ExperimentCLIStop(
                exitCode: 65, state: .refused, code: error.code,
                reason: error.reason, repairAction: error.repairAction, payload: payload)
        } catch CocoaError.fileReadNoSuchFile where family == "design" {
            throw ExperimentCLIStop(
                exitCode: 66, state: .notFound, code: "designNotFound",
                reason: "There is no template named '\(positionals[0])' in this workspace.",
                repairAction: "\(ExperimentCLIHelp.program) design list  (the templates this "
                    + "workspace holds)")
        }
        return try result(review, family: family, verb: verb, positionals: positionals,
                          digestFlag: digestFlag, sink: sink)
    }

    /// The reviewed-digest flag each family already uses for its other
    /// reviewed edits, so the digest an inspection printed is the one typed.
    static func digestFlag(_ family: String) -> String {
        switch family {
        case "design": "--file-sha256"
        case "agent": "--artifact-sha256"
        default: "--manifest-sha256"
        }
    }

    /// A digest given without `--yes` is checked, so a caller can confirm it
    /// still matches before asking the researcher.
    private static func reviewed(
        _ review: WorkspaceHousekeeping.Review, digest: String?,
        current: KeyPath<WorkspaceHousekeeping.Review, String?>
    ) throws -> WorkspaceHousekeeping.Review {
        guard let digest, digest != review[keyPath: current] else { return review }
        throw ExperimentError.refusing(
            .staleManifest, "The file changed after the digest you gave was read.",
            repair: "Read the new preview, show the researcher what changed, and use its digest.")
    }

    private static func parse(
        _ arguments: ArraySlice<String>, digestFlag: String
    ) throws -> (positionals: [String], digest: String?, confirmed: Bool) {
        var positionals: [String] = []
        var digest: String?
        var confirmed = false
        var iterator = arguments.makeIterator()
        while let argument = iterator.next() {
            if argument == digestFlag {
                digest = iterator.next()
            } else if argument == "--yes" {
                confirmed = true
            } else {
                positionals.append(argument)
            }
        }
        return (positionals, digest, confirmed)
    }

    private static func result(
        _ review: WorkspaceHousekeeping.Review, family: String, verb: String,
        positionals: [String], digestFlag: String, sink: ExperimentCLISink
    ) throws -> ExperimentCLIResult {
        let data = try JSONEncoder().encode(review)
        var payload = try JSONDecoder().decode([String: JSONValue].self, from: data)
        let subject = family == "agent" ? "agent" : family == "design" ? "template" : "study"
        if review.applied {
            let line = verb == "rename"
                ? "Renamed \(subject) '\(review.name)' to '\(review.newName ?? "")'."
                : "Moved \(subject) '\(review.name)' to \(review.destination)/."
            sink.out(line)
            for effect in review.effects where !effect.hasPrefix("Moves ") { sink.out(effect) }
            // The review's own `advisories` stay in the payload: the
            // envelope's advisory vocabulary is closed and cross-engine.
            return ExperimentCLIResult(message: line, changed: true, payload: payload)
        }
        let digest = review.manifestFileSHA256 ?? review.designFileSHA256
            ?? review.artifactFileSHA256 ?? ""
        let confirm = "\(family) \(verb) " + positionals.joined(separator: " ")
            + " \(digestFlag) \(digest) --yes"
        payload["confirmCommand"] = .string("\(ExperimentCLIHelp.program) \(confirm)")
        sink.out("Preview — nothing has changed yet.")
        for effect in review.effects { sink.out("  " + effect) }
        for advisory in review.advisories { sink.out("  note: " + advisory) }
        sink.out("To apply, after the researcher agrees: \(ExperimentCLIHelp.program) \(confirm)")
        return ExperimentCLIResult(
            message: "Preview of \(verb) for \(subject) '\(review.name)'; nothing changed.",
            changed: false, payload: payload,
            nextAction: .init(
                verb: confirm, missingPermissionFlags: [digestFlag, "--yes"],
                detail: "Show the researcher what will change and apply only after they agree."))
    }

    private static func usage(family: String, verb: String) -> ExperimentError {
        let shape: String
        switch (family, verb) {
        case ("experiment", "rename"): shape = "experiment rename <name> <new-name>"
        case ("experiment", "delete"): shape = "experiment delete <name>"
        case ("design", "rename"): shape = "design rename <name> <new-name>"
        case ("design", "delete"): shape = "design delete <name>"
        default: shape = "agent delete <path>"
        }
        return .malformed(
            "usage: \(ExperimentCLIHelp.program) \(shape) [\(digestFlag(family)) <sha256> --yes]",
            repair: "\(ExperimentCLIHelp.program) \(family) \(verb) --help")
    }
}
