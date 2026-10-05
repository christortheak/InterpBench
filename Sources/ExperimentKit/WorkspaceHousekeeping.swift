import Foundation

/// Rename and delete for what a researcher authored: draft studies, templates,
/// and agents.
///
/// These operations used to exist only behind app buttons, so a coding
/// assistant could only do them by hand-editing files the workspace guide says
/// to leave alone. Every operation here has two halves:
///
/// - a REVIEW, which reads the current state, applies every rule, and says in
///   plain sentences what the operation would change, together with the exact
///   file digest it read; and
/// - an APPLY, which takes that digest back, refuses if the file changed in
///   between, and then performs the change through the same store owner the
///   app uses (`ExperimentStore.rename`, `ExperimentStore.moveDraftToTrash`,
///   `StudyTemplateStore.rename` and `.delete`, `ModelVariantStore.delete`).
///
/// Nothing is erased. A deleted study, template, or agent moves into a
/// `.trash-<time>` folder beside where it lived, which every listing skips and
/// from which it can be moved back by hand. `runs/` is never rewritten: a
/// study's runs keep the name they recorded, and an agent a run saved is not
/// deleted (only entries in the agent library, `runs/model-variants/`, are).
///
/// The Python client has a twin of every rule here
/// (`steerlab_server/client/housekeeping.py`); the two print the same keys.
public enum WorkspaceHousekeeping {

    /// What a review or an apply reports. Both command lines print exactly
    /// these keys; absent fields are omitted rather than null.
    public struct Review: Encodable, Sendable, Equatable {
        public enum Kind: String, Encodable, Sendable { case study, template, agent }
        public enum Operation: String, Encodable, Sendable { case rename, delete }

        public let kind: Kind
        public let operation: Operation
        /// False for a preview, true once the change was made.
        public var applied: Bool
        /// The study or template name, or the agent's own name.
        public let name: String
        /// The resolved new name, for a rename.
        public let newName: String?
        /// The workspace-relative folder (study, template) or file (agent).
        public let path: String
        /// Where it goes. A preview of a delete shows the `.trash-<time>`
        /// pattern; an applied delete shows the folder actually used.
        public var destination: String
        public let manifestFileSHA256: String?
        public let designFileSHA256: String?
        public let artifactFileSHA256: String?
        /// The study's lifecycle status (always `draft` when the review passed).
        public let status: String?
        /// Runs that recorded the study's current name. They are never changed.
        public let runsRecordingName: Int?
        /// Studies created from the template. They are never changed.
        public let studiesFromTemplate: Int?
        /// Studies that use the agent. Empty whenever the review passed.
        public let usedBy: [String]?
        /// What the operation changes, in plain sentences.
        public var effects: [String]
        /// Things worth knowing that do not stop the operation.
        public var advisories: [String]
    }

    /// The refusal code for deleting an agent that a study still uses. Not a
    /// lifecycle gate: nothing about any study is wrong, the agent is simply
    /// in use.
    public static let agentInUseCode = "agentInUse"
    /// The refusal code for deleting an agent a run saved: run folders are
    /// evidence, and `runs/` is append-only.
    public static let agentIsRunEvidenceCode = "agentIsRunEvidence"

    // MARK: - Studies

    /// Preview renaming a draft study.
    public static func reviewStudyRename(
        name: String, to newName: String, workspaceRoot: URL
    ) throws -> Review {
        let study = try DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: name)
        return try reviewStudyRename(study, to: newName)
    }

    static func reviewStudyRename(_ study: DraftAuthoringSnapshot, to newName: String) throws -> Review {
        let name = study.manifest.name
        try requireDraft(study.manifest, operation: .rename)
        let target = ExperimentStore.resolvedRenameTarget(newName)
        guard target.contains(where: { $0.isLetter || $0.isNumber }) else {
            throw ExperimentError.malformed(
                "The new name needs at least one letter or digit.",
                repair: "Choose a name of lowercase letters, digits, and hyphens.")
        }
        guard target != name else {
            throw ExperimentError.malformed(
                "The study is already called '\(name)'.",
                repair: "Choose a different name.")
        }
        let destination = ExperimentRepository(workspaceRoot: study.workspaceRoot).directory
            .appending(component: target)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ExperimentError.malformed(
                "A study named '\(target)' already exists.",
                repair: "Choose a name that is not already in the study list.")
        }
        let runs = ExperimentStore.runsStamped(experimentName: name)
        var effects = [
            "Moves experiments/\(name)/ to experiments/\(target)/ and changes the "
                + "study's name inside it."
        ]
        if runs > 0 {
            effects.append(
                "\(runs) existing \(runs == 1 ? "run recorded" : "runs recorded") the "
                    + "name '\(name)'. Runs are never changed, so "
                    + "\(runs == 1 ? "it" : "they") will no longer be listed under "
                    + "'\(target)'.")
        }
        effects.append("Nothing in runs/ is changed.")
        return Review(
            kind: .study, operation: .rename, applied: false, name: name, newName: target,
            path: "experiments/\(name)", destination: "experiments/\(target)",
            manifestFileSHA256: study.file.sha256, designFileSHA256: nil,
            artifactFileSHA256: nil, status: study.manifest.status.rawValue,
            runsRecordingName: runs, studiesFromTemplate: nil, usedBy: nil,
            effects: effects, advisories: [])
    }

    /// Rename a draft study the caller reviewed. `expectedFileSHA256` is the
    /// `manifestFileSHA256` the review returned.
    public static func renameStudy(
        name: String, to newName: String, expectedFileSHA256: String, workspaceRoot: URL
    ) throws -> Review {
        let reviewed = try DraftAuthoringSnapshot.review(
            name: name, workspaceRoot: workspaceRoot, expectedFileSHA256: expectedFileSHA256)
        var review = try reviewStudyRename(reviewed, to: newName)
        let outcome = try ExperimentStore.rename(
            experimentName: name, to: newName, reviewed: reviewed)
        review.applied = true
        review.destination = "experiments/\(outcome.newName)"
        return review
    }

    /// Preview deleting a draft study.
    public static func reviewStudyDelete(name: String, workspaceRoot: URL) throws -> Review {
        try reviewStudyDelete(DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: name))
    }

    static func reviewStudyDelete(_ study: DraftAuthoringSnapshot) throws -> Review {
        let name = study.manifest.name
        try requireDraft(study.manifest, operation: .delete)
        let runs = ExperimentStore.runsStamped(experimentName: name)
        var effects = [
            "Moves experiments/\(name)/ to experiments/.trash-<time>/\(name)/. Nothing "
                + "is erased: move the folder back to restore the study."
        ]
        if runs > 0 {
            effects.append(
                "\(runs) \(runs == 1 ? "run recorded" : "runs recorded") this study's "
                    + "name. \(runs == 1 ? "It stays" : "They stay") in runs/ unchanged.")
        }
        effects.append("Nothing in runs/ is changed.")
        return Review(
            kind: .study, operation: .delete, applied: false, name: name, newName: nil,
            path: "experiments/\(name)", destination: "experiments/.trash-<time>/\(name)",
            manifestFileSHA256: study.file.sha256, designFileSHA256: nil,
            artifactFileSHA256: nil, status: study.manifest.status.rawValue,
            runsRecordingName: runs, studiesFromTemplate: nil, usedBy: nil,
            effects: effects, advisories: [])
    }

    /// The one implementation of deleting a draft: the app's Delete button and
    /// both command lines' `experiment delete` end here. Returns the folder
    /// the draft was moved to.
    @discardableResult
    public static func deleteStudy(reviewed: DraftAuthoringSnapshot) throws -> URL {
        try DraftAuthoringTransaction.perform(reviewed: reviewed) { name in
            try ExperimentStore.moveDraftToTrash(name: name)
        }
    }

    /// Delete a draft study the caller reviewed.
    public static func deleteStudy(
        name: String, expectedFileSHA256: String, workspaceRoot: URL
    ) throws -> Review {
        let reviewed = try DraftAuthoringSnapshot.review(
            name: name, workspaceRoot: workspaceRoot, expectedFileSHA256: expectedFileSHA256)
        var review = try reviewStudyDelete(reviewed)
        let moved = try deleteStudy(reviewed: reviewed)
        review.applied = true
        review.destination = try relativePath(moved, in: workspaceRoot)
        return review
    }

    /// Frozen and complete studies keep their name and their folder: their
    /// runs point at them. The repair is the one way to iterate on one.
    static func requireDraft(
        _ manifest: ExperimentManifest, operation: Review.Operation
    ) throws {
        guard manifest.status != .draft else { return }
        let name = manifest.name
        let status = manifest.status.rawValue
        let reason: String
        switch operation {
        case .rename:
            reason = "'\(name)' is \(status), so it keeps its name: its runs point at "
                + "it by name. Only a draft can be renamed."
        case .delete:
            reason = "'\(name)' is \(status), so it cannot be deleted: its runs point "
                + "at it. Only a draft can be deleted."
        }
        throw ExperimentError.refusing(
            .statusImmutable, reason,
            repair: "steerlab-cli experiment duplicate \(name) \(name)-v2  "
                + "(the copy is an editable draft)")
    }

    // MARK: - Templates

    /// Preview renaming a template.
    public static func reviewTemplateRename(
        name: String, to newName: String, workspaceRoot: URL
    ) throws -> Review {
        try reviewTemplateRename(StudyDesignSnapshot(workspaceRoot: workspaceRoot, name: name), to: newName)
    }

    static func reviewTemplateRename(_ design: StudyDesignSnapshot, to newName: String) throws -> Review {
        let name = design.template.name
        let target = ExperimentStore.resolvedRenameTarget(newName)
        guard target.contains(where: { $0.isLetter || $0.isNumber }) else {
            throw StudyDesignAuthoringError(
                code: "invalidDesignName",
                reason: "The new template name needs at least one letter or digit.",
                repairAction: "Choose a name of lowercase letters, digits, and hyphens.")
        }
        guard target != name else {
            throw StudyDesignAuthoringError(
                code: "invalidDesignName", reason: "The template is already called '\(name)'.",
                repairAction: "Choose a different name.")
        }
        let destination = design.workspaceRoot.appending(components: "templates", target)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw StudyDesignAuthoringError(
                code: "designNameTaken", reason: "A template named '\(target)' already exists.",
                repairAction: "Choose a name that is not already in the template list.")
        }
        let minted = studiesCreated(fromTemplate: name, workspaceRoot: design.workspaceRoot)
        var effects = [
            "Moves templates/\(name)/ to templates/\(target)/ and changes the "
                + "template's name inside it."
        ]
        if minted > 0 {
            effects.append(
                "\(minted) \(minted == 1 ? "study was" : "studies were") created from "
                    + "it. \(minted == 1 ? "It keeps" : "They keep") the old name in "
                    + "the record of where \(minted == 1 ? "it" : "they") came from, "
                    + "and \(minted == 1 ? "is" : "are") otherwise unchanged.")
        }
        return Review(
            kind: .template, operation: .rename, applied: false, name: name,
            newName: target, path: "templates/\(name)", destination: "templates/\(target)",
            manifestFileSHA256: nil, designFileSHA256: design.file.sha256,
            artifactFileSHA256: nil, status: nil, runsRecordingName: nil,
            studiesFromTemplate: minted, usedBy: nil, effects: effects, advisories: [])
    }

    /// Rename a template the caller reviewed.
    public static func renameTemplate(
        name: String, to newName: String, expectedFileSHA256: String, workspaceRoot: URL
    ) throws -> Review {
        let reviewed = try StudyDesignAuthoring.review(
            name: name, workspaceRoot: workspaceRoot, expectedFileSHA256: expectedFileSHA256)
        var review = try reviewTemplateRename(reviewed, to: newName)
        let resolved = try renameTemplate(reviewed: reviewed, to: newName)
        review.applied = true
        review.destination = "templates/\(resolved)"
        return review
    }

    /// The reviewed rename, under the design's lock. The move itself is the
    /// app's owner, `StudyTemplateStore.rename`.
    @discardableResult
    public static func renameTemplate(reviewed: StudyDesignSnapshot, to newName: String) throws -> String {
        try requireActiveWorkspace(reviewed.workspaceRoot)
        return try StudyDesignAuthoring.withReviewedDesign(reviewed) { _ in
            let target = reviewed.workspaceRoot.appending(
                components: "templates", ExperimentStore.resolvedRenameTarget(newName), "template.json")
            return try ManifestFileTransaction.withLock(
                manifestURL: target, workspaceRoot: reviewed.workspaceRoot
            ) {
                try StudyTemplateStore.rename(templateName: reviewed.template.name, to: newName)
            }
        }
    }

    /// Preview deleting a template.
    public static func reviewTemplateDelete(name: String, workspaceRoot: URL) throws -> Review {
        try reviewTemplateDelete(StudyDesignSnapshot(workspaceRoot: workspaceRoot, name: name))
    }

    static func reviewTemplateDelete(_ design: StudyDesignSnapshot) throws -> Review {
        let name = design.template.name
        let minted = studiesCreated(fromTemplate: name, workspaceRoot: design.workspaceRoot)
        var effects = [
            "Moves templates/\(name)/ to templates/.trash-<time>/\(name)/. Nothing is "
                + "erased: move the folder back to restore the template."
        ]
        if minted > 0 {
            effects.append(
                "\(minted) \(minted == 1 ? "study was" : "studies were") created from "
                    + "it. \(minted == 1 ? "It is an" : "They are") ordinary "
                    + "\(minted == 1 ? "study" : "studies") and \(minted == 1 ? "is" : "are") "
                    + "not changed.")
        }
        return Review(
            kind: .template, operation: .delete, applied: false, name: name, newName: nil,
            path: "templates/\(name)", destination: "templates/.trash-<time>/\(name)",
            manifestFileSHA256: nil, designFileSHA256: design.file.sha256,
            artifactFileSHA256: nil, status: nil, runsRecordingName: nil,
            studiesFromTemplate: minted, usedBy: nil, effects: effects, advisories: [])
    }

    /// Delete a template the caller reviewed.
    public static func deleteTemplate(
        name: String, expectedFileSHA256: String, workspaceRoot: URL
    ) throws -> Review {
        let reviewed = try StudyDesignAuthoring.review(
            name: name, workspaceRoot: workspaceRoot, expectedFileSHA256: expectedFileSHA256)
        var review = try reviewTemplateDelete(reviewed)
        let moved = try deleteTemplate(reviewed: reviewed)
        review.applied = true
        review.destination = try relativePath(moved, in: workspaceRoot)
        return review
    }

    /// The reviewed delete, under the design's lock. The move itself is the
    /// app's owner, `StudyTemplateStore.delete`.
    @discardableResult
    public static func deleteTemplate(reviewed: StudyDesignSnapshot) throws -> URL {
        try requireActiveWorkspace(reviewed.workspaceRoot)
        return try StudyDesignAuthoring.withReviewedDesign(reviewed) { _ in
            try StudyTemplateStore.delete(name: reviewed.template.name)
        }
    }

    static func studiesCreated(fromTemplate name: String, workspaceRoot: URL) -> Int {
        studyManifests(in: workspaceRoot).readable
            .filter { $0.manifest.templateProvenance?.template == name }.count
    }

    // MARK: - Agents

    /// Preview deleting an agent. Refused while any study uses it.
    public static func reviewAgentDelete(path: String, workspaceRoot: URL) throws -> Review {
        try reviewAgentDelete(AgentArtifactSnapshot(workspaceRoot: workspaceRoot, path: path))
    }

    static func reviewAgentDelete(_ agent: AgentArtifactSnapshot) throws -> Review {
        let folder = try ModelVariantStore.trashableFolder(
            for: agent.record, workspaceRoot: agent.workspaceRoot)
        let usage = try requireUnused(agent)
        let folderPath = try relativePath(folder, in: agent.workspaceRoot)
        let trashPath = try relativePath(folder.deletingLastPathComponent(), in: agent.workspaceRoot)
        let effects = [
            "Moves \(folderPath)/ to \(trashPath)/.trash-<time>/\(folder.lastPathComponent)/. "
                + "Nothing is erased: move the folder back to restore the agent.",
            "No study uses this agent, so no study changes.",
        ]
        return Review(
            kind: .agent, operation: .delete, applied: false,
            name: agent.record.artifact.name, newName: nil, path: agent.path,
            destination: "\(trashPath)/.trash-<time>/\(folder.lastPathComponent)",
            manifestFileSHA256: nil, designFileSHA256: nil,
            artifactFileSHA256: agent.file.sha256, status: nil, runsRecordingName: nil,
            studiesFromTemplate: nil, usedBy: [], effects: effects,
            advisories: usage.unreadable.map {
                "Study '\($0)' could not be read, so it was not checked for this agent."
            })
    }

    /// Delete an agent the caller reviewed.
    public static func deleteAgent(
        path: String, expectedFileSHA256: String, workspaceRoot: URL
    ) throws -> Review {
        let reviewed = try StudyAgentAuthoring.reviewArtifact(
            path: path, workspaceRoot: workspaceRoot, expectedFileSHA256: expectedFileSHA256)
        var review = try reviewAgentDelete(reviewed)
        let moved = try deleteAgent(reviewed: reviewed)
        review.applied = true
        review.destination = try relativePath(moved, in: workspaceRoot)
        return review
    }

    /// The one implementation of deleting an agent: the app's Delete button
    /// and both command lines' `agent delete` end here. Re-reads the file and
    /// re-checks every study immediately before the move.
    @discardableResult
    public static func deleteAgent(reviewed: AgentArtifactSnapshot) throws -> URL {
        try requireActiveWorkspace(reviewed.workspaceRoot)
        let current = try ManifestFileTransaction.snapshot(at: reviewed.record.url)
        guard current.sha256 == reviewed.file.sha256 else {
            throw ExperimentError.refusing(
                .artifactPin, "The agent changed after it was reviewed; nothing was deleted.",
                repair: "steerlab-cli agent delete \(reviewed.path)  (preview it again)")
        }
        _ = try requireUnused(reviewed)
        return try ModelVariantStore.delete(reviewed.record, workspaceRoot: reviewed.workspaceRoot)
    }

    /// The studies that use an agent, by name, and the studies that could not
    /// be read to check. A study uses an agent when one of its agent arms, its
    /// confirmation policy, or its panel's seats names the agent's file.
    public static func studiesUsingAgent(
        path: String, workspaceRoot: URL
    ) -> (users: [String], unreadable: [String]) {
        guard let target = try? ManifestFileTransaction.canonicalPath(
            workspaceRoot.appending(path: path))
        else { return ([], []) }
        func names(_ reference: String?) -> Bool {
            guard let reference, !reference.isEmpty,
                let resolved = try? ManifestFileTransaction.canonicalPath(
                    ExperimentStore.resolveProjectPath(reference, root: workspaceRoot))
            else { return false }
            return resolved == target
        }
        let studies = studyManifests(in: workspaceRoot)
        var users: [String] = []
        for study in studies.readable {
            let manifest = study.manifest
            var references = manifest.variantConditions.map(\.artifactPath)
            if let source = manifest.perturbationPolicy?.sourceAgent.artifactPath {
                references.append(source)
            }
            if let scenarioPath = manifest.multiAgentScenarioPath, !scenarioPath.isEmpty,
                let data = try? Data(
                    contentsOf: ExperimentStore.resolveProjectPath(scenarioPath, root: workspaceRoot)),
                let scenario = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
            {
                references.append(contentsOf: scenario.agents.compactMap(\.variantArtifactPath))
            }
            if references.contains(where: names) { users.append(study.name) }
        }
        return (users.sorted(), studies.unreadable)
    }

    @discardableResult
    static func requireUnused(_ agent: AgentArtifactSnapshot) throws -> (users: [String], unreadable: [String]) {
        let usage = studiesUsingAgent(path: agent.path, workspaceRoot: agent.workspaceRoot)
        guard usage.users.isEmpty else {
            let count = usage.users.count
            throw Refusal(
                code: agentInUseCode, path: agent.path, usedBy: usage.users,
                reason: "This agent is used by \(count == 1 ? "a study" : "\(count) studies"): "
                    + usage.users.joined(separator: ", ") + ". An agent a study uses "
                    + "cannot be deleted.",
                repairAction: "steerlab-cli experiment manifest \(usage.users[0])  (see "
                    + "which arm uses it). In a draft, attach a different agent with "
                    + "steerlab-cli experiment attach-agent, or remove the arm in the "
                    + "app's Studies section, then preview the delete again. Frozen and "
                    + "complete studies keep their agents.")
        }
        return usage
    }

    /// An agent delete that declined: the agent is in use (`agentInUse`,
    /// carrying the studies), or a run saved it (`agentIsRunEvidence`).
    public struct Refusal: Error, LocalizedError, Sendable, Equatable {
        public let code: String
        public let path: String
        public let usedBy: [String]
        public let reason: String
        public let repairAction: String
        public var errorDescription: String? { reason }

        public init(code: String, path: String, usedBy: [String] = [],
                    reason: String, repairAction: String) {
            self.code = code
            self.path = path
            self.usedBy = usedBy
            self.reason = reason
            self.repairAction = repairAction
        }
    }

    // MARK: - Shared

    /// Studies in a workspace: those that decode, and the names of those that
    /// do not. Trash folders and other hidden entries are skipped.
    static func studyManifests(
        in workspaceRoot: URL
    ) -> (readable: [(name: String, manifest: ExperimentManifest)], unreadable: [String]) {
        let storage = ExperimentRepository(workspaceRoot: workspaceRoot)
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: storage.directory.path)) ?? [])
            .filter { !$0.hasPrefix(".") }.sorted()
        var readable: [(String, ExperimentManifest)] = []
        var unreadable: [String] = []
        for name in names {
            let url = storage.manifestURL(name)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            if let manifest = try? storage.load(name: name) {
                readable.append((name, manifest))
            } else {
                unreadable.append(name)
            }
        }
        return (readable, unreadable)
    }

    /// Move `item` into a fresh `.trash-<time>` folder under `parent`, never
    /// replacing anything. Returns where it landed.
    static func moveToTrash(_ item: URL, under parent: URL) throws -> URL {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
        let trash = parent.appending(component: ".trash-\(stamp)")
        try FileManager.default.createDirectory(at: trash, withIntermediateDirectories: true)
        var destination = trash.appending(component: item.lastPathComponent)
        var counter = 2
        while FileManager.default.fileExists(atPath: destination.path) {
            destination = trash.appending(component: "\(item.lastPathComponent)-\(counter)")
            counter += 1
        }
        try FileManager.default.moveItem(at: item, to: destination)
        return destination
    }

    static func relativePath(_ url: URL, in workspaceRoot: URL) throws -> String {
        let root = try ManifestFileTransaction.canonicalPath(workspaceRoot)
        let path = try ManifestFileTransaction.canonicalPath(url)
        return path.hasPrefix(root + "/") ? String(path.dropFirst(root.count + 1)) : path
    }

    /// The store owners these operations reuse resolve the active workspace;
    /// a review from another workspace must not be applied through them.
    static func requireActiveWorkspace(_ root: URL) throws {
        guard try ManifestFileTransaction.canonicalPath(root)
            == ManifestFileTransaction.canonicalPath(ExperimentStore.workspaceRoot)
        else {
            throw ExperimentError.refusing(
                .staleManifest, "The active workspace changed after this was reviewed.",
                repair: "Return to the reviewed workspace and preview the change again.")
        }
    }
}
