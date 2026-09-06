import Foundation

public struct StudyDesignSaveResult: Sendable {
    public let snapshot: StudyDesignSnapshot
    public let created: Bool
    public let hashBefore: String?
    public let warnings: [String]
    public var changed: Bool { created || hashBefore.map { $0 != StudyTemplateStore.hash(snapshot.template) } == true }
}

/// Saves the invariant study settings through the existing stripping and panel
/// hoisting rules. Source and destination reviews are distinct authorities.
public enum StudyDesignSaving {
    public static func create(from source: StudyDesignSourceReview, name requestedName: String? = nil,
                              description: String? = nil) throws -> StudyDesignSaveResult {
        try source.withCurrentSource {
            let root = source.study.workspaceRoot
            let library = root.appending(component: "templates")
            try requireDirectories(["templates"], root: root)
            return try ManifestFileTransaction.withLock(manifestURL: library, workspaceRoot: root) {
                try requireDirectories(["templates"], root: root)
                let study = source.study.manifest
                let body = StudyTemplateStore.strippedBody(study)
                let warnings = source.panel?.warnings ?? []
                if let provenance = study.templateProvenance,
                    let existing = try? StudyDesignSnapshot(workspaceRoot: root, name: provenance.template) {
                    var candidate = existing.template
                    candidate.study = body
                    if samePanel(source.panel?.semantic, as: existing), StudyTemplateStore.hash(candidate) == StudyTemplateStore.hash(existing.template) {
                        return .init(snapshot: existing, created: false, hashBefore: StudyTemplateStore.hash(existing.template), warnings: warnings)
                    }
                }
                let base = ExperimentStore.canonicalSlug(requestedName ?? study.name)
                guard base.contains(where: { $0.isLetter || $0.isNumber }) else {
                    throw ExperimentError.malformed("A design name needs letters or digits.", repair: "Supply a usable name for the reusable design.")
                }
                var name = base
                var suffix = 1
                while try exists(library.appending(component: name)) {
                    suffix += 1
                    name = "\(base)-\(suffix)"
                }
                let directory = library.appending(component: name)
                let file = directory.appending(component: "template.json")
                return try ManifestFileTransaction.withLock(manifestURL: file, workspaceRoot: root) {
                    guard try !exists(directory) else {
                        throw ExperimentError.refusing(.staleManifest, "Another operation occupied the new design's directory.",
                            repair: "Inspect the design library and choose a fresh destination before saving again.")
                    }
                    let semantic = try source.panel.map {
                        try pinPanel($0.semantic, reusing: study.multiAgentSemanticScenarioPath, root: root)
                    }
                    let template = StudyTemplate(name: name, templateDescription: description ?? study.experimentDescription,
                        parentTemplate: study.templateProvenance?.template, semanticScenario: semantic, study: body)
                    try ManifestFileTransaction.requireCurrent(.absent, at: file)
                    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                    try encode(template).write(to: file, options: .atomic)
                    return .init(snapshot: try StudyDesignSnapshot(workspaceRoot: root, name: name),
                        created: true, hashBefore: nil, warnings: warnings)
                }
            }
        }
    }

    public static func update(from source: StudyDesignSourceReview, reviewed: StudyDesignSnapshot) throws -> StudyDesignSaveResult {
        guard try ManifestFileTransaction.canonicalPath(source.study.workspaceRoot) == ManifestFileTransaction.canonicalPath(reviewed.workspaceRoot) else {
            throw StudyDesignAuthoringError(code: "designWorkspaceChanged", reason: "The study and design reviews belong to different workspaces.",
                repairAction: "Inspect the source study and destination design in the same workspace.")
        }
        guard source.study.manifest.templateProvenance?.template == reviewed.template.name else {
            throw StudyDesignAuthoringError(code: "designLineageMismatch", reason: "The study was not created from this design.",
                repairAction: "Save it as a new design, or inspect the design named by the study's lineage.")
        }
        return try StudyDesignAuthoring.withReviewedDesign(reviewed) { file in
            try source.withCurrentSource {
                var updated = reviewed.template
                updated.study = StudyTemplateStore.strippedBody(source.study.manifest)
                var warnings = source.panel?.warnings ?? []
                if let panel = source.panel?.semantic {
                    if !samePanel(panel, as: reviewed) {
                        updated.semanticScenario = try pinPanel(panel,
                            reusing: source.study.manifest.multiAgentSemanticScenarioPath, root: reviewed.workspaceRoot)
                        warnings.append("This design now declares a different panel. Earlier studies retain their original inputs and can read as diverged from the current design.")
                    }
                } else if updated.semanticScenario != nil {
                    updated.semanticScenario = nil
                    warnings.append("The source study carries no panel, so the design no longer declares one. Seat castings require a panel to be saved back first.")
                }
                if updated != reviewed.template { try encode(updated).write(to: file, options: .atomic) }
                return .init(snapshot: try StudyDesignSnapshot(workspaceRoot: reviewed.workspaceRoot, name: updated.name),
                    created: false, hashBefore: StudyTemplateStore.hash(reviewed.template), warnings: warnings)
            }
        }
    }

    private static func samePanel(_ panel: MultiAgentScenario?, as reviewed: StudyDesignSnapshot) -> Bool {
        guard let panel else { return reviewed.template.semanticScenario == nil }
        guard let ref = reviewed.template.semanticScenario,
            let stored = try? StudyTemplateStore.loadSemanticPanel(ref, workspaceRoot: reviewed.workspaceRoot) else { return false }
        return stored == panel
    }

    /// Reuse only an equivalent local source. Otherwise publish immutable,
    /// content-addressed semantic bytes in the ordinary panel library.
    private static func pinPanel(_ panel: MultiAgentScenario, reusing recorded: String?, root: URL) throws -> StudyTemplate.SemanticScenarioRef {
        let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
        if let recorded, !recorded.isEmpty {
            let url = ExperimentStore.resolveProjectPath(recorded, root: root)
            if let canonical = try? ManifestFileTransaction.canonicalPath(url), canonical.hasPrefix(prefix),
                let data = try? Data(contentsOf: url), let stored = try? JSONDecoder().decode(MultiAgentScenario.self, from: data),
                stored == panel {
                return .init(path: String(canonical.dropFirst(prefix.count)), hash: MultiAgentScenarioStore.hash(data))
            }
        }
        try requireDirectories(["prompts", "panels"], root: root)
        let data = try encode(panel)
        let hash = MultiAgentScenarioStore.hash(data)
        let path = "prompts/panels/semantic-\(hash).json"
        let url = root.appending(path: path)
        return try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
            try requireDirectories(["prompts", "panels"], root: root)
            if try exists(url) {
                guard try FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType == .typeRegular,
                    try Data(contentsOf: url) == data else {
                    throw ExperimentError.refusing(.artifactPin, "The semantic panel destination does not contain its named bytes.",
                        repair: "Inspect the panel library and restore its verified content; do not overwrite a pinned input.")
                }
            } else {
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
            }
            return .init(path: path, hash: hash)
        }
    }

    private static func encode(_ value: some Encodable) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(value)
    }

    private static func exists(_ url: URL) throws -> Bool {
        do { _ = try FileManager.default.attributesOfItem(atPath: url.path); return true }
        catch CocoaError.fileReadNoSuchFile { return false }
    }

    private static func requireDirectories(_ components: [String], root: URL) throws {
        var url = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(root))
        for component in components {
            url.append(component: component)
            do {
                guard try FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType == .typeDirectory else {
                    throw StudyDesignAuthoringError(code: "unsafeDesignPath", reason: "Design publication requires ordinary workspace directories.",
                        repairAction: "Inspect the design and panel libraries; do not redirect authoring into immutable evidence through links.")
                }
            } catch CocoaError.fileReadNoSuchFile { return }
        }
    }
}
