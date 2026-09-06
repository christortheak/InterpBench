import Foundation

/// Pipeline declaration authorship uses the document the editor reviewed.
/// Validation is shared by all callers, including callers without a UI.
public enum StudyPipelineAuthoring {
    @discardableResult
    public static func saveBlock(_ block: JSONValue?, reviewed: DraftAuthoringSnapshot) throws -> DraftAuthoringSnapshot {
        let violations = ExperimentStore.pipelineBlockViolations(block)
        guard violations.isEmpty else {
            throw ExperimentError.malformed("Invalid pipeline declaration: " + violations.joined(separator: "; "),
                repair: "Correct the declared stages and gates before saving the reviewed draft.")
        }
        var manifest = reviewed.manifest
        manifest.pipeline = block
        return try DraftAuthoringTransaction.replace(manifest, reviewed: reviewed)
    }

    @discardableResult
    public static func save(_ draft: PipelineDraft?, reviewed: DraftAuthoringSnapshot) throws -> DraftAuthoringSnapshot {
        let block = draft?.encoded()
        let violations = (draft?.completenessViolations ?? [])
            + ExperimentStore.pipelineBlockViolations(block)
        guard violations.isEmpty else {
            throw ExperimentError.malformed("Invalid pipeline declaration: " + violations.joined(separator: "; "),
                repair: "Correct the stages and complete each declared gate before saving the reviewed draft.")
        }
        var manifest = reviewed.manifest
        manifest.pipeline = block
        return try DraftAuthoringTransaction.replace(manifest, reviewed: reviewed)
    }
}
