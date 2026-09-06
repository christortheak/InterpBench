import Foundation

/// The same saved/reused design result for the CLI and workbench API.
struct StudyDesignSavingDocument: Encodable {
    let ok = true
    let sourceStudy: String
    let sourceManifestFileSHA256: String
    let created: Bool
    let changed: Bool
    let hashBefore: String?
    let design: StudyDesignDocument
    let warnings: [String]

    init(_ result: StudyDesignSaveResult, source: StudyDesignSourceReview) throws {
        sourceStudy = source.study.manifest.name
        sourceManifestFileSHA256 = source.study.file.sha256
        created = result.created
        changed = result.changed
        hashBefore = result.hashBefore
        design = try StudyDesignDocument(result.snapshot)
        warnings = result.warnings
    }
}
