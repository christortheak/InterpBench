import Foundation

/// Public batch input uses the same casting document as single instantiation.
/// Shape errors refuse before publication; each casting's admission is reported
/// independently, using one retained design review for the entire batch.
public struct StudyDesignBatchInput: Sendable {
    private struct Row: Sendable {
        let casting: Data
        let name: String?
    }
    private let rows: [Row]

    public init(_ data: Data) throws {
        guard let fields = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            Set(fields.keys) == ["rows"], let values = fields["rows"] as? [[String: Any]], !values.isEmpty else {
            throw Self.malformed()
        }
        rows = try values.map { value in
            guard Set(value.keys).isSubset(of: ["casting", "studyName"]),
                let casting = value["casting"] as? [String: Any],
                value["studyName"] == nil || (value["studyName"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
                throw Self.malformed()
            }
            return Row(casting: try JSONSerialization.data(withJSONObject: casting), name: value["studyName"] as? String)
        }
    }

    public func mint(reviewed: StudyDesignSnapshot) throws -> StudyTemplateStore.BatchMint {
        // A stale top-level request is refused before any row is attempted.
        try StudyDesignAuthoring.withReviewedDesign(reviewed) { _ in }
        let castings = rows.map { row in Result { try StudyDesignCastingInput.resolve(row.casting, reviewed: reviewed) } }
        return StudyDesignInstantiation.mintBatch(reviewed: reviewed, reviewedCastings: castings, names: rows.map(\.name))
    }

    private static func malformed() -> ExperimentError {
        .malformed("A batch contains a nonempty rows array. Each row contains casting and an optional nonempty studyName; no other fields are accepted.",
            repair: "Supply {\"rows\":[{\"casting\":{\"agents\":[]},\"studyName\":\"baseline\"}]}. Inspect the design and agent digests first.")
    }
}

/// Shared wire result. A partial batch is not rolled back or silently retried.
struct StudyDesignBatchDocument: Encodable {
    let ok: Bool
    let changed: Bool
    let workspaceRoot: String
    let design: String
    let designFileSHA256: String
    let batchGroup: String
    let results: [StudyTemplateStore.RowMint]
    let minted: [String]
    let repairAction: String?

    init(_ batch: StudyTemplateStore.BatchMint, reviewed: StudyDesignSnapshot) {
        ok = batch.failures.isEmpty
        changed = !batch.minted.isEmpty
        workspaceRoot = reviewed.workspaceRoot.path
        design = reviewed.template.name
        designFileSHA256 = reviewed.file.sha256
        batchGroup = batch.batchGroup
        results = batch.results
        minted = batch.minted
        repairAction = ok ? nil : "Keep the successful studies. Review each failed row's issue, then submit only repaired failed rows in a new batch; repeating the entire batch creates additional studies."
    }
}
