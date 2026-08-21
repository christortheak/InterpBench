import Foundation

/// Ordering for the playground's vector pickers (field report 2026-08-04:
/// "an incredibly long, poorly sorted list"). One rule for both the local
/// and server catalogs: concepts A→Z as menu SECTIONS, and within a
/// concept the newest extraction first (a researcher re-extracting all
/// afternoon wants the latest at the top), name as the deterministic
/// tie-break.
public enum VectorPickerOrdering {

    public static func grouped<T>(
        _ items: [T],
        concept: (T) -> String?,
        name: (T) -> String,
        extractionDate: (T) -> String?
    ) -> [(concept: String, items: [T])] {
        let byConcept = Dictionary(grouping: items) {
            concept($0)?.isEmpty == false ? concept($0)! : name($0)
        }
        return byConcept.keys.sorted().map { key in
            let sorted = byConcept[key]!.sorted { left, right in
                // ISO-8601 timestamps compare lexicographically; absent
                // dates sink to the bottom of their concept.
                let leftDate = extractionDate(left) ?? ""
                let rightDate = extractionDate(right) ?? ""
                if leftDate != rightDate { return leftDate > rightDate }
                return name(left) < name(right)
            }
            return (concept: key, items: sorted)
        }
    }
}
