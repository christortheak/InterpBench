import Foundation

/// Recipe-grouped presentation for the variant editor's vector pickers
/// (field report 2026-08-06: "198 artifacts, ~30 concepts — every
/// extract/validate/sweep re-derives the SAME recipe, so the list is mostly
/// equivalent copies").
///
/// The rule this file implements, and WHY each half of it is what it is:
///
/// - A DERIVATION is one artifact. A RECIPE is `(model, normalized method,
///   stimulus-set hash)`. Two derivations of the same recipe are
///   deterministic re-derivations of each other — same pinned stimulus bytes,
///   same method, same model — so offering both is offering the same choice
///   twice. Only the NEWEST derivation of a recipe is offered; the older ones
///   are HIDDEN outright (no expandable "older copies" list — the researcher
///   confirmed there is nothing to choose between them).
/// - A DISTINCT recipe is a different EXPERIMENTAL object: a stimulus-hash
///   change or a method change makes a vector that is not interchangeable
///   with its predecessor. Those stay VISIBLE, all of them, under their
///   concept — and every recipe but the newest is marked `superseded` so the
///   history reads honestly instead of vanishing.
///
/// Method identity is the normalized `recipeMethod ?? extractionMethod` —
/// the same normalization the freshness index and the server catalog's
/// `resolvedMethod` use. Unknown method strings travel VERBATIM (a
/// `designatedReference` sidecar must never collapse into an
/// `emotionGrandMean` one just because this file has not heard of it).
///
/// Pure and generic over the record type, so the LOCAL (`VectorArtifact`)
/// and SERVER (`RemoteVectorRecord`) pickers route through one tested rule
/// and can never disagree. No filesystem, no networking: callers pass in
/// already-filtered records.
public enum VectorRecipeGrouping {

    // MARK: - Output shapes

    /// One offered artifact: the newest derivation of one distinct recipe.
    public struct Offer<Item: Sendable>: Identifiable, Sendable {
        /// The record itself — the caller still owns id/localization
        /// semantics (a server row's `workspaceRelativeID` is not this
        /// file's business).
        public let item: Item
        /// The id the picker TAGS with — supplied by the caller exactly as
        /// it stores it today. This file introduces no id scheme of its own.
        public let id: String
        public let concept: String
        /// Normalized method token (empty when the artifact records none).
        public let method: String
        /// Pinned stimulus-set hash (empty when the record does not carry
        /// one — an older server's catalog row).
        public let stimulusSetHash: String
        /// Extraction timestamp verbatim (empty when unrecorded).
        public let extractionDate: String
        /// "concept · method · date", disambiguated by artifact name only
        /// when two recipes of one concept would otherwise read identically.
        public let label: String
        /// What the picker renders: `label`, plus the superseded marker.
        public let displayLabel: String
        /// This concept has a NEWER distinct recipe — the vector is still a
        /// legitimate choice, just not the current one.
        public let isSuperseded: Bool
        /// How many older re-derivations of THIS recipe are hidden behind
        /// this row.
        public let hiddenDerivationCount: Int
    }

    /// One concept's menu section.
    public struct ConceptSection<Item: Sendable>: Identifiable, Sendable {
        public let concept: String
        /// Newest recipe first.
        public let offers: [Offer<Item>]
        public var id: String { concept }
    }

    /// Everything a picker needs: sections to render, the flat render-order
    /// list (for defaults and counting), and the offered-id set that decides
    /// whether a stored ref needs the non-pickable current-selection row.
    public struct Presentation<Item: Sendable>: Sendable {
        /// Concepts A→Z.
        public let sections: [ConceptSection<Item>]
        /// The same offers, flattened in render order.
        public let offers: [Offer<Item>]
        /// Older same-recipe derivations hidden across the whole picker.
        public let hiddenDerivationCount: Int
        /// Ids of the offered artifacts, in the caller's own id vocabulary.
        public let offeredIDs: Set<String>

        public var isEmpty: Bool { offers.isEmpty }
    }

    // MARK: - The rule

    /// Group `items` into the recipe-collapsed picker model.
    ///
    /// Accessors mirror the two record shapes without this file importing
    /// either: `recipeMethod`/`extractionMethod` are normalized here (first
    /// non-empty wins), and every optional missing on a record degrades to
    /// "unrecorded" rather than to a wrong merge — an artifact with no
    /// stimulus hash groups under the empty hash, which only ever merges
    /// with other hash-less artifacts of the same concept and method.
    public static func present<Item: Sendable>(
        _ items: [Item],
        concept: (Item) -> String?,
        modelID: (Item) -> String?,
        recipeMethod: (Item) -> String?,
        extractionMethod: (Item) -> String?,
        stimulusSetHash: (Item) -> String?,
        extractionDate: (Item) -> String?,
        name: (Item) -> String,
        id: (Item) -> String
    ) -> Presentation<Item> {

        let rows: [Row<Item>] = items.map { item in
            let itemName = name(item)
            return Row(
                item: item,
                id: id(item),
                name: itemName,
                // Mirrors `VectorPickerOrdering`: a concept-less artifact
                // files under its own name rather than under a blank
                // heading.
                concept: trimmed(concept(item)).isEmpty
                    ? itemName : trimmed(concept(item)),
                model: trimmed(modelID(item)),
                method: normalizedMethod(
                    recipeMethod: recipeMethod(item),
                    extractionMethod: extractionMethod(item)),
                hash: trimmed(stimulusSetHash(item)),
                date: trimmed(extractionDate(item)))
        }

        var sections: [ConceptSection<Item>] = []
        var flat: [Offer<Item>] = []
        var offeredIDs: Set<String> = []
        var hiddenTotal = 0

        let byConcept = Dictionary(grouping: rows, by: \.concept)
        for conceptKey in byConcept.keys.sorted() {
            let conceptRows = byConcept[conceptKey] ?? []

            // Distinct recipe = (model, method, stimulus hash). Model is in
            // the key so two models' artifacts can never hide each other,
            // even though today's callers pre-filter to one base model.
            let byRecipe = Dictionary(grouping: conceptRows) {
                RecipeKey(model: $0.model, method: $0.method, hash: $0.hash)
            }

            // Within a recipe: newest derivation wins. ISO-8601 timestamps
            // compare lexicographically; an undated artifact sinks below
            // every dated one, name then id as the deterministic tie-break
            // (the picker must not reshuffle between launches).
            var representatives: [(key: RecipeKey, row: Row<Item>, hidden: Int)] = []
            for key in byRecipe.keys {
                let derivations = (byRecipe[key] ?? []).sorted { left, right in
                    if left.date != right.date { return left.date > right.date }
                    if left.name != right.name { return left.name < right.name }
                    return left.id < right.id
                }
                guard let newest = derivations.first else { continue }
                representatives.append(
                    (key: key, row: newest, hidden: derivations.count - 1))
            }

            // Recipes within a concept: newest first, then a stable key
            // ordering so equal-dated recipes keep a fixed position.
            representatives.sort { left, right in
                if left.row.date != right.row.date { return left.row.date > right.row.date }
                if left.key.method != right.key.method {
                    return left.key.method < right.key.method
                }
                if left.key.hash != right.key.hash { return left.key.hash < right.key.hash }
                return left.key.model < right.key.model
            }

            // Base labels first, so a collision inside this concept can be
            // resolved before anything is rendered.
            let baseLabels = representatives.map {
                label(concept: conceptKey, method: $0.key.method, date: $0.row.date)
            }
            var labelCounts: [String: Int] = [:]
            for label in baseLabels { labelCounts[label, default: 0] += 1 }

            var offers: [Offer<Item>] = []
            for (index, entry) in representatives.enumerated() {
                let base = baseLabels[index]
                // Two DISTINCT recipes reading identically (a method upgrade
                // and its predecessor extracted the same day) would be an
                // unpickable menu; the artifact name breaks the tie.
                let resolved = (labelCounts[base] ?? 0) > 1
                    ? "\(base) · \(entry.row.name)" : base
                let superseded = index > 0
                offers.append(
                    Offer(
                        item: entry.row.item,
                        id: entry.row.id,
                        concept: conceptKey,
                        method: entry.key.method,
                        stimulusSetHash: entry.key.hash,
                        extractionDate: entry.row.date,
                        label: resolved,
                        displayLabel: superseded ? "\(resolved) · superseded" : resolved,
                        isSuperseded: superseded,
                        hiddenDerivationCount: entry.hidden))
                hiddenTotal += entry.hidden
                offeredIDs.insert(entry.row.id)
            }

            sections.append(ConceptSection(concept: conceptKey, offers: offers))
            flat.append(contentsOf: offers)
        }

        return Presentation(
            sections: sections, offers: flat,
            hiddenDerivationCount: hiddenTotal, offeredIDs: offeredIDs)
    }

    // MARK: - Captions

    /// The one line that tells a researcher WHY the picker is shorter than
    /// the catalog. Nil when nothing was hidden — no noise in the common
    /// single-derivation case.
    public static func hiddenDerivationsCaption(_ count: Int) -> String? {
        guard count > 0 else { return nil }
        return "\(count) older re-derivation\(count == 1 ? "" : "s") hidden — "
            + "same model, method and stimulus hash as the row above it, so "
            + "they derive identically"
    }

    /// Display spelling for a normalized method token. Known values get the
    /// house wording; anything else travels verbatim, because a method this
    /// file has not heard of is still a real distinction.
    public static func methodLabel(_ method: String) -> String {
        switch method {
        case "caaMeanDifference", "meanDifference": "CAA"
        case "repeLAT", "lat": "LAT"
        case "repeReaderLAT": "RepE reader LAT"
        case "emotionGrandMean": "grand-mean"
        case "designatedReference": "designated-reference"
        case "jlensTokenDirection": "J-lens token"
        case "": "method unrecorded"
        default: method
        }
    }

    /// `recipeMethod ?? extractionMethod`, normalized: the first non-empty
    /// of the two, trimmed. Empty means the artifact records neither.
    public static func normalizedMethod(
        recipeMethod: String?, extractionMethod: String?
    ) -> String {
        let recipe = trimmed(recipeMethod)
        return recipe.isEmpty ? trimmed(extractionMethod) : recipe
    }

    // MARK: - Internals

    private struct RecipeKey: Hashable {
        var model: String
        var method: String
        var hash: String
    }

    /// One artifact with every grouping field already normalized.
    private struct Row<Item> {
        var item: Item
        var id: String
        var name: String
        var concept: String
        var model: String
        var method: String
        var hash: String
        var date: String
    }

    private static func trimmed(_ value: String?) -> String {
        value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private static func label(concept: String, method: String, date: String) -> String {
        let day = date.isEmpty ? "undated" : String(date.prefix(10))
        return "\(concept) · \(methodLabel(method)) · \(day)"
    }
}
