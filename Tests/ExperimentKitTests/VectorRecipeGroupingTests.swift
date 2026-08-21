import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the variant editor's recipe-grouped vector picker.
/// The fixture mirrors the real workspace's shape (field report 2026-08-06:
/// 198 artifacts, ~30 concepts): the same recipe re-derived many times by
/// repeated extract/validate/sweep, one concept whose stimulus hash changed
/// mid-history, one whose method was upgraded, and concepts with a single
/// recipe. No filesystem, no networking — synthetic records only.
struct VectorRecipeGroupingTests {

    /// A synthetic record shaped like both real ones (`VectorArtifact
    /// .sidecar` and `RemoteVectorRecord`) without depending on either.
    private struct Fake: Sendable {
        var id: String
        var name: String
        var concept: String?
        var model: String = "google/gemma-3-27b-it"
        var recipeMethod: String?
        var extractionMethod: String?
        var hash: String?
        var date: String?
    }

    private func present(_ items: [Fake]) -> VectorRecipeGrouping.Presentation<Fake> {
        VectorRecipeGrouping.present(
            items,
            concept: { $0.concept },
            modelID: { $0.model },
            recipeMethod: { $0.recipeMethod },
            extractionMethod: { $0.extractionMethod },
            stimulusSetHash: { $0.hash },
            extractionDate: { $0.date },
            name: { $0.name },
            id: { $0.id })
    }

    /// Five deterministic re-derivations of one recipe (what an afternoon of
    /// extract/validate/sweep actually produces).
    private func rederivations(
        concept: String, method: String, hash: String, days: [String]
    ) -> [Fake] {
        days.map { day in
            Fake(
                id: "runs/\(day)-\(concept)/\(concept)",
                name: "\(concept)-\(day)",
                concept: concept,
                recipeMethod: method,
                hash: hash,
                date: "\(day)T09:00:00Z")
        }
    }

    /// The whole fixture: 12 artifacts, 4 concepts, 6 distinct recipes.
    private var catalog: [Fake] {
        var items: [Fake] = []
        // sympathy: ONE recipe, re-derived 5×.
        items += rederivations(
            concept: "sympathy", method: "emotionGrandMean", hash: "aaaa1111",
            days: ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05"])
        // fear: stimulus-hash change mid-history (9a30d4db → 978b84c7),
        // each side re-derived twice.
        items += rederivations(
            concept: "fear", method: "emotionGrandMean", hash: "9a30d4db",
            days: ["2026-06-10", "2026-06-11"])
        items += rederivations(
            concept: "fear", method: "emotionGrandMean", hash: "978b84c7",
            days: ["2026-07-20", "2026-07-21"])
        // authority: method upgrade emotionGrandMean → designatedReference
        // on the SAME stimulus bytes.
        items += rederivations(
            concept: "authority", method: "emotionGrandMean", hash: "cccc3333",
            days: ["2026-05-02"])
        items += rederivations(
            concept: "authority", method: "designatedReference", hash: "cccc3333",
            days: ["2026-08-01"])
        // deference: a single recipe, single derivation.
        items += rederivations(
            concept: "deference", method: "caaMeanDifference", hash: "dddd4444",
            days: ["2026-07-15"])
        return items
    }

    // MARK: Collapse

    @Test("Same-recipe re-derivations collapse to the newest; older are hidden")
    func newestWinsWithinARecipe() throws {
        let result = present(catalog)
        #expect(catalog.count == 12)
        // 6 distinct recipes across 4 concepts.
        #expect(result.offers.count == 6)
        #expect(result.sections.count == 4)
        // 12 artifacts − 6 offered = 6 hidden re-derivations.
        #expect(result.hiddenDerivationCount == 6)

        let sympathy = try #require(result.sections.first { $0.concept == "sympathy" })
        #expect(sympathy.offers.count == 1)
        let offer = try #require(sympathy.offers.first)
        #expect(offer.item.name == "sympathy-2026-07-05")
        #expect(offer.hiddenDerivationCount == 4)
        #expect(offer.isSuperseded == false)
    }

    @Test("A single-recipe concept hides nothing and is never superseded")
    func singleRecipeConcept() throws {
        let result = present(catalog)
        let deference = try #require(result.sections.first { $0.concept == "deference" })
        #expect(deference.offers.count == 1)
        let offer = try #require(deference.offers.first)
        #expect(offer.hiddenDerivationCount == 0)
        #expect(offer.isSuperseded == false)
        #expect(offer.displayLabel == "deference · CAA · 2026-07-15")
    }

    // MARK: Distinct recipes stay visible

    @Test("A stimulus-hash change makes a DISTINCT recipe; both stay visible")
    func stimulusHashChangeStaysVisible() throws {
        let result = present(catalog)
        let fear = try #require(result.sections.first { $0.concept == "fear" })
        #expect(fear.offers.count == 2)
        // Newest recipe first.
        #expect(fear.offers[0].stimulusSetHash == "978b84c7")
        #expect(fear.offers[0].isSuperseded == false)
        #expect(fear.offers[0].item.name == "fear-2026-07-21")
        // The older stimulus set is NOT hidden — it is a different
        // experimental object — but it is marked.
        #expect(fear.offers[1].stimulusSetHash == "9a30d4db")
        #expect(fear.offers[1].isSuperseded == true)
        #expect(fear.offers[1].displayLabel.hasSuffix("· superseded"))
        // One older re-derivation hidden on each side.
        #expect(fear.offers[0].hiddenDerivationCount == 1)
        #expect(fear.offers[1].hiddenDerivationCount == 1)
    }

    @Test("A method upgrade on identical stimuli is a distinct recipe too")
    func methodUpgradeStaysVisible() throws {
        let result = present(catalog)
        let authority = try #require(result.sections.first { $0.concept == "authority" })
        #expect(authority.offers.count == 2)
        #expect(authority.offers[0].method == "designatedReference")
        #expect(authority.offers[0].isSuperseded == false)
        #expect(authority.offers[1].method == "emotionGrandMean")
        #expect(authority.offers[1].isSuperseded == true)
        // Same stimulus bytes on both — only the method distinguishes them.
        #expect(authority.offers[0].stimulusSetHash == authority.offers[1].stimulusSetHash)
    }

    // MARK: Ordering and labels

    @Test("Concepts are alphabetical; the flat list follows section order")
    func ordering() {
        let result = present(catalog)
        #expect(result.sections.map(\.concept) == ["authority", "deference", "fear", "sympathy"])
        #expect(result.offers.map(\.concept)
            == ["authority", "authority", "deference", "fear", "fear", "sympathy"])
    }

    @Test("Labels read concept · method · date, with the superseded marker")
    func labels() throws {
        let result = present(catalog)
        let authority = try #require(result.sections.first { $0.concept == "authority" })
        #expect(authority.offers[0].label == "authority · designated-reference · 2026-08-01")
        #expect(authority.offers[0].displayLabel == authority.offers[0].label)
        #expect(authority.offers[1].displayLabel
            == "authority · grand-mean · 2026-05-02 · superseded")
    }

    @Test("Two distinct recipes reading identically are disambiguated by name")
    func labelCollisionDisambiguation() {
        // Same concept, same method, same day — different stimulus hashes.
        let items = [
            Fake(
                id: "runs/a/x", name: "alpha", concept: "risk",
                recipeMethod: "emotionGrandMean", hash: "1111",
                date: "2026-07-01T09:00:00Z"),
            Fake(
                id: "runs/b/x", name: "beta", concept: "risk",
                recipeMethod: "emotionGrandMean", hash: "2222",
                date: "2026-07-01T09:00:00Z"),
        ]
        let result = present(items)
        #expect(result.offers.count == 2)
        let labels = Set(result.offers.map(\.label))
        #expect(labels.count == 2)
        #expect(labels.contains("risk · grand-mean · 2026-07-01 · alpha"))
        #expect(labels.contains("risk · grand-mean · 2026-07-01 · beta"))
    }

    // MARK: Identity and stability

    @Test("Offered ids are the caller's ids, unchanged")
    func idsAreStable() {
        let result = present(catalog)
        #expect(result.offeredIDs.count == 6)
        for offer in result.offers {
            #expect(offer.id == offer.item.id)
            #expect(result.offeredIDs.contains(offer.id))
        }
        // Every offered id is a real catalog id — nothing synthesized.
        let catalogIDs = Set(catalog.map(\.id))
        #expect(result.offeredIDs.isSubset(of: catalogIDs))
    }

    @Test("A ref pointing at a hidden older derivation is NOT offered")
    func storedHiddenRefFallsThrough() {
        let result = present(catalog)
        // The 2026-07-01 sympathy derivation is hidden behind 2026-07-05.
        let hidden = "runs/2026-07-01-sympathy/sympathy"
        #expect(catalog.contains { $0.id == hidden })
        #expect(!result.offeredIDs.contains(hidden))
        // …which is exactly the condition the view's non-pickable
        // current-selection row keys off.
        #expect(WorkspaceScoping.selectionOutsideInventory(
            hidden, inventory: Array(result.offeredIDs)))
    }

    @Test("Input order does not change the result")
    func orderIndependence() {
        let forward = present(catalog)
        let reversed = present(catalog.reversed())
        #expect(forward.offers.map(\.id) == reversed.offers.map(\.id))
        #expect(forward.hiddenDerivationCount == reversed.hiddenDerivationCount)
    }

    // MARK: Degenerate inputs

    @Test("Method identity is recipeMethod ?? extractionMethod")
    func methodNormalization() {
        #expect(VectorRecipeGrouping.normalizedMethod(
            recipeMethod: "caaMeanDifference", extractionMethod: "meanDifference")
            == "caaMeanDifference")
        #expect(VectorRecipeGrouping.normalizedMethod(
            recipeMethod: nil, extractionMethod: "lat") == "lat")
        #expect(VectorRecipeGrouping.normalizedMethod(
            recipeMethod: "  ", extractionMethod: "lat") == "lat")
        #expect(VectorRecipeGrouping.normalizedMethod(
            recipeMethod: nil, extractionMethod: nil) == "")
        // An unheard-of method travels verbatim rather than collapsing.
        #expect(VectorRecipeGrouping.methodLabel("someFutureRecipe") == "someFutureRecipe")
    }

    @Test("Different base models never hide each other")
    func modelIsPartOfTheRecipe() {
        let items = [
            Fake(
                id: "a", name: "a", concept: "fear", model: "gemma",
                recipeMethod: "emotionGrandMean", hash: "1111",
                date: "2026-07-01T09:00:00Z"),
            Fake(
                id: "b", name: "b", concept: "fear", model: "qwen",
                recipeMethod: "emotionGrandMean", hash: "1111",
                date: "2026-06-01T09:00:00Z"),
        ]
        let result = present(items)
        #expect(result.offers.count == 2)
        #expect(result.hiddenDerivationCount == 0)
    }

    @Test("Undated and concept-less artifacts degrade instead of merging")
    func missingFields() throws {
        let items = [
            Fake(id: "a", name: "orphan", concept: nil, recipeMethod: nil, hash: nil, date: nil),
            Fake(
                id: "b", name: "dated", concept: "fear",
                recipeMethod: "emotionGrandMean", hash: "1111", date: nil),
            Fake(
                id: "c", name: "later", concept: "fear",
                recipeMethod: "emotionGrandMean", hash: "1111",
                date: "2026-07-01T09:00:00Z"),
        ]
        let result = present(items)
        // The concept-less artifact files under its own name.
        #expect(result.sections.map(\.concept) == ["fear", "orphan"])
        let orphan = try #require(result.sections.first { $0.concept == "orphan" })
        #expect(orphan.offers[0].displayLabel == "orphan · method unrecorded · undated")
        // A dated derivation outranks an undated one of the same recipe.
        let fear = try #require(result.sections.first { $0.concept == "fear" })
        #expect(fear.offers.count == 1)
        #expect(fear.offers[0].item.name == "later")
        #expect(fear.offers[0].hiddenDerivationCount == 1)
    }

    @Test("An empty catalog presents empty, with no caption")
    func emptyCatalog() {
        let result = present([])
        #expect(result.isEmpty)
        #expect(result.sections.isEmpty)
        #expect(result.hiddenDerivationCount == 0)
        #expect(VectorRecipeGrouping.hiddenDerivationsCaption(0) == nil)
    }

    @Test("The hidden-count caption names the reason, and counts singular")
    func caption() throws {
        let one = try #require(VectorRecipeGrouping.hiddenDerivationsCaption(1))
        #expect(one.hasPrefix("1 older re-derivation hidden"))
        let many = try #require(VectorRecipeGrouping.hiddenDerivationsCaption(7))
        #expect(many.hasPrefix("7 older re-derivations hidden"))
        #expect(many.contains("stimulus hash"))
    }
}
