import Foundation
import Testing

@testable import ExperimentKit

/// Data-safety tests for the Model Variants tab's injections editor logic:
/// a definition's injection refs must survive an open → save round trip even
/// when the ACTIVE workspace cannot resolve them (a server-composed
/// definition opened in a Local workspace, a deleted vector, another
/// server's paths). Silent dropping of refs is the failure mode under test.
@Suite struct VariantInjectionEditingTests {

    /// A server-composed definition: refs are server-side artifact paths
    /// that no local catalog will ever resolve.
    private let foreignRefs = [
        ModelVariantArtifact.InjectionRef(
            concept: "independent",
            vectorArtifactID: "/srv/runs/2026-07-06T000000Z-concept-independent/independent",
            layer: 17,
            alpha: 3.25),
        ModelVariantArtifact.InjectionRef(
            concept: "fear",
            vectorArtifactID: "/srv/runs/2026-07-01T000000Z-concept-fear/fear",
            layer: 12,
            alpha: -0.5),
    ]

    private func encode(_ refs: [ModelVariantArtifact.InjectionRef]) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(refs)
    }

    @Test func foreignRefsSurviveOpenSaveRoundTripBytePreserved() throws {
        // Open in a workspace whose catalog resolves NOTHING (e.g. Local
        // workspace looking at server paths), then save without touching a
        // thing: the emitted injections array must be byte-identical.
        let drafts = VariantInjectionEditing.drafts(from: foreignRefs)
        let saved = VariantInjectionEditing.injectionRefs(
            drafts: drafts,
            resolveConcept: { _ in nil })
        #expect(saved == foreignRefs)
        #expect(try encode(saved) == encode(foreignRefs))
    }

    @Test func mixedResolvableAndForeignRefsAllSurvive() throws {
        // A definition mixing refs this workspace knows with refs it does
        // not: the known one refreshes its concept from the catalog, the
        // foreign one round-trips verbatim — neither is dropped.
        let localRef = ModelVariantArtifact.InjectionRef(
            concept: "authority",
            vectorArtifactID: "/Users/x/runs/2026-06-20T000000Z-concept-authority/authority",
            layer: 20,
            alpha: 1.5)
        let refs = [localRef] + foreignRefs
        let catalog = [localRef.vectorArtifactID: "authority"]
        let saved = VariantInjectionEditing.injectionRefs(
            drafts: VariantInjectionEditing.drafts(from: refs),
            resolveConcept: { catalog[$0] })
        #expect(saved == refs)
        #expect(try encode(saved) == encode(refs))
    }

    @Test func resolvableRefTakesTheCatalogsConcept() {
        // A ref the workspace CAN resolve uses the catalog's concept (the
        // repick-freshness rule); the stored concept is only the fallback.
        let drafts = VariantInjectionEditing.drafts(from: [
            ModelVariantArtifact.InjectionRef(
                concept: "stale-name", vectorArtifactID: "v1", layer: 3, alpha: 1)
        ])
        let saved = VariantInjectionEditing.injectionRefs(
            drafts: drafts,
            resolveConcept: { $0 == "v1" ? "fresh-name" : nil })
        #expect(saved.map(\.concept) == ["fresh-name"])
        #expect(saved.map(\.vectorArtifactID) == ["v1"])
    }

    @Test func explicitNoneIsTheOnlySanctionedRemoval() {
        // Setting the picker to "None" (vectorArtifactID = nil) is the user's
        // deliberate removal; an editor-added row whose vector vanished
        // between add and save is still kept (labeled), never dropped.
        var drafts = VariantInjectionEditing.drafts(from: foreignRefs)
        drafts[0].vectorArtifactID = nil
        let orphan = VariantInjectionEditing.Draft(
            vectorArtifactID: "gone/from/catalog", originalConcept: nil,
            layer: 8, alpha: 2)
        let saved = VariantInjectionEditing.injectionRefs(
            drafts: drafts + [orphan],
            resolveConcept: { _ in nil })
        #expect(saved.map(\.vectorArtifactID) == [
            foreignRefs[1].vectorArtifactID, "gone/from/catalog",
        ])
        #expect(saved.map(\.concept) == ["fear", "vector"])
    }

    @Test func editedLayerAndAlphaApplyWithoutTouchingTheRef() {
        // Editing numbers on a foreign ref keeps concept + id verbatim and
        // honors the edit — data-safety does not mean read-only.
        var drafts = VariantInjectionEditing.drafts(from: foreignRefs)
        drafts[0].layer = 9
        drafts[0].alpha = 0.75
        let saved = VariantInjectionEditing.injectionRefs(
            drafts: drafts,
            resolveConcept: { _ in nil })
        #expect(saved[0].concept == "independent")
        #expect(saved[0].vectorArtifactID == foreignRefs[0].vectorArtifactID)
        #expect(saved[0].layer == 9)
        #expect(saved[0].alpha == 0.75)
        #expect(saved[1] == foreignRefs[1])
    }

    @Test func unresolvedLabelSpellsOutTheForeignRef() {
        #expect(
            VariantInjectionEditing.unresolvedLabel(
                concept: "independent", ref: "/srv/runs/x/independent")
                == "independent · /srv/runs/x/independent (not in this workspace's catalog)")
        #expect(
            VariantInjectionEditing.unresolvedLabel(concept: nil, ref: "/srv/runs/x/v")
                == "vector · /srv/runs/x/v (not in this workspace's catalog)")
    }
}
