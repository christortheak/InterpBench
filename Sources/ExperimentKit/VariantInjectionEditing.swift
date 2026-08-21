import Foundation
import SteeringKit

/// Pure logic for the Model Variants tab's injections editor: how a saved
/// definition's injection refs become editable drafts and how drafts become
/// refs again on save.
///
/// Data-safety rule this encodes: an injection ref the ACTIVE workspace
/// cannot resolve (a server-composed definition opened in a Local workspace,
/// a vector deleted since capture, a ref from another server) must
/// round-trip UNTOUCHED through open → save — never silently dropped. The
/// editor renders it as foreign ("… (not in this workspace's catalog)") and
/// the user can delete it explicitly, but saving must not eat it. Definitions
/// are git-versioned recipes visible in every workspace; the workspace only
/// scopes which NEW vectors the picker offers.
public enum VariantInjectionEditing {

    /// One editable injection row. `originalConcept` carries the concept the
    /// loaded artifact recorded for this ref, so an unresolvable ref can be
    /// re-emitted verbatim on save (and labeled) without a catalog lookup.
    /// Nil for rows added fresh in the editor (those are picked from the
    /// workspace catalog, so they resolve by construction at add time).
    public struct Draft: Equatable, Sendable {
        public var vectorArtifactID: String?
        public var originalConcept: String?
        public var layer: Int
        /// α when steering, λ when ablating.
        public var alpha: Double
        /// Defaulted so every existing construction is unchanged.
        public var mode: InterventionPlan.Mode

        public init(
            vectorArtifactID: String?,
            originalConcept: String? = nil,
            layer: Int,
            alpha: Double,
            mode: InterventionPlan.Mode = .add
        ) {
            self.vectorArtifactID = vectorArtifactID
            self.originalConcept = originalConcept
            self.layer = layer
            self.alpha = alpha
            self.mode = mode
        }
    }

    /// Open: a definition's refs as editable drafts, concepts preserved.
    public static func drafts(
        from injections: [ModelVariantArtifact.InjectionRef]
    ) -> [Draft] {
        injections.map {
            Draft(
                vectorArtifactID: $0.vectorArtifactID,
                originalConcept: $0.concept,
                layer: $0.layer,
                alpha: $0.alpha,
                mode: $0.effectiveMode)
        }
    }

    /// Save: drafts back to refs. `resolveConcept` is the ACTIVE workspace's
    /// catalog lookup (local sidecars or the active server's vector listing).
    /// - resolves → the catalog's concept (fresh metadata for a re-pick);
    /// - does not resolve but the draft carries its original concept → the
    ///   ref is re-emitted with the ORIGINAL concept and untouched id (the
    ///   round-trip guarantee; layer/alpha stay whatever the editor holds);
    /// - does not resolve and never had a concept (an editor-added row whose
    ///   vector vanished between add and save) → kept too, labeled "vector",
    ///   because dropping is the one unacceptable outcome;
    /// - no vector picked (explicit "None") → omitted: that is the user's
    ///   deliberate removal, the only sanctioned way a ref leaves the array.
    public static func injectionRefs(
        drafts: [Draft],
        resolveConcept: (String) -> String?
    ) -> [ModelVariantArtifact.InjectionRef] {
        drafts.compactMap { draft in
            guard let id = draft.vectorArtifactID else { return nil }
            let concept = resolveConcept(id) ?? draft.originalConcept ?? "vector"
            return ModelVariantArtifact.InjectionRef(
                concept: concept,
                vectorArtifactID: id,
                layer: draft.layer,
                alpha: draft.alpha,
                // Absent for steering, so an untouched agent's bytes — and
                // therefore its artifact hash and promotion key — do not move.
                mode: draft.mode == .ablate ? .ablate : nil)
        }
    }

    /// Picker row label for a draft whose ref the active workspace cannot
    /// resolve — rendered, selectable-as-current, never offered anew.
    public static func unresolvedLabel(concept: String?, ref: String) -> String {
        "\(concept ?? "vector") · \(ref) (not in this workspace's catalog)"
    }
}
