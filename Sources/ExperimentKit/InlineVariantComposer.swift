import Foundation
import SteeringKit

/// Pure assembly of an INLINE server variant spec from the chat Steering
/// pane's live control state, plus the stored-vs-inline send decision.
///
/// Design rule this encodes: in a server workspace the stored variant is a
/// *seed* for exploration, exactly like the local workspace. An untouched seed
/// sends the stored `variantPath` + pinned hash (exact provenance); any edit —
/// or controls built with no stored variant selected — composes an inline
/// `ModelVariantArtifact` (the `/api/variants/upload` payload schema, so the
/// wire shape is identical by construction) and the server stamps it
/// `source: "inline"` with no path/hash, so a tweaked chat never claims a
/// stored variant's identity.
///
/// Everything here is pure and MainActor-free so the composition and the
/// dirty-tracking rule are unit-testable without a server or a model.
public enum InlineVariantComposer {

    /// One enabled steering box, already resolved to a SERVER vector artifact
    /// (`vectorPath` is the server-side id/path from `GET /api/vectors` — a
    /// local vector path must never appear here; strict per-substrate
    /// availability is the caller's filter).
    public struct Slot: Equatable, Sendable {
        public var concept: String
        public var vectorPath: String
        public var layer: Int
        /// α when steering, λ when ablating.
        public var alpha: Double
        /// Steer or ablate. Carried all the way to the server (2026-07-27):
        /// when this field did not exist, an ablation composed in the
        /// Playground reached the cluster as an ORDINARY steering injection
        /// with `alpha = λ` — indistinguishable from steering because the
        /// wire spells the default by omitting it. The picker said ablate and
        /// the model steered.
        public var mode: InterventionPlan.Mode
        /// Ablation-direction centering ("neutralMean" or nil = none),
        /// carried to the server the same way `mode` is: the Mac never holds
        /// server vector bytes, so a server-workspace ablation can only be
        /// centered by declaring it in the composed spec and letting
        /// `variant_injections` apply the artifact's stored mean.
        public var centering: String?

        public init(
            concept: String, vectorPath: String, layer: Int, alpha: Double,
            mode: InterventionPlan.Mode = .add,
            centering: String? = nil
        ) {
            self.concept = concept
            self.vectorPath = vectorPath
            self.layer = layer
            self.alpha = alpha
            self.mode = mode
            self.centering = centering
        }
    }

    /// Snapshot of every control that participates in a server-side variant
    /// generation. Equatable so "dirty since seeded" is literally
    /// `seededState != currentState` — no per-field bookkeeping to forget.
    ///
    /// `slots` is already the *effective* injection list: empty when the
    /// master "Inject vectors" switch is off, only enabled boxes with a
    /// resolved server vector otherwise. Same for `adapters` and the adapter
    /// toggle. That way toggling steering off IS a state change (dirty), and
    /// the composed spec never smuggles disabled controls.
    public struct ControlState: Equatable, Sendable {
        public var baseModelID: String
        public var slots: [Slot]
        public var adapters: [ModelVariantArtifact.AdapterRef]
        public var bandWidth: Int
        public var alphaInNormUnits: Bool
        /// Carried through from a seeded variant (not editable in chat yet) so
        /// an edited spec doesn't silently drop the variant's neutral basis.
        public var neutralPCBasisPath: String?
        public var neutralPCBasisLabel: String?
        public var promptMode: String
        public var qwenThinkingEnabled: Bool
        public var temperature: Double
        public var systemPrompt: String

        public init(
            baseModelID: String,
            slots: [Slot] = [],
            adapters: [ModelVariantArtifact.AdapterRef] = [],
            bandWidth: Int = 1,
            alphaInNormUnits: Bool = false,
            neutralPCBasisPath: String? = nil,
            neutralPCBasisLabel: String? = nil,
            promptMode: String,
            qwenThinkingEnabled: Bool = false,
            temperature: Double = 0,
            systemPrompt: String = ""
        ) {
            self.baseModelID = baseModelID
            self.slots = slots
            self.adapters = adapters
            self.bandWidth = bandWidth
            self.alphaInNormUnits = alphaInNormUnits
            self.neutralPCBasisPath = neutralPCBasisPath
            self.neutralPCBasisLabel = neutralPCBasisLabel
            self.promptMode = promptMode
            self.qwenThinkingEnabled = qwenThinkingEnabled
            self.temperature = temperature
            self.systemPrompt = systemPrompt
        }

        /// Anything that makes a plain generate insufficient.
        public var hasInterventions: Bool { !slots.isEmpty || !adapters.isEmpty }
    }

    /// One steering box as the chat holds it, BEFORE catalog resolution —
    /// the raw input to `resolveSlots`. Kept engine-agnostic (plain ids) so
    /// the drop-accounting rule is pure and unit-testable.
    public struct SlotInput: Equatable, Sendable {
        public var vectorID: String?
        public var layer: Int
        public var alpha: Double
        public var enabled: Bool
        public var mode: InterventionPlan.Mode
        /// See `Slot.centering` — meaningful only when `mode == .ablate`.
        public var centering: String?

        public init(
            vectorID: String?, layer: Int, alpha: Double, enabled: Bool,
            mode: InterventionPlan.Mode = .add,
            centering: String? = nil
        ) {
            self.vectorID = vectorID
            self.layer = layer
            self.alpha = alpha
            self.enabled = enabled
            self.mode = mode
            self.centering = centering
        }
    }

    /// Honest composition result: the slots that made it into the effective
    /// injection list, plus the vector ids of ENABLED slots that were dropped
    /// because they did not resolve against the server catalog. A non-empty
    /// `unresolvedVectorIDs` means the user configured steering that will NOT
    /// be sent — callers must surface it (caption/error/refusal), never
    /// swallow it.
    public struct SlotResolution: Equatable, Sendable {
        public var slots: [Slot]
        public var unresolvedVectorIDs: [String]

        public init(slots: [Slot], unresolvedVectorIDs: [String]) {
            self.slots = slots
            self.unresolvedVectorIDs = unresolvedVectorIDs
        }
    }

    /// Resolve the chat's steering boxes into composer slots, accounting for
    /// every drop. Rules:
    /// - master switch off → no slots AND no drops (intentionally unsteered);
    /// - disabled boxes are skipped, not dropped (intentional);
    /// - enabled boxes with no vector picked are skipped, not dropped
    ///   (nothing was claimed);
    /// - enabled boxes whose id resolves (`concept` returns a name) are
    ///   composed;
    /// - enabled boxes whose id does NOT resolve are reported in
    ///   `unresolvedVectorIDs` — the silent-drop class this seam exists to
    ///   make loud. The id is still excluded from the composed spec: an
    ///   unresolvable (possibly local-MLX) path must never reach the server.
    public static func resolveSlots(
        steeringEnabled: Bool,
        slots: [SlotInput],
        concept: (String) -> String?
    ) -> SlotResolution {
        guard steeringEnabled else {
            return SlotResolution(slots: [], unresolvedVectorIDs: [])
        }
        var resolved: [Slot] = []
        var unresolved: [String] = []
        for slot in slots where slot.enabled {
            guard let id = slot.vectorID else { continue }
            if let concept = concept(id) {
                resolved.append(
                    Slot(
                        concept: concept, vectorPath: id, layer: slot.layer,
                        alpha: slot.alpha, mode: slot.mode,
                        centering: slot.mode == .ablate ? slot.centering : nil))
            } else {
                unresolved.append(id)
            }
        }
        return SlotResolution(slots: resolved, unresolvedVectorIDs: unresolved)
    }

    /// Status line for a REFUSED server-workspace Save Variant (a definition
    /// with silently missing injections is a provenance hazard — never write
    /// one). Nil when nothing was dropped.
    public static func unresolvedSlotRefusal(unresolvedVectorIDs: [String]) -> String? {
        guard !unresolvedVectorIDs.isEmpty else { return nil }
        let list = unresolvedVectorIDs.map { "'\($0)'" }.joined(separator: ", ")
        if unresolvedVectorIDs.count == 1 {
            return "not saved: slot \(list) does not resolve in the server catalog "
                + "listing — hit Refresh artifacts (or re-pick the vector), then save again"
        }
        return "not saved: \(unresolvedVectorIDs.count) slots (\(list)) do not resolve "
            + "in the server catalog listing — hit Refresh artifacts (or re-pick the "
            + "vectors), then save again"
    }

    /// Persistent-caption warning while enabled slots are being dropped from
    /// composed sends — the user must never chat believing they are steering
    /// when they are not. Nil when nothing is dropped.
    public static func unresolvedSlotCaption(count: Int) -> String? {
        guard count > 0 else { return nil }
        return "\(count) enabled slot\(count == 1 ? "" : "s") unresolved — "
            + "NOT sent; Refresh artifacts"
    }

    /// Error-surface message for a send that proceeded while dropping
    /// enabled slots (set alongside the caption so the condition is visible
    /// next to the transcript, not only under the variant picker).
    public static func unresolvedSendWarning(unresolvedVectorIDs: [String]) -> String? {
        guard !unresolvedVectorIDs.isEmpty else { return nil }
        let count = unresolvedVectorIDs.count
        let list = unresolvedVectorIDs.map { "'\($0)'" }.joined(separator: ", ")
        return "\(count) enabled steering slot\(count == 1 ? "" : "s") (\(list)) "
            + "did not resolve in the server catalog listing and w"
            + (count == 1 ? "as" : "ere")
            + " NOT sent — this generation ran without \(count == 1 ? "it" : "them"); "
            + "hit Refresh artifacts or re-pick the vector"
    }

    /// The stored-variant SHADOW warning: when a stored variant is selected
    /// but its spec could NOT be seeded into the controls (older server
    /// without the detail route, transient fetch failure), dirty-tracking is
    /// impossible — every send runs the STORED variant by path and the
    /// configured steering rows silently never ride. This names that state
    /// whenever rows are configured, so the user cannot chat believing extra
    /// vectors compose with the agent when they do not. Nil when seeding
    /// worked (edits switch to inline normally), when no stored variant is
    /// selected, or when no rows are configured.
    public static func unseededStoredVariantWarning(
        storedVariantSelected: Bool,
        seeded: Bool,
        configuredSlotCount: Int
    ) -> String? {
        guard storedVariantSelected, !seeded, configuredSlotCount > 0 else { return nil }
        let rows = "\(configuredSlotCount) steering row"
            + (configuredSlotCount == 1 ? "" : "s")
        return "\(rows) configured, but the stored agent's spec was not "
            + "seeded — sends run the STORED agent only and these rows do "
            + "NOT ride; re-select the agent (to seed the controls) or "
            + "clear it to steer with these rows"
    }

    /// How the next server send should run.
    public enum Disposition: Equatable, Sendable {
        /// Untouched seed → the stored `variantPath` + pinned hash.
        case storedVariant
        /// Edited seed, or controls built from scratch → inline spec.
        case inlineVariant
        /// No variant and no interventions → plain `/api/generate/stream`.
        case plainGenerate
    }

    /// The send decision. `seeded == nil` with a stored selection means the
    /// controls could not be seeded (older server without the detail route):
    /// dirty-tracking is impossible, so the stored path keeps exact provenance
    /// — the pre-existing behavior.
    public static func disposition(
        storedVariantSelected: Bool,
        seeded: ControlState?,
        current: ControlState
    ) -> Disposition {
        if storedVariantSelected {
            if let seeded, seeded != current { return .inlineVariant }
            return .storedVariant
        }
        return current.hasInterventions ? .inlineVariant : .plainGenerate
    }

    /// Compose the inline spec. Field-for-field the `variant` object of the
    /// upload payload (`ModelVariantArtifact`'s own Codable encoding):
    /// `injections[].vectorArtifactID` are SERVER artifact paths, adapters are
    /// the server's adapter refs, and the trimmed-empty system prompt encodes
    /// as absent — same normalization a saved local variant gets.
    public static func compose(
        _ state: ControlState,
        name: String = "chat-inline"
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name,
            baseModelID: state.baseModelID,
            adapters: state.adapters,
            injections: state.slots.map {
                ModelVariantArtifact.InjectionRef(
                    concept: $0.concept,
                    vectorArtifactID: $0.vectorPath,
                    layer: $0.layer,
                    alpha: $0.alpha,
                    // Absent for steering (the encoder omits `.add`), so a
                    // steering spec's bytes are unchanged; present and
                    // explicit for ablation, which is the whole point.
                    mode: $0.mode == .ablate ? .ablate : nil,
                    // Same wire discipline as `mode`: only an ablation's
                    // declared centering travels; the default stays absent.
                    centering: $0.mode == .ablate ? $0.centering : nil)
            },
            bandWidth: state.bandWidth,
            alphaInNormUnits: state.alphaInNormUnits,
            neutralPCBasisPath: state.neutralPCBasisPath,
            neutralPCBasisLabel: state.neutralPCBasisLabel,
            promptMode: state.promptMode,
            qwenThinkingEnabled: state.qwenThinkingEnabled,
            temperature: state.temperature,
            systemPrompt: state.systemPrompt)
    }

    /// A server-workspace "Save Variant": the composed spec IS the saved
    /// definition — same assembly as an inline send, so what lands in the
    /// local variant store (definitions are git-versioned recipes) is exactly
    /// what the user is running: the SERVER base model and SERVER vector /
    /// adapter refs. When no server model is picked yet, the model already
    /// loaded on the server is the base. Nil when neither exists — there is
    /// no truthful base model to record.
    public static func captureArtifact(
        state: ControlState,
        name: String,
        loadedServerModelID: String?
    ) -> ModelVariantArtifact? {
        var state = state
        if state.baseModelID.isEmpty, let loadedServerModelID,
            !loadedServerModelID.isEmpty
        {
            state.baseModelID = loadedServerModelID
        }
        guard !state.baseModelID.isEmpty else { return nil }
        return compose(state, name: name)
    }
}
