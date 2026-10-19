import Foundation

/// What saving an edited agent over its stored file keeps from that file.
///
/// The agent editor has controls only for what an agent DOES. It has none for
/// the record of how the agent came to be: the promotion certificate (which
/// sweep cell, chosen how, from which evidence), the creation time, and an
/// older file's system-prompt hash. Rebuilding the agent from the editor
/// alone and writing it over the file therefore erased the promotion record on
/// every "Save Changes", even with nothing edited, so a promoted agent became
/// indistinguishable from a hand-made one.
///
/// The rule: an edit that leaves what the agent does unchanged (a rename, or
/// no edit at all) keeps the record exactly. An edit that changes it cannot
/// keep the record, because the record would then vouch for settings it never
/// tested; the record is dropped and the caller says so.
public enum AgentEditProvenance {

    public struct Merge: Sendable, Equatable {
        /// The artifact to write over the stored file.
        public var artifact: ModelVariantArtifact
        /// True when the stored file had a promotion record that this save
        /// does not keep, because the edit changes what the agent does.
        public var droppedPromotion: Bool
    }

    public static func merge(
        edited: ModelVariantArtifact, over stored: ModelVariantArtifact
    ) -> Merge {
        var merged = edited
        merged.createdAt = stored.createdAt
        // An older file stores only the system prompt's hash; the editor shows
        // it as an empty prompt. Unchanged in the editor means unchanged.
        let promptUnchanged = (edited.systemPrompt ?? "") == (stored.systemPrompt ?? "")
        if promptUnchanged {
            merged.systemPrompt = stored.systemPrompt
            merged.systemPromptHash = stored.systemPromptHash
        }
        if doesTheSame(merged, as: stored) {
            merged.promotion = stored.promotion
            return Merge(artifact: merged, droppedPromotion: false)
        }
        return Merge(artifact: merged, droppedPromotion: stored.promotion != nil)
    }

    /// Whether two agents act identically: everything but the name, the
    /// record of their making, and display labels. Vector references are
    /// compared workspace-relative, the form both are saved in.
    static func doesTheSame(_ a: ModelVariantArtifact, as b: ModelVariantArtifact) -> Bool {
        func behaviour(_ artifact: ModelVariantArtifact) -> ModelVariantArtifact {
            var copy = ModelVariantStore.workspaceRelativeNormalized(artifact)
            copy.name = ""
            copy.createdAt = ""
            copy.promotion = nil
            copy.neutralPCBasisLabel = nil
            copy.injections = copy.injections.map { injection in
                var plain = injection
                plain.concept = ""  // a label; the vector reference is what acts
                return plain
            }
            return copy
        }
        return behaviour(a) == behaviour(b)
    }
}
