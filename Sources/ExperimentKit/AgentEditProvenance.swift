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
/// The rule:
/// - an edit that leaves what the agent does unchanged (a rename, or no edit
///   at all) writes the stored content back with only the name changed, so
///   the promotion record, the injections' concept labels (which a verify
///   gate compares with the sweep's), and every other value are kept (a file
///   the Mac wrote comes back byte for byte; a file another engine wrote is
///   re-encoded with the same values). A
///   system prompt stored without its hash gains the hash, the repair a freeze
///   refusal asks for ("re-save the variant");
/// - an edit that changes what the agent does cannot keep the record, because
///   the record would vouch for settings it never tested. The record is
///   dropped and the caller says so.
public enum AgentEditProvenance {

    public struct Merge: Sendable, Equatable {
        /// The artifact to write over the stored file.
        public var artifact: ModelVariantArtifact
        /// True when the stored file had a promotion record that this save
        /// does not keep, because the edit changes what the agent does.
        public var droppedPromotion: Bool
    }

    /// The title and message of the question Save Changes asks when the
    /// edits would drop the record. Asked at save time, after the editor's
    /// own reference rewrites, so it never fires on an agent nobody edited.
    public static func dialogTitle(agent: String) -> String {
        "Save changes to '\(agent)'?"
    }

    /// `inLibrary` is false for an agent a run saved: its folder is evidence,
    /// so it is never rewritten and the only way on is a copy.
    public static func dialogMessage(inLibrary: Bool) -> String {
        inLibrary
            ? "These edits change what the agent does, so its promotion record would no "
                + "longer describe it, and Save Changes and Remove the Record removes it. "
                + "Save as New Agent keeps this agent and its record as they are."
            : "These edits change what the agent does. This agent was saved by a run, "
                + "and a run's folder is kept as evidence, so the edits can only be saved "
                + "as a new agent; this agent and its promotion record stay as they are."
    }

    /// Said after a save that dropped the record.
    public static func afterSaveNote(agent: String) -> String {
        "Updated agent '\(agent)' and removed its promotion record: the edits change "
            + "what the agent does, so the record no longer describes it."
    }

    /// The name Save as New Agent gives the copy: the editor's name when no
    /// library agent has it, else that name with "-edited" (and a number)
    /// appended, so the original keeps its name.
    public static func copyName(_ name: String, taken: [String]) -> String {
        // The same comparison the editor's collision check uses.
        let names = taken.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        func isTaken(_ candidate: String) -> Bool {
            names.contains { $0.compare(candidate, options: .caseInsensitive) == .orderedSame }
        }
        let base = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !isTaken(base) { return base }
        var candidate = base + "-edited"
        var counter = 2
        while isTaken(candidate) {
            candidate = base + "-edited-\(counter)"
            counter += 1
        }
        return candidate
    }

    public static func merge(
        edited: ModelVariantArtifact, over stored: ModelVariantArtifact
    ) -> Merge {
        // An older file stores only the system prompt's hash; the editor shows
        // it as an empty prompt. Unchanged in the editor means unchanged.
        let promptUnchanged = (edited.systemPrompt ?? "") == (stored.systemPrompt ?? "")
        if doesTheSame(edited, as: stored) {
            var kept = stored
            kept.name = edited.name
            // A stored reference outside this workspace (a cluster's or
            // another machine's absolute path) that names the same file as
            // the editor's row is written in the editor's workspace-relative
            // form: portable, and no machine path left in the file. The
            // record vouches for the vector by its hash, not its path.
            for index in kept.injections.indices where index < edited.injections.count {
                let storedReference = kept.injections[index].vectorArtifactID
                guard storedReference.hasPrefix("/"),
                    ArtifactIdentity.workspaceRelative(storedReference) == storedReference
                else { continue }
                let editedReference = ArtifactIdentity.workspaceRelative(
                    edited.injections[index].vectorArtifactID)
                guard !editedReference.hasPrefix("/"),
                    ArtifactIdentity.canonical(storedReference)
                        == ArtifactIdentity.canonical(editedReference)
                else { continue }
                kept.injections[index].vectorArtifactID = editedReference
            }
            if promptUnchanged, (kept.systemPromptHash ?? "").isEmpty {
                kept.systemPromptHash = edited.systemPromptHash
            }
            return Merge(artifact: kept, droppedPromotion: false)
        }
        var merged = edited
        merged.createdAt = stored.createdAt
        if promptUnchanged {
            merged.systemPrompt = stored.systemPrompt
            merged.systemPromptHash = (stored.systemPromptHash ?? "").isEmpty
                ? edited.systemPromptHash : stored.systemPromptHash
        }
        return Merge(artifact: merged, droppedPromotion: stored.promotion != nil)
    }

    /// Whether two agents act identically: everything but the name, the
    /// record of their making, and labels. Vector references are compared
    /// workspace-relative, the form both are saved in, and every field in its
    /// effective form (an absent mode is steering, empty ablation layers are
    /// every layer, absent centering is none), so spellings that mean the same
    /// thing never read as an edit.
    static func doesTheSame(_ a: ModelVariantArtifact, as b: ModelVariantArtifact) -> Bool {
        func behaviour(_ artifact: ModelVariantArtifact) -> ModelVariantArtifact {
            var copy = ModelVariantStore.workspaceRelativeNormalized(artifact)
            copy.name = ""
            copy.createdAt = ""
            copy.promotion = nil
            copy.neutralPCBasisLabel = nil
            // The prompt is compared as the editor shows it, below: its hash
            // is derived from the text, and a hash-only older file shows as
            // an empty prompt the editor cannot change without typing one.
            copy.systemPrompt = nil
            copy.systemPromptHash = nil
            copy.adapters = copy.adapters.map { adapter in
                var plain = adapter
                plain.name = ""  // labels; the directory and hashes are what load
                plain.artifactPath = ""
                return plain
            }
            copy.injections = copy.injections.map { injection in
                let ablates = injection.effectiveMode == .ablate
                var plain = injection
                // By resolved location, not spelling: the editor shows a
                // saved reference as its catalog entry, and a reference
                // written on another machine resolves into this workspace.
                plain.vectorArtifactID = ArtifactIdentity.canonical(injection.vectorArtifactID)
                plain.concept = ""  // a label; the vector reference is what acts
                plain.mode = ablates ? .ablate : nil
                plain.layers = ablates && !(injection.layers ?? []).isEmpty
                    ? injection.layers : nil
                plain.centering = ablates && injection.effectiveCentering != "none"
                    ? injection.centering : nil
                return plain
            }
            return copy
        }
        return behaviour(a) == behaviour(b)
            && (a.systemPrompt ?? "") == (b.systemPrompt ?? "")
    }
}
