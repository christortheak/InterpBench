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
///
/// All of that is for an agent in the library (`runs/model-variants/`). An
/// agent a run saved (`runs/<run>/<name>.json`, how the Python engine's save
/// and promote store them) lives in that run's folder, which is evidence:
/// Save Changes never writes it, and saves the edits as a new library agent
/// instead (`saveChangesPlan`).
public enum AgentEditProvenance {

    public struct Merge: Sendable, Equatable {
        /// The artifact to write over the stored file.
        public var artifact: ModelVariantArtifact
        /// True when the stored file had a promotion record that this save
        /// does not keep, because the edit changes what the agent does.
        public var droppedPromotion: Bool
    }

    /// What Save Changes does with the agent being edited.
    public enum SaveChangesPlan: Sendable, Equatable {
        /// A library agent: `merge.artifact` is written over its file. When
        /// `merge.droppedPromotion` is true the editor asks first.
        case updateInPlace(Merge)
        /// An agent a run saved, with something to save: the editor's agent
        /// becomes a new library agent, and the run's file is not written.
        case saveAsNewAgent
        /// An agent a run saved, and saving would write exactly what its file
        /// already holds: nothing is written, so no pointless copy appears.
        case nothingToSave
    }

    /// Decides Save Changes for the agent stored as `stored`. `inLibrary` is
    /// `ModelVariantStore.isInLibrary` of its file.
    ///
    /// For an agent a run saved, "nothing to save" means the in-place save
    /// would have written the file's own content back: no edit, and no
    /// repair either. A save that would stamp a missing system-prompt hash
    /// (the repair a freeze refusal asks for) or make a vector reference
    /// portable is something to save, and goes to the copy.
    public static func saveChangesPlan(
        edited: ModelVariantArtifact, over stored: ModelVariantArtifact, inLibrary: Bool
    ) -> SaveChangesPlan {
        let merge = merge(edited: edited, over: stored)
        if inLibrary { return .updateInPlace(merge) }
        return ModelVariantStore.workspaceRelativeNormalized(merge.artifact) == stored
            ? .nothingToSave : .saveAsNewAgent
    }

    /// `saveChangesPlan` for a listed agent, judged by where its file is.
    public static func saveChangesPlan(
        edited: ModelVariantArtifact, over record: ModelVariantRecord
    ) -> SaveChangesPlan {
        saveChangesPlan(
            edited: edited, over: record.artifact,
            inLibrary: ModelVariantStore.isInLibrary(record.url))
    }

    /// The title and message of the question Save Changes asks when the
    /// edits would drop the record. Asked at save time, after the editor's
    /// own reference rewrites, so it never fires on an agent nobody edited.
    /// Only a library agent is asked: an agent a run saved is never written,
    /// so its edits go to a new agent and its record stays where it is.
    public static func dialogTitle(agent: String) -> String {
        "Save changes to '\(agent)'?"
    }

    public static func dialogMessage() -> String {
        "These edits change what the agent does, so its promotion record would no "
            + "longer describe it, and Save Changes and Remove the Record removes it. "
            + "Save as New Agent keeps this agent and its record as they are."
    }

    /// Said after a save that dropped the record.
    public static func afterSaveNote(agent: String) -> String {
        "Updated agent '\(agent)' and removed its promotion record: the edits change "
            + "what the agent does, so the record no longer describes it."
    }

    // MARK: - Agents a run saved

    /// The editor's save button. For an agent a run saved it names what the
    /// button does, before anyone presses it.
    public static func saveButtonTitle(editing: Bool, inLibrary: Bool) -> String {
        guard editing else { return "Save New Agent" }
        return inLibrary ? "Save Changes" : "Save as New Agent"
    }

    /// The save button's help for an agent a run saved.
    public static let savedByRunButtonHelp =
        "this agent was saved by a run, and a run's folder is kept as evidence, so "
        + "your edits are saved as a new agent in the library and the run's agent "
        + "stays as it is"

    /// The caption under the agent picker. `run` is the folder of the run
    /// that saved the agent, nil for a library agent.
    public static func editingCaption(agent: String, savedByRun run: String?) -> String {
        guard let run else {
            return "editing the saved agent '\(agent)' — Save Changes updates it in "
                + "place; choose New… above to start a fresh definition instead"
        }
        return "editing '\(agent)', which the run \(run) saved — a run's folder is "
            + "kept as evidence, so Save as New Agent saves your edits as a new agent "
            + "in the library and leaves this one as it is; choose New… above to "
            + "start a fresh definition instead"
    }

    /// Said after Save as New Agent. `savedByRun` is true when the original
    /// is an agent a run saved, which Save Changes never rewrites.
    public static func savedAsNewAgentNote(
        agent: String, original: String, savedByRun: Bool, originalHasRecord: Bool
    ) -> String {
        guard savedByRun else {
            return "Saved a new agent '\(agent)'; '\(original)' is unchanged."
        }
        let note = "Saved a new agent '\(agent)' in the library. '\(original)' was saved "
            + "by a run, and a run's folder is kept as evidence, so it is unchanged."
        return originalHasRecord
            ? note + " The new agent has no promotion record; that record stays with "
                + "'\(original)'."
            : note
    }

    /// Said when Save Changes on an agent a run saved finds nothing to save.
    public static func nothingToSaveNote(agent: String) -> String {
        "Nothing to save: the editor shows '\(agent)' as it is stored. It was saved "
            + "by a run, and a run's folder is kept as evidence, so edits you make are "
            + "saved as a new agent in the library."
    }

    /// The name Save as New Agent gives the copy: the editor's name when no
    /// listed agent has it, else that name with "-edited" (and a number)
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
    /// every layer, a layer list is a set whatever its order or repeats,
    /// absent centering is none), so spellings that mean the same thing never
    /// read as an edit.
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
                plain.layers = ablates ? injection.declaredAblationLayers : nil
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
