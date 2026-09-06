import Foundation
import Observation
import SteeringKit

/// Study/design commands and inventory. Store admission remains authoritative.
/// The coordinator supplies presentation callbacks and workspace model choices;
/// this owner never retains a panel, chat service or job controller.
@Observable @MainActor
public final class StudyManagementController {
    public let draft: StudyDraftState
    public let designs = StudyDesignLibrary()
    public private(set) var experiments: [ExperimentManifest] = []
    public private(set) var displayLabels: [String: String] = [:]
    public var renameInvitation: String?
    public var selectedName: String? {
        didSet {
            if oldValue != selectedName { presentation.selectionChanged() }
        }
    }
    public var selected: ExperimentManifest? {
        experiments.first { $0.name == selectedName }
    }
    @ObservationIgnored private var authoringReview: DraftAuthoringSnapshot?
    @ObservationIgnored private var reviewedDrafts: [String: DraftAuthoringSnapshot] = [:]
    @ObservationIgnored var presentation = StudyManagementPresentation()

    public init(draft: StudyDraftState) { self.draft = draft }

    public func refresh() {
        let root = ExperimentStore.workspaceRoot
        let storage = ExperimentRepository(workspaceRoot: root)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: storage.directory.path)) ?? []
        let snapshots = names.compactMap { try? DraftAuthoringSnapshot(workspaceRoot: root, name: $0) }
        reviewedDrafts = Dictionary(uniqueKeysWithValues: snapshots.map { ($0.manifest.name, $0) })
        experiments = snapshots.map(\.manifest).sorted { $0.createdAt > $1.createdAt }
        displayLabels = ExperimentStore.displayLabels(experiments)
        refreshTemplates()
        if let selectedName, !experiments.contains(where: { $0.name == selectedName }) {
            self.selectedName = nil
        }
        presentation.refreshed()
    }

    /// Persist the document shown by this owner, using the exact read that
    /// supplied the displayed manifest. The target does not follow selection
    /// or a workspace switch during an operation.
    func persistReviewedDraft(_ manifest: ExperimentManifest) throws {
        let reviewed = try reviewedDraft(named: manifest.name)
        acceptAuthoringResult(try DraftAuthoringTransaction.replace(manifest, reviewed: reviewed))
    }

    /// Advancing the editor's review is explicit: selection/reload starts it,
    /// and a successful command from that editor advances it. Inventory refresh
    /// alone may update the displayed catalog but cannot authorize old fields.
    func beginAuthoringReview(named name: String?) {
        authoringReview = name.flatMap { reviewedDrafts[$0] }
    }

    func acceptAuthoringResult(_ saved: DraftAuthoringSnapshot) {
        reviewedDrafts[saved.manifest.name] = saved
        if authoringReview?.manifest.name == saved.manifest.name,
            authoringReview?.workspaceRoot == saved.workspaceRoot
        {
            authoringReview = saved
        }
    }

    public var selectedDraftNeedsReload: Bool {
        guard let selected else { return false }
        guard let reviewed = authoringReview,
            reviewed.manifest.name == selected.name,
            reviewed.workspaceRoot == ExperimentStore.workspaceRoot.standardizedFileURL
        else { return true }
        return reviewedDrafts[selected.name]?.file.sha256 != reviewed.file.sha256
    }

    func reviewedDraft(named name: String) throws -> DraftAuthoringSnapshot {
        guard let reviewed = authoringReview, reviewed.manifest.name == name,
            reviewed.workspaceRoot == ExperimentStore.workspaceRoot.standardizedFileURL
        else {
            throw ExperimentError.refusing(.staleManifest,
                "The reviewed draft is unavailable in this workspace.",
                repair: "Use Discard edits and reload to review the saved study, then apply the edit again.")
        }
        return reviewed
    }

    public func refreshTemplates() { designs.refresh(experiments: experiments) }

    private func note(_ message: String, severity: PanelNotice.Severity = .info) {
        presentation.note(message, severity)
    }

    private func clearFormError(_ field: StudyDraftState.FormField) {
        draft.clearFormError(field)
    }

    private func refuse(_ field: StudyDraftState.FormField, _ message: String) {
        draft.formErrors[field] = message
        note(message, severity: .error)
    }

    /// What to call this study in lists: its label when one is set, its
    /// canonical name otherwise. The canonical name stays the identity —
    /// directory, run stamps, CLI arguments — so surfaces render it as
    /// secondary text rather than dropping it.
    public func displayName(_ manifest: ExperimentManifest) -> String {
        displayLabels[manifest.name] ?? manifest.name
    }

    /// One-click study creation: mint a readable, unique placeholder name and
    /// invite the rename immediately. Naming a study before it exists is a
    /// decision the researcher cannot yet make — the draft can be renamed for
    /// as long as it stays a draft, so the name is not a gate on getting in.
    public func newStudy(context: StudyCreationContext?) {
        let placeholder = ExperimentStore.placeholderStudyName()
        draft.newName = placeholder
        create(context: context)
        // Only invite the rename when the draft actually landed.
        if selectedName == placeholder { renameInvitation = placeholder }
    }

    public func create(context: StudyCreationContext?) {
        guard let context else { return }
        do {
            // Workspace-scoped fallback: a server-target draft must never be
            // silently pinned to a local MLX id the server can't load.
            // Optional up-front revision pin (App gap A7): the store always
            // accepted it; empty keeps the auto-pin-from-HF-cache behavior.
            let revision = draft.newRevision.trimmingCharacters(in: .whitespacesAndNewlines)
            // The workspace fallback used to apply only when
            // studyBaseModelID was EMPTY. But loading any study sets it, so
            // after the first selection it was never empty and a new draft
            // silently inherited the previous study's model — including one
            // this workspace cannot offer, which then rendered as a
            // "(not installed)" row nobody could have chosen.
            let workspaceDefault = context.workspaceDefaultModelID
            let inventory = context.modelOptions
            let carried = draft.studyBaseModelID.trimmingCharacters(in: .whitespacesAndNewlines)
            let seed =
                (!carried.isEmpty && (inventory.isEmpty || inventory.contains(carried)))
                ? carried
                : workspaceDefault
            if !carried.isEmpty, seed != carried {
                note(
                    "new draft pinned to '\(seed)': the carried-over model "
                        + "'\(carried)' is not available in this workspace",
                    severity: .info)
            }
            var manifest = try ExperimentStore.create(
                name: draft.newName,
                description: draft.newDescription,
                modelID: seed,
                modelRevision: revision.isEmpty ? nil : revision)
            manifest.studyKind = draft.studyKind
            manifest.temperature = 0
            manifest.maxTokens = 2048
            manifest.promptMode = .chatAssistant
            manifest.qwenThinkingEnabled = nil
            manifest.reasoningEffort = ReasoningEffort.off.rawValue
            manifest.reasoningMaxTokens = nil
            try ExperimentStore.save(manifest)
            draft.newName = ""
            draft.newDescription = ""
            draft.newRevision = ""
            refresh()
            selectedName = manifest.name
            note(
                "created draft protocol '\(manifest.name)' (model \(manifest.modelID)"
                    + (manifest.modelRevision.map { ", revision \($0.prefix(12))…)" } ?? ")"),
                severity: .success)
        } catch {
            note(
                "Couldn't create the draft — check the name isn't already in "
                    + "use and the workspace is writable. Details: \(error)",
                severity: .error)
        }
    }

    /// Rename the selected study. Two independent effects, both optional,
    /// applied in one action so the researcher sees one "Rename":
    ///
    /// - `canonicalName` (drafts only) moves `experiments/<old>/` to
    ///   `experiments/<new>/` and rewrites the manifest name. Runs already
    ///   recorded under the old name are immutable and are NOT touched — the
    ///   outcome's note says so, because they stop listing under this study.
    /// - `label` writes the hash-exempt display-label sidecar, allowed at
    ///   every status: it is not manifest content, so it cannot move the
    ///   content hash or re-epoch a frozen study's runs.
    ///
    /// The canonical rename runs FIRST so the label lands in the study's
    /// final directory.
    public func renameSelected(canonicalName: String?, label: String?) {
        guard let name = selectedName else { return }
        clearFormError(.rename)
        var current = name
        var messages: [String] = []
        if let canonicalName,
            ExperimentStore.resolvedRenameTarget(canonicalName) != name
        {
            do {
                let outcome = try ExperimentStore.rename(
                    experimentName: name, to: canonicalName)
                current = outcome.newName
                messages.append("renamed '\(outcome.oldName)' → '\(outcome.newName)'")
                if let runsNote = outcome.runsNote { messages.append(runsNote) }
            } catch {
                refuse(.rename, "Couldn't rename the study — nothing changed. \(error)")
                return
            }
        }
        if let label {
            do {
                try ExperimentStore.setDisplayLabel(label, experimentName: current)
                let normalized = ExperimentStore.normalizedDisplayLabel(label)
                messages.append(
                    normalized.isEmpty
                        ? "cleared the display label"
                        : "display label set to \"\(normalized)\"")
            } catch {
                refuse(
                    .rename,
                    "Couldn't save the display label — the study folder must be "
                        + "writable. Details: \(error)")
                return
            }
        }
        guard !messages.isEmpty else { return }
        refresh()
        selectedName = current
        note(messages.joined(separator: " — "), severity: .success)
    }

    /// "New from Study" in the Templates tab: mint a design from ANY study.
    ///
    /// Deliberately quieter than `loadSelectedStudyAsTemplate`. In the library
    /// the researcher is looking at the list, so the DEDUP case needs no
    /// sentence — selecting the design that already existed shows the result.
    /// A fresh mint and a divergence both say something, because both changed
    /// the library.
    public func newDesignFromStudy(named name: String) {
        clearFormError(.template)
        do {
            let mint = try StudyTemplateStore.templateFromStudy(experimentName: name)
            refreshTemplates()
            designs.selectedTemplateName = mint.template.name
            for warning in mint.warnings { note(warning, severity: .warning) }
            guard mint.minted else { return }  // silent dedup: the selection IS the answer
            if let parent = mint.divergedFrom {
                note(
                    "created design '\(mint.template.name)' — '\(name)' had "
                        + "diverged from '\(parent)', so this is a new design "
                        + "rather than an edit of that one",
                    severity: .success)
            } else {
                note("created design '\(mint.template.name)'", severity: .success)
            }
        } catch {
            refuse(
                .template,
                "Couldn't load '\(name)' as a design — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    /// Edit a design's description. METADATA, not design: the description is
    /// excluded from the content hash (see `StudyTemplateStore.hash`), so this
    /// cannot make an instance diverge — which is exactly why it is the one
    /// field the read-only library lets you change.
    public func updateTemplateDescription(_ name: String, to description: String) {
        clearFormError(.template)
        do {
            var template = try StudyTemplateStore.load(name: name)
            guard template.templateDescription != description else { return }
            template.templateDescription = description
            try StudyTemplateStore.save(template)
            refreshTemplates()
        } catch {
            refuse(
                .template,
                "Couldn't save the description — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    /// "Edit design…": mint the agentless scratch draft a design is REVISED
    /// through, and select it.
    ///
    /// The whole affordance is this one line of indirection. Revising a design
    /// means editing a manifest, there is exactly one manifest editor (Studies),
    /// and a second editor in the design library would drift from it the first
    /// time a field is added to only one — so the design is cast into an
    /// ordinary draft and the researcher edits THAT. The view navigates to
    /// Studies on a non-nil return; nothing here knows about tabs.
    @discardableResult
    public func editDesign(_ name: String) -> String? {
        clearFormError(.template)
        do {
            let draft = try StudyTemplateStore.mintEditDraft(templateName: name)
            refresh()
            selectedName = draft.name
            note(
                "opened '\(draft.name)' — an ordinary draft of design "
                    + "'\(name)'. Edit it here, then Save back to design to "
                    + "update '\(name)' in place",
                severity: .success)
            return draft.name
        } catch {
            refuse(
                .template,
                "Couldn't open design '\(name)' for editing — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
            return nil
        }
    }

    /// "New Template": the blank version of the same loop — a scratch draft with
    /// no design behind it, which becomes a design via "Save as new design".
    ///
    /// Deliberately the ORDINARY from-scratch creation path (`newStudy`), not a
    /// second one: a design authored from nothing and a study authored from
    /// nothing are the same manifest, and the only difference is what the
    /// researcher does with it at the end.
    ///
    /// The draft is NOT pre-stamped with a lineage line saying "unsaved design".
    /// `templateProvenance` names a design and pins its hash; stamping a design
    /// that does not exist would make `agreement(of:)` report `.designMissing`
    /// and the lineage line say "no longer in the library" — a false statement
    /// about the library, to hint at an intention. The pointer is UI copy
    /// instead.
    @discardableResult
    public func newDesignDraft(context: StudyCreationContext?) -> String? {
        clearFormError(.template)
        newStudy(context: context)
        return selectedName
    }

    /// Overwrite the design the selected draft names, in place.
    ///
    /// The counterpart to `newDesignFromStudy`: both are offered, both are
    /// worded for what they do, and neither is a default. The in-place write
    /// bumps the design's content hash; studies minted from it earlier keep the
    /// hash stamped at their own mint time, so their lineage lines go on saying
    /// what they were actually minted from (see
    /// `StudyTemplateStore.saveStudyBackToDesign`).
    public func saveSelectedStudyBackToDesign() {
        guard let manifest = selected else { return }
        clearFormError(.template)
        if let refusal = designs.saveBackToDesignRefusal(for: manifest) {
            refuse(.template, "Couldn't save back to a design — " + refusal)
            return
        }
        do {
            let update = try StudyTemplateStore.saveStudyBackToDesign(
                experimentName: manifest.name)
            refreshTemplates()
            designs.selectedTemplateName = update.design
            for warning in update.warnings { note(warning, severity: .warning) }
            guard update.changed else {
                note(
                    "design '\(update.design)' already matched "
                        + "'\(manifest.name)' — nothing about the recipe moved",
                    severity: .info)
                return
            }
            note(
                "updated design '\(update.design)' in place "
                    + "(\(update.hashBefore.prefix(12))… → "
                    + "\(update.hashAfter.prefix(12))…) — studies minted from it "
                    + "earlier keep their original lineage stamps",
                severity: .success)
        } catch {
            refuse(
                .template,
                "Couldn't update the design — nothing was written. "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    public func renameTemplate(_ oldName: String, to newName: String) {
        clearFormError(.template)
        do {
            let resolved = try StudyTemplateStore.rename(
                templateName: oldName, to: newName)
            refreshTemplates()
            designs.selectedTemplateName = resolved
            guard resolved != oldName else { return }
            note(
                "renamed template '\(oldName)' → '\(resolved)' — studies already "
                    + "minted from it keep the old name in their lineage stamp",
                severity: .success)
        } catch {
            refuse(
                .template,
                "Couldn't rename the template — nothing changed. "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    public func deleteTemplate(_ name: String) {
        clearFormError(.template)
        do {
            try StudyTemplateStore.delete(name: name)
            if designs.selectedTemplateName == name { designs.selectedTemplateName = nil }
            refreshTemplates()
            note(
                "deleted template '\(name)' — studies minted from it are "
                    + "untouched ordinary drafts",
                severity: .success)
        } catch {
            refuse(.template, "Couldn't delete the template: \(error)")
        }
    }

    /// The display names of the studies minted in the same batch as this one,
    /// excluding itself. Empty when the study has no batch.
    public func batchSiblings(_ manifest: ExperimentManifest) -> [String] {
        guard let batch = manifest.templateProvenance?.batchGroup else { return [] }
        return
            experiments
            .filter {
                $0.templateProvenance?.batchGroup == batch
                    && $0.name != manifest.name
            }
            .map { displayName($0) }
    }

    /// Move the selected DRAFT to a `.trash-<timestamp>` sibling (App gap
    /// A12) — never a destructive delete; frozen/completed studies refuse
    /// inside the store with the immutability line.
    public func deleteSelectedDraft() {
        guard let name = selectedName else { return }
        do {
            let destination = try ExperimentStore.moveDraftToTrash(name: name)
            selectedName = nil
            refresh()
            note(
                "moved draft '\(name)' to "
                    + "experiments/\(destination.deletingLastPathComponent().lastPathComponent)/"
                    + "\(destination.lastPathComponent) — recover it from there if needed",
                severity: .success)
        } catch {
            note(
                "Couldn't move the draft to trash — nothing was deleted; "
                    + "frozen studies can never be deleted, and the "
                    + "experiments/ folder must be writable. Details: \(error)",
                severity: .error)
        }
    }

    public func duplicateSelected() {
        guard let name = selectedName else { return }
        var candidate = "\(name)-2"
        var counter = 2
        while (try? ExperimentStore.load(name: candidate)) != nil {
            counter += 1
            candidate = "\(name)-\(counter)"
        }
        do {
            let copy = try ExperimentStore.duplicate(name: name, as: candidate)
            refresh()
            selectedName = copy.name
            note("created draft '\(copy.name)'", severity: .success)
        } catch {
            note(
                "Couldn't duplicate the study — nothing was created; check "
                    + "the experiments/ folder is writable. Details: \(error)",
                severity: .error)
        }
    }

    /// Why the selected study cannot be deleted, or nil.
    ///
    /// Draft-only, and that is the STORE's rule, not a UI choice:
    /// `ExperimentStore.moveDraftToTrash` refuses anything frozen or complete
    /// with the immutability line, because a frozen manifest is what every run
    /// directory's stamp points at. Surfaced as a reason rather than a hidden
    /// button so the answer to "why can't I delete this?" is on the control.
    public var deleteSelectedStudyRefusal: String? {
        guard let manifest = selected else { return "select a study first" }
        guard manifest.status == .draft else {
            return "'\(manifest.name)' is \(manifest.status.rawValue) — frozen "
                + "and completed studies are immutable and cannot be deleted "
                + "(their runs stamp them); duplicate as a draft to iterate"
        }
        return nil
    }

}
