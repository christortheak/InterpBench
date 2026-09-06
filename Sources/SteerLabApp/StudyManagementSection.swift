import ExperimentKit
import SteeringKit
import SwiftUI

/// Study selection and lifecycle controls, including management dialogs and cross-tab invitations.
struct StudyManagementSection: View {
    @Bindable var panel: ExperimentPanel
    var openTemplates: () -> Void
    /// A12: delete-draft confirmation (move-to-trash, never destructive).
    @State private var confirmDeleteDraft = false
    /// Runs stamped with the study being deleted, read at click time (never
    /// per frame — it scans runs/) so the confirmation can say what is at stake.
    @State private var deleteDraftRunCount = 0
    /// The one Rename affordance, offered for every study whatever its
    /// status: a draft renames for real, a frozen/complete study takes a
    /// display label (see `ExperimentStore.rename` for why the two differ).
    @State private var renameSheet: RenameStudySheet?
    /// The template-instantiation sheet (the cell table) and the two modest
    /// library affordances beside it. A template is a draft of the library —
    /// never frozen, nothing stamps its name into evidence — so rename and
    /// delete are unconditional, unlike the study equivalents.
    /// The new-studies sheet (the cell table). The design LIBRARY — rename,
    /// delete, the design summary — lives in the Templates tab; Studies only
    /// casts designs into studies.
    @State private var templateSheet: TemplateInstantiationRequest?

    /// Study list row label; manifests with a declared sweep carry a small
    /// "optimization" badge (Home's studyRowLabel shows the same).
    /// Labels for the duplicate FAMILIES in the current list, keyed by study
    /// name. `x`, `x-2` and `x-2-2` are three distinct names, so nothing else
    /// treats them as ambiguous — yet they are exactly the set that cannot be
    /// told apart, since duplicating-to-iterate is how the lifecycle says to
    /// change a study.
    private var duplicateFamilyLabels: [String: ArtifactDisambiguation.Label] {
        var out: [String: ArtifactDisambiguation.Label] = [:]
        for (_, labels) in ArtifactDisambiguation.familyLabels(
            panel.management.experiments)
        {
            for label in labels { out[label.id] = label }
        }
        return out
    }

    private func studyPickerLabel(
        _ manifest: ExperimentManifest,
        families: [String: ArtifactDisambiguation.Label] = [:]
    ) -> String {
        // A display label leads, but never REPLACES the canonical name: run
        // directories, config.json stamps and CLI arguments all speak the
        // canonical one, so it has to stay correlatable here.
        let display = panel.management.displayName(manifest)
        var label =
            display == manifest.name
            ? "\(manifest.name)  [\(manifest.status.rawValue)]"
            : "\(display)  ·  \(manifest.name)  [\(manifest.status.rawValue)]"
        if manifest.sweep != nil {
            label += "  · optimization"
        }
        // Lineage badge instead of a filter control: a batch of six castings
        // reads as six adjacent rows sharing one template name, which is the
        // grouping a researcher actually wants and costs no new UI.
        if let template = manifest.templateProvenance?.template {
            label += "  · from \(template)"
        }
        // What distinguishes this one from its duplicates — or that nothing
        // does, which is itself the answer worth having.
        if let distinguisher = families[manifest.name]?.distinguisher {
            label += "  · \(distinguisher)"
        }
        return label
    }

    var body: some View {
        @Bindable var management = panel.management
        @Bindable var draft = management.draft
        Section {
            Picker("Draft", selection: $management.selectedName) {
                Text("select…").tag(String?.none)
                let families = duplicateFamilyLabels
                ForEach(panel.management.experiments, id: \.name) { manifest in
                    // Optimization badge (mirrors Home's studyRowLabel):
                    // Studies reads as inventory/provenance; optimization
                    // authoring lives in Agents → Optimizations.
                    Text(studyPickerLabel(manifest, families: families))
                        .tag(String?.some(manifest.name))
                }
            }
            .help(
                "studies are versioned manifests in experiments/<name>/ — "
                    + "'optimization' marks a declared sweep (authored in "
                    + "Agents → Optimizations)")

            VStack(alignment: .leading, spacing: 6) {
                // The new-study flow starts from a DESIGN choice
                // (2026-08-06). "From scratch" is the blank interface
                // exactly as before; a saved design opens the
                // new-studies table prefilled, so the only decision left
                // is the casting.
                newStudyDesignPicker(panel: panel)
                // One click in, name it after. Naming a study before it
                // exists is a decision the researcher cannot yet make,
                // and a draft is renamable for as long as it stays a
                // draft — so the name is no longer a gate on starting.
                HStack(spacing: 8) {
                    Button("New Study") { startNewStudy(panel: panel) }
                        .help(
                            panel.management.designs.newStudyDesign.designName == nil
                                ? "creates a draft pinned to the currently selected "
                                    + "model under a placeholder name, and opens "
                                    + "Rename so you can name it now"
                                : "opens the new-studies table on this design — one "
                                    + "ordinary draft per casting")
                    if let manifest = panel.management.selected {
                        Button {
                            openRename(manifest)
                        } label: {
                            Label("Rename…", systemImage: "pencil")
                        }
                        .help(
                            manifest.status == .draft
                                ? "change this draft's name, its display label, or both"
                                : "frozen and completed studies keep their name for "
                                    + "run provenance — Rename sets a display label")
                        Button("Duplicate as Draft") { management.duplicateSelected() }
                            .help(StudyControlCopy.duplicateHelp)
                        Button("Delete…", role: .destructive) {
                            deleteDraftRunCount = ExperimentStore.runsStamped(
                                experimentName: manifest.name)
                            confirmDeleteDraft = true
                        }
                        .disabled(panel.management.deleteSelectedStudyRefusal != nil)
                        .help(panel.management.deleteSelectedStudyRefusal ?? StudyControlCopy.deleteStudyHelp)
                        .confirmationDialog(
                            "Delete draft '\(manifest.name)'?",
                            isPresented: $confirmDeleteDraft,
                            titleVisibility: .visible
                        ) {
                            Button(
                                "Move '\(manifest.name)' to trash",
                                role: .destructive
                            ) {
                                management.deleteSelectedDraft()
                            }
                        } message: {
                            Text(deleteDraftMessage(manifest))
                        }
                    }
                }
                if let refusal = panel.draft.formErrors[.rename] {
                    Label(refusal, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                // The naming/model detail the one-click button skips.
                // Reachable, never in the way (capability preserved).
                DisclosureGroup("Advanced") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(
                            "Create a study with a specific name and a "
                                + "pinned model revision up front. The normal "
                                + "path is New Study, then Rename — a draft "
                                + "renames freely for as long as it stays a "
                                + "draft, and the revision auto-pins from the "
                                + "local HF cache at the first "
                                + "extract/validate. Use this when you already "
                                + "know both, e.g. reproducing a study on a "
                                + "named model snapshot."
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        TextField("new study name", text: $draft.newName)
                            .help("creates a draft pinned to the currently selected model")
                        TextField(
                            "question or purpose", text: $draft.newDescription,
                            axis: .vertical
                        )
                        .lineLimit(1...3)
                        .help("short domain-neutral purpose for the draft protocol")
                        TextField(
                            "model revision (optional commit hash)",
                            text: $draft.newRevision
                        )
                        .font(.caption.monospaced())
                        .help(
                            "pins the exact HF snapshot commit up front — a frozen "
                                + "study must never silently run another model version")
                        Text(
                            "empty = auto-pin from the local HF cache at the first "
                                + "extract/validate"
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        Button("Create Draft") { panel.management.create(context: panel.studyCreationContext) }
                            .disabled(panel.draft.newName.isEmpty)
                    }
                }
            }
            // New Study mints a placeholder name and asks for the rename
            // immediately — the one-click flow is only an improvement if
            // naming follows it.
            .onChange(of: panel.management.renameInvitation) {
                consumeRenameInvitation(panel: panel)
            }
        } header: {
            // A15: the persistent-notices bell lives in the section
            // header so overnight failures are one click away.
            HStack {
                Text("Study")
                Spacer()
                NoticesBellButton()
            }
        }
        .sheet(item: $renameSheet) { sheet in
            RenameStudyWindow(sheet: sheet, panel: panel)
        }
        .sheet(item: $templateSheet) { request in
            TemplateInstantiationSheet(request: request, panel: panel)
        }
        // The Templates tab's Instantiate opens the new-studies flow PRE-LOADED
        // with that design. Consumed on CHANGE (the same-tab paths) and on
        // APPEAR (the cross-section handoff — this view is not on screen when
        // Templates sets it).
        .onChange(of: panel.management.designs.templateInstantiationInvitation) {
            consumeInstantiationInvitation(panel: panel)
        }
        .onAppear {
            panel.refresh()
            consumeInstantiationInvitation(panel: panel)
            // Same one-shot pattern, same reason as the instantiation
            // invitation: Templates' "New Template" creates the draft and
            // navigates here, so this view is not on screen when the rename
            // invitation is set and the onChange above never fires.
            consumeRenameInvitation(panel: panel)
        }
    }

    /// The first control in the new-study flow: what this study starts FROM.
    ///
    /// The design LIBRARY is the Templates tab; this is only the choice, so
    /// that "new study" is one flow with two beginnings rather than two flows
    /// the researcher has to know to pick between.
    @ViewBuilder
    private func newStudyDesignPicker(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        @Bindable var designs = panel.management.designs
        Picker("Start from", selection: $designs.newStudyDesign) {
            ForEach(StudyDesignChoice.choices(designs: panel.management.designs.templates)) { choice in
                Text(choice.label).tag(choice)
            }
        }
        .help(
            "From scratch opens the blank draft interface. A saved design "
                + "opens the new-studies table, prefilled with that design's "
                + "task file and pins, instruments, sampling policy and judges "
                + "— so the only thing left to decide is the casting.")
        if panel.management.designs.templates.isEmpty {
            HStack(spacing: 6) {
                Text("No saved designs yet.")
                Button("Open Templates") { openTemplates() }
                    .buttonStyle(.link)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    /// "New Study" does one of the two things the design picker selected.
    private func startNewStudy(panel: ExperimentPanel) {
        if let design = panel.management.designs.newStudyDesign.designName {
            openInstantiation(design, panel: panel)
        } else {
            panel.management.newStudy(context: panel.studyCreationContext)
        }
    }

    /// Opens the new-studies table on a design, mirroring the panel's current
    /// Remote options so the totals line counts the jobs this batch will
    /// really create.
    private func openInstantiation(
        _ name: String, panel: ExperimentPanel, permuting: [SeatOccupant] = []
    ) {
        panel.clearFormError(.template)
        templateSheet = TemplateInstantiationRequest(
            templateName: name,
            shardsPerStudy: max(1, panel.submission.remoteParallelJobs),
            jobNoun: panel.submission.remoteExecutor == "slurm" ? "Slurm jobs" : "jobs",
            canSubmit: panel.canSubmitBundles,
            permuting: permuting)
    }

    /// One-shot handoff for "this draft was just created — name it now".
    /// Consumed on change (the in-tab New Study path) and on appear (Templates'
    /// New Template, which creates the draft in another section).
    private func consumeRenameInvitation(panel: ExperimentPanel) {
        guard let invited = panel.management.renameInvitation,
            let manifest = panel.management.experiments.first(where: { $0.name == invited })
        else { return }
        panel.management.renameInvitation = nil
        openRename(manifest)
    }

    /// One-shot handoff from the Templates tab (and from any in-tab path that
    /// resolves a design): open the flow on it, then clear the flag.
    private func consumeInstantiationInvitation(panel: ExperimentPanel) {
        guard let invited = panel.management.designs.templateInstantiationInvitation else { return }
        panel.management.designs.templateInstantiationInvitation = nil
        panel.management.designs.newStudyDesign = .design(invited.design)
        openInstantiation(
            invited.design, panel: panel, permuting: invited.permuting)
    }

    private func deleteDraftMessage(_ manifest: ExperimentManifest) -> String {
        var message =
            "Moves experiments/\(manifest.name)/ (the draft manifest and its "
            + "pinned snapshots) to a .trash-<timestamp> sibling inside "
            + "experiments/. Nothing is destructively removed — recover it "
            + "from there if needed. Run artifacts under runs/ are untouched."
        if deleteDraftRunCount > 0 {
            // A draft CAN have runs (only freeze is one-way, not running), and
            // those runs stamp this study's name — deleting the manifest
            // leaves them unresolvable from the app.
            message +=
                " \(deleteDraftRunCount) run(s) already stamp this "
                + "draft's name; they stay in runs/ but will no longer resolve "
                + "back to a study here."
        }
        return message
    }

    /// Opens Rename on `manifest`, selecting it first so the panel action
    /// (which always targets the selection) cannot act on another study.
    private func openRename(_ manifest: ExperimentManifest) {
        panel.clearFormError(.rename)
        panel.management.selectedName = manifest.name
        renameSheet = RenameStudySheet(
            name: manifest.name,
            status: manifest.status,
            label: panel.management.displayLabels[manifest.name] ?? "",
            runsStamped: ExperimentStore.runsStamped(experimentName: manifest.name))
    }
}
