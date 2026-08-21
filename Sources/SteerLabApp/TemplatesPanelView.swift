import ExperimentKit
import SwiftUI

/// The design library: every saved study DESIGN, what each one measures, and
/// the two ways in and out (save a study as a design; cast a design into new
/// studies).
///
/// Deliberately WITHOUT a manifest editor (settled scoping decision,
/// 2026-08-06). Revising a design is a round trip — "Edit design…" mints an
/// agentless scratch draft, the Studies editor edits it, "Save back to design"
/// writes it here — and a second editor in this tab would be a parallel
/// implementation of the Studies one: the first field added to only one of them
/// is the moment the two start telling the researcher different things about
/// the same manifest. So this tab reads designs and edits exactly one thing,
/// the description, which is metadata excluded from the design's content hash.
///
/// What the three buttons here are FOR (2026-08-06 round-trip pass): the
/// researcher asked why a design cannot be edited or written from scratch. The
/// answer is that it can — through the one editor. "Edit design…" and "New
/// Template" are the two ends of that loop made first-class, so the round trip
/// is a thing you press rather than a procedure you have to know.
///
/// Every rule it renders lives in `ExperimentPanel` / `StudyDesignSummary`
/// (ExperimentKit, unit-tested). This file decides nothing.
struct TemplatesPanelView: View {
    @Bindable var service: ChatService
    /// Section navigation, injected by the workbench shell — Instantiate hands
    /// the design to the Studies tab and goes there.
    var navigate: (WorkbenchSection) -> Void = { _ in }

    @State private var renamingTemplate: String?
    @State private var templateRenameText = ""
    @State private var confirmDeleteTemplate = false
    /// The description being edited, and the design it belongs to — so
    /// switching designs never carries one design's text onto another.
    @State private var descriptionDraft = ""
    @State private var descriptionOwner: String?

    private var panel: ExperimentPanel { service.experiments }

    /// Reloads the description editor when the selection changes, so one
    /// design's unsaved text never lands on another.
    private func syncDescriptionDraft() {
        let selected = panel.selectedTemplateName
        guard descriptionOwner != selected else { return }
        descriptionOwner = selected
        descriptionDraft = panel.selectedTemplate?.templateDescription ?? ""
    }

    var body: some View {
        @Bindable var panel = service.experiments
        Form {
            newDesignSection(panel: panel)
            librarySection(panel: panel)
            if let template = panel.selectedTemplate {
                metadataSection(template, panel: panel)
                designSummarySection(template, panel: panel)
                actionsSection(template, panel: panel)
            }
            if let status = panel.status {
                Section("Status") {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .onAppear {
            panel.refresh()
            syncDescriptionDraft()
        }
        .onChange(of: panel.selectedTemplateName) { syncDescriptionDraft() }
        .alert(
            "Rename design",
            isPresented: Binding(
                get: { renamingTemplate != nil },
                set: { if !$0 { renamingTemplate = nil } })
        ) {
            TextField("design name", text: $templateRenameText)
            Button("Cancel", role: .cancel) { renamingTemplate = nil }
            Button("Rename") {
                if let old = renamingTemplate {
                    panel.renameTemplate(old, to: templateRenameText)
                }
                renamingTemplate = nil
            }
        } message: {
            Text("Studies already minted from this design keep the OLD name "
                + "in their lineage stamp — provenance records what was true "
                + "when it was written.")
        }
    }

    // MARK: New design

    /// The two ways a design gets into the library: from a study that exists,
    /// or from nothing.
    ///
    /// The study picker spans EVERY study at any status: the replication worth
    /// repeating is usually one that already ran, and a design carries no
    /// lifecycle stamps, so a frozen study saves exactly as a draft does.
    ///
    /// "New Template" is the blank end of the same loop. It creates an ordinary
    /// scratch draft and hands it to the Studies editor — there is no
    /// design-shaped blank form, because a design IS a manifest and the manifest
    /// editor is in Studies.
    @ViewBuilder
    private func newDesignSection(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        Section("New Design") {
            HStack(spacing: 8) {
                Button {
                    guard panel.newDesignDraft() != nil else { return }
                    navigate(.studies)
                } label: {
                    Label("New Template", systemImage: "plus.square.on.square")
                }
                .help(Self.newTemplateHelp)
            }
            Text(Self.fromScratchPointer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Divider()
            Picker("Study", selection: $panel.templateSourceStudyName) {
                Text("select…").tag(String?.none)
                ForEach(panel.experiments, id: \.name) { manifest in
                    Text(studyPickerLabel(manifest, panel: panel))
                        .tag(String?.some(manifest.name))
                }
            }
            HStack(spacing: 8) {
                Button {
                    guard let name = panel.templateSourceStudyName else { return }
                    panel.newDesignFromStudy(named: name)
                } label: {
                    Label("New from Study", systemImage: "square.on.square")
                }
                .disabled(panel.templateSourceStudyName == nil)
                .help(Self.newFromStudyHelp)
            }
            if let refusal = panel.formErrors[.template] {
                Label(refusal, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            Text(Self.designVsDuplicate)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func studyPickerLabel(
        _ manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> String {
        // Display label leads, canonical name stays visible: run directories
        // and CLI arguments speak the canonical one.
        let display = panel.displayName(manifest)
        return display == manifest.name
            ? "\(manifest.name)  [\(manifest.status.rawValue)]"
            : "\(display)  ·  \(manifest.name)  [\(manifest.status.rawValue)]"
    }

    // MARK: The library

    @ViewBuilder
    private func librarySection(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        Section("Designs") {
            if panel.templates.isEmpty {
                Text("No designs yet. Save a study you intend to repeat — the "
                    + "design keeps its task file and pins, instruments, "
                    + "judges and sampling policy, and holds no agents.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Picker("Design", selection: $panel.selectedTemplateName) {
                    Text("select…").tag(String?.none)
                    ForEach(panel.templates, id: \.name) { template in
                        Text(templateRowLabel(template))
                            .tag(String?.some(template.name))
                    }
                }
            }
        }
    }

    private func templateRowLabel(_ template: StudyTemplate) -> String {
        "\(template.name)  ·  \(template.intent.displayName)  ·  "
            + template.study.modelID
    }

    // MARK: Metadata (the one editable field)

    @ViewBuilder
    private func metadataSection(
        _ template: StudyTemplate, panel: ExperimentPanel
    ) -> some View {
        Section(template.name) {
            TextField(
                "what this design is for", text: $descriptionDraft,
                axis: .vertical
            )
            .lineLimit(1 ... 3)
            .onSubmit {
                panel.updateTemplateDescription(template.name, to: descriptionDraft)
            }
            .help(
                "a note for the researcher. Excluded from the design's content "
                    + "hash, so editing it can never make a minted study "
                    + "diverge — which is why it is the one field this library "
                    + "lets you change")
            Text("press return to save")
                .font(.caption2)
                .foregroundStyle(.secondary)
            LabeledContent("Created", value: shortDate(template.createdAt))
                .font(.caption)
            if let parent = template.parentTemplate {
                LabeledContent("Diverged from", value: parent)
                    .font(.caption)
                    .help(
                        "this design was saved from a study that had changed "
                            + "since it was instantiated from that design")
            }
        }
    }

    private func shortDate(_ iso: String) -> String {
        guard !iso.isEmpty else { return "—" }
        return String(iso.prefix(19)).replacingOccurrences(of: "T", with: " ")
    }

    // MARK: The read-only design summary

    /// What this design MEASURES, in the units a result is read in. Rendered
    /// from `StudyDesignSummary` verbatim — no formatting happens here.
    @ViewBuilder
    private func designSummarySection(
        _ template: StudyTemplate, panel: ExperimentPanel
    ) -> some View {
        Section("Design") {
            ForEach(panel.designSummary(template)) { row in
                LabeledContent(row.label) {
                    Text(row.value)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
            }
            Text("read-only here. To revise it: Edit design… mints a scratch "
                + "draft, you edit that draft in the Studies editor, and Save "
                + "back to design writes it onto this design in place.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Actions

    @ViewBuilder
    private func actionsSection(
        _ template: StudyTemplate, panel: ExperimentPanel
    ) -> some View {
        Section {
            HStack(spacing: 8) {
                Button("Instantiate…") {
                    // Cross-section handoff: Studies opens the new-studies
                    // table on this design (consumed on appear as well as on
                    // change — Studies is not on screen when this is set).
                    panel.templateInstantiationInvitation =
                        TemplateInstantiationInvitation(design: template.name)
                    navigate(.studies)
                }
                .help(
                    "opens the Studies tab's new-studies table on this design: "
                        + "pick the cast, see what the batch will cost, then "
                        + "mint one ordinary draft study per casting")
                Button("Edit design…") {
                    // The round trip's first leg. The draft is ORDINARY — full
                    // editor, no special mode — and the second leg is the
                    // Studies tab's "Save back to design".
                    guard panel.editDesign(template.name) != nil else { return }
                    navigate(.studies)
                }
                .help(Self.editDesignHelp)
                Button("Rename…") {
                    templateRenameText = template.name
                    renamingTemplate = template.name
                }
                Button("Delete", role: .destructive) { confirmDeleteTemplate = true }
                    .confirmationDialog(
                        "Delete design '\(template.name)'?",
                        isPresented: $confirmDeleteTemplate,
                        titleVisibility: .visible
                    ) {
                        Button("Delete '\(template.name)'", role: .destructive) {
                            panel.deleteTemplate(template.name)
                        }
                    } message: {
                        Text("Removes templates/\(template.name)/. Studies "
                            + "already minted from it are ordinary drafts and "
                            + "are untouched — they keep the design's name in "
                            + "their lineage stamp.")
                    }
            }
        }
    }

    // MARK: Copy

    private static let newFromStudyHelp =
        "strips the chosen study to its design: every generation and "
        + "measurement setting, no agents. Offered at any status — a design "
        + "carries no lifecycle stamps, so a frozen study saves exactly as a "
        + "draft does. An unchanged instance of an existing design selects "
        + "that design rather than minting a near-duplicate."

    private static let editDesignHelp =
        "mints an agentless scratch draft of this design (named "
        + "<design>-edit) and opens it in the Studies editor — the ONE manifest "
        + "editor. Change anything there, then use Save back to design to "
        + "update this design in place. The draft is an ordinary study: it can "
        + "be kept, run, frozen or deleted like any other."

    private static let newTemplateHelp =
        "creates a blank scratch draft and opens it in the Studies editor. "
        + "Author the design there, then use Save as new design to put it in "
        + "this library."

    private static let fromScratchPointer =
        "There is no blank design form: a design IS a study manifest, so a new "
        + "one is authored in the Studies editor and saved back here. New "
        + "Template starts that draft; Save as new design (in Studies) or New "
        + "from Study below completes the loop."

    private static let designVsDuplicate =
        "Saving a study as a design strips its agents and re-derives the "
        + "derived pins (the instrument scope is re-pinned against the task "
        + "file at every instantiation). Duplicate as Draft, in Studies, does "
        + "the opposite: it copies everything, agents included."
}
