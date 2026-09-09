import ExperimentKit
import SwiftUI

/// The design library: every saved study DESIGN, what each one measures, and
/// the two ways in and out (save a study as a design; cast a design into new
/// studies).
///
/// Deliberately WITHOUT a manifest editor (settled scoping decision,
/// 2026-08-06). Revising a design is a round trip — "Edit template…" creates an
/// agentless scratch draft, the Studies editor edits it, "Save back to template"
/// writes it here — and a second editor in this tab would be a parallel
/// implementation of the Studies one: the first field added to only one of them
/// is the moment the two start telling the researcher different things about
/// the same manifest. So this tab reads designs and edits exactly one thing,
/// the description, which is metadata excluded from the design's content hash.
///
/// What the three buttons here are FOR (2026-08-06 round-trip pass): the
/// researcher asked why a design cannot be edited or written from scratch. The
/// answer is that it can — through the one editor. "Edit template…" and "New
/// Blank Template" are the two ends of that loop made first-class, so the round
/// trip is a thing you press rather than a procedure you have to know.
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
    @State private var descriptionOwner: DescriptionContext?
    @State private var descriptionReview: StudyDesignSnapshot?
    @State private var descriptionMessage: String?
    private struct DescriptionContext: Equatable {
        let workspaceRoot: URL
        let name: String?
    }

    private var panel: ExperimentPanel { service.experiments }

    /// Reloads the description editor when the selection changes, so one
    /// design's unsaved text never lands on another.
    private func syncDescriptionDraft(force: Bool = false) {
        let context = DescriptionContext(workspaceRoot: ExperimentStore.workspaceRoot.standardizedFileURL,
            name: panel.management.designs.selectedTemplateName)
        guard force || descriptionOwner != context else { return }
        descriptionOwner = context
        descriptionReview = nil
        descriptionDraft = ""
        descriptionMessage = nil
        guard let name = context.name else { return }
        do {
            let reviewed = try StudyDesignSnapshot(workspaceRoot: context.workspaceRoot, name: name)
            descriptionReview = reviewed
            descriptionDraft = reviewed.template.templateDescription
        } catch { descriptionMessage = "Couldn't read the template: \(error.localizedDescription)" }
    }

    var body: some View {
        @Bindable var panel = service.experiments
        Form {
            newDesignSection(panel: panel)
            librarySection(panel: panel)
            if let template = panel.management.designs.selectedTemplate {
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
        .onChange(of: panel.management.designs.selectedTemplateName) { syncDescriptionDraft() }
        .onChange(of: ExperimentStore.workspaceRoot.path) { syncDescriptionDraft() }
        .alert(
            "Rename template",
            isPresented: Binding(
                get: { renamingTemplate != nil },
                set: { if !$0 { renamingTemplate = nil } })
        ) {
            TextField("template name", text: $templateRenameText)
            Button("Cancel", role: .cancel) { renamingTemplate = nil }
            Button("Rename") {
                if let old = renamingTemplate {
                    panel.management.renameTemplate(old, to: templateRenameText)
                }
                renamingTemplate = nil
            }
            // An empty or unchanged name was accepted and then refused two
            // sections away, after the alert had closed (UI audit 2026-09-06).
            .disabled(
                templateRenameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || templateRenameText == renamingTemplate)
        } message: {
            Text("Studies already created from this template keep the OLD name "
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
    /// "New Blank Template" is the blank end of the same loop. It creates an
    /// ordinary scratch draft and hands it to the Studies editor — there is no
    /// design-shaped blank form, because a design IS a manifest and the manifest
    /// editor is in Studies.
    @ViewBuilder
    private func newDesignSection(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        @Bindable var designs = panel.management.designs
        Section("New Template") {
            HStack(spacing: 8) {
                Button {
                    guard panel.management.newDesignDraft(context: panel.studyCreationContext) != nil else { return }
                    navigate(.studies)
                } label: {
                    Label("New Blank Template", systemImage: "plus.square.on.square")
                }
                .help(Self.newBlankDesignHelp)
            }
            Text(Self.fromScratchPointer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Divider()
            Picker("Study", selection: $designs.templateSourceStudyName) {
                Text("select…").tag(String?.none)
                ForEach(panel.management.experiments, id: \.name) { manifest in
                    Text(studyPickerLabel(manifest, panel: panel))
                        .tag(String?.some(manifest.name))
                }
            }
            .help(
                "the study whose settings become the template — every study in "
                    + "this workspace at any status, because a template carries "
                    + "no lifecycle stamps")
            HStack(spacing: 8) {
                Button {
                    guard let name = panel.management.designs.templateSourceStudyName else { return }
                    do {
                        let source = try panel.management.reviewDesignSource(named: name)
                        panel.management.newDesignFromStudy(reviewedSource: source)
                    } catch { panel.draft.formErrors[.template] = error.localizedDescription }
                } label: {
                    Label("New from Study", systemImage: "square.on.square")
                }
                .disabled(panel.management.designs.templateSourceStudyName == nil)
                .help(Self.newFromStudyHelp)
            }
            if let refusal = panel.draft.formErrors[.template] {
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
        let display = panel.management.displayName(manifest)
        return display == manifest.name
            ? "\(manifest.name)  [\(manifest.status.rawValue)]"
            : "\(display)  ·  \(manifest.name)  [\(manifest.status.rawValue)]"
    }

    // MARK: The library

    @ViewBuilder
    private func librarySection(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        @Bindable var designs = panel.management.designs
        Section("Templates") {
            if panel.management.designs.templates.isEmpty {
                Text("No templates yet. Save a study you intend to repeat — the "
                    + "template keeps its task file and pins, instruments, "
                    + "judges and sampling policy, and holds no agents.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Picker("Template", selection: $designs.selectedTemplateName) {
                    Text("select…").tag(String?.none)
                    ForEach(panel.management.designs.templates, id: \.name) { template in
                        Text(templateRowLabel(template))
                            .tag(String?.some(template.name))
                    }
                }
                .help(selectedDesignHelp)
            }
        }
    }

    private func templateRowLabel(_ template: StudyTemplate) -> String {
        "\(template.name)  ·  \(template.intent.displayName)  ·  "
            + template.study.modelID
    }

    /// Picker rows truncate in a 560 pt column, so the hover text carries the
    /// selected row in full plus what the three parts mean (UI audit
    /// 2026-09-06).
    private var selectedDesignHelp: String {
        let format =
            "each row reads name · what kind of study it is · the model it "
            + "pins"
        guard let selected = panel.management.designs.selectedTemplate else {
            return format
        }
        return "\(templateRowLabel(selected)) — \(format)"
    }

    // MARK: Metadata (the one editable field)

    @ViewBuilder
    private func metadataSection(
        _ template: StudyTemplate, panel: ExperimentPanel
    ) -> some View {
        Section(template.name) {
            if let descriptionMessage {
                Label(descriptionMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            TextField(
                "what this template is for", text: $descriptionDraft,
                axis: .vertical
            )
            .lineLimit(1 ... 3)
            .onSubmit { saveDescription(template, panel: panel) }
            .help(
                "a note for the researcher — excluded from the template's content "
                    + "hash, so editing it can never make a created study "
                    + "diverge, which is why it is the one field this library "
                    + "lets you change")
            // Return saves, but a vertical-axis field reads as a newline
            // field, and the discard button used to sit ABOVE the thing it
            // discards (UI audit 2026-09-06).
            HStack(spacing: 8) {
                Button("Save Description") { saveDescription(template, panel: panel) }
                    .disabled(!descriptionHasChange)
                    .help(
                        descriptionHasChange
                            ? "writes the description onto this template — the "
                                + "same thing pressing Return in the field does"
                            : "nothing to save — the field matches the saved "
                                + "description")
                Button("Discard description edits and reload") { syncDescriptionDraft(force: true) }
                    .disabled(!descriptionHasChange)
                    .help(
                        "re-reads the template from disk and replaces the field "
                            + "above; no template file is changed")
                Text("or press return in the field")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Created", value: shortDate(template.createdAt))
                .font(.caption)
                .help("when this template was first written into the library")
            if let parent = template.parentTemplate {
                LabeledContent("Diverged from", value: parent)
                    .font(.caption)
                    .help(
                        "this template was saved from a study that had changed "
                            + "since it was instantiated from that template")
            }
        }
    }

    /// Whether the field differs from the design as last read.
    private var descriptionHasChange: Bool {
        guard let reviewed = descriptionReview else { return false }
        return descriptionDraft != reviewed.template.templateDescription
    }

    /// The one write this library performs. Shared by Return and the button so
    /// the two can never diverge.
    private func saveDescription(_ template: StudyTemplate, panel: ExperimentPanel) {
        guard let reviewed = descriptionReview, reviewed.template.name == template.name else {
            descriptionMessage = "Reload this template, then save the description again."
            return
        }
        if let saved = panel.management.updateTemplateDescription(reviewed: reviewed, to: descriptionDraft) {
            descriptionReview = saved
            descriptionMessage = nil
        } else {
            descriptionMessage =
                "The description was not saved. Your edits are kept here; "
                + "reload the saved template and try again."
        }
    }

    /// Localized, in this machine's time zone — the ISO string was sliced by
    /// hand and shown as if it were local time (UI audit 2026-09-06).
    private func shortDate(_ iso: String) -> String {
        guard !iso.isEmpty else { return "—" }
        guard let date = Self.parseTimestamp(iso) else {
            return String(iso.prefix(19)).replacingOccurrences(of: "T", with: " ")
        }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    /// Both spellings the stores write (with and without fractional seconds).
    private static func parseTimestamp(_ iso: String) -> Date? {
        let plain = ISO8601DateFormatter()
        if let date = plain.date(from: iso) { return date }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: iso)
    }

    // MARK: The read-only design summary

    /// What this design MEASURES, in the units a result is read in. Rendered
    /// from `StudyDesignSummary` verbatim — no formatting happens here.
    @ViewBuilder
    private func designSummarySection(
        _ template: StudyTemplate, panel: ExperimentPanel
    ) -> some View {
        Section("Template") {
            ForEach(panel.management.designs.designSummary(template)) { row in
                LabeledContent(row.label) {
                    Text(row.value)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
            }
            Text("read-only here. To revise it: Edit template… creates a scratch "
                + "draft, you edit that draft in the Studies editor, and Save "
                + "back to template writes it onto this template in place.")
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
                    panel.management.designs.templateInstantiationInvitation =
                        TemplateInstantiationInvitation(design: template.name)
                    navigate(.studies)
                }
                .help(
                    "opens the Studies tab's new-studies table on this template: "
                        + "pick the cast, see what the batch will cost, then "
                        + "create one ordinary draft study per casting")
                Button("Edit template…") {
                    // The round trip's first leg. The draft is ORDINARY — full
                    // editor, no special mode — and the second leg is the
                    // Studies tab's "Save back to template".
                    guard panel.management.editDesign(template.name) != nil else { return }
                    navigate(.studies)
                }
                .help(Self.editDesignHelp)
                Button("Rename…") {
                    templateRenameText = template.name
                    renamingTemplate = template.name
                }
                .help(
                    "renames templates/<name>/ — studies already created from "
                        + "this template keep the OLD name in their lineage "
                        + "stamp, because provenance records what was true "
                        + "when it was written")
                Button("Delete", role: .destructive) { confirmDeleteTemplate = true }
                    .help(
                        "removes this template from the library after a "
                            + "confirmation — studies created from it are "
                            + "ordinary drafts and are untouched")
                    .confirmationDialog(
                        "Delete template '\(template.name)'?",
                        isPresented: $confirmDeleteTemplate,
                        titleVisibility: .visible
                    ) {
                        Button("Delete '\(template.name)'", role: .destructive) {
                            panel.management.deleteTemplate(template.name)
                        }
                    } message: {
                        Text("Removes templates/\(template.name)/. Studies "
                            + "already created from it are ordinary drafts and "
                            + "are untouched — they keep the template's name in "
                            + "their lineage stamp.")
                    }
            }
        }
    }

    // MARK: Copy

    private static let newFromStudyHelp =
        "strips the chosen study to its template: every generation and "
        + "measurement setting, no agents. Offered at any status — a template "
        + "carries no lifecycle stamps, so a frozen study saves exactly as a "
        + "draft does. An unchanged instance of an existing template selects "
        + "that template rather than creating a near-duplicate."

    private static let editDesignHelp =
        "creates an agentless scratch draft of this template (named "
        + "<template>-edit) and opens it in the Studies editor — the ONE manifest "
        + "editor. Change anything there, then use Save back to template to "
        + "update this template in place. The draft is an ordinary study: it can "
        + "be kept, run, frozen or deleted like any other."

    private static let newBlankDesignHelp =
        "creates a blank scratch draft and opens it in the Studies editor. "
        + "Author the template there, then use Save as new template to put it in "
        + "this library"

    private static let fromScratchPointer =
        "There is no blank template form: a template IS a study manifest, so a new "
        + "one is authored in the Studies editor and saved back here. New "
        + "Blank Template starts that draft; Save as new template (in Studies) or "
        + "New from Study below completes the loop."

    private static let designVsDuplicate =
        "Saving a study as a template strips its agents and re-derives the "
        + "derived pins (the instrument scope is re-pinned against the task "
        + "file at every instantiation). Duplicate as Draft, in Studies, does "
        + "the opposite: it copies everything, agents included."
}
