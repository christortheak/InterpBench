import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

// Evaluation-pane clarity subviews (2026-07-20 researcher round, item 4;
// reworked 2026-07-21, issues 5 and 7 of the authoring walkthrough).
// Extracted into their own file because `ExperimentsPanelView` sits at the
// type-checker's limits.

/// The single-source rubric controls (2026-07-21, issue 5): the rubric is
/// ONE file — chosen from prompts/rubrics/ or anywhere in the workspace,
/// shown with the standard file affordances (eye viewer, pinned hash,
/// pencil in-app editor), pinned by hash at Save Evaluation Settings. The
/// inline scratchpad appears ONLY while no file is chosen (the engine
/// ignores inline text whenever a file is pinned) and its one job besides
/// exploratory judging is writing itself out as the study's rubric file.
struct RubricFileControls: View {
    let manifest: ExperimentManifest
    @Bindable var panel: ExperimentPanel

    @State private var templateStatus: String?

    private var isDraft: Bool { manifest.status == .draft }

    private var selectedFile: String {
        panel.judgeRubricFile.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static var rubricTypes: [UTType] {
        [UTType(filenameExtension: "md") ?? .plainText, .plainText]
    }

    var body: some View {
        HStack(spacing: 6) {
            Text("Rubric file")
                .font(.caption)
                .foregroundStyle(.secondary)
            Menu {
                ForEach(panel.rubricFileOptions, id: \.self) { file in
                    Button(file) { panel.judgeRubricFile = file }
                }
                if !selectedFile.isEmpty {
                    Button("clear (no rubric file)") { panel.judgeRubricFile = "" }
                }
            } label: {
                Text(selectedFile.isEmpty ? "choose…" : selectedFile)
                    .font(.caption.monospaced())
                    .lineLimit(1)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(!isDraft)
            .help(
                "rubric files under prompts/rubrics/ — pinned by hash at "
                    + "Save Evaluation Settings; the folder button picks a "
                    + "rubric anywhere else in the workspace")
            WorkspacePathChooseButton(
                message: "Choose a judge rubric file inside the workspace",
                allowedTypes: Self.rubricTypes,
                startingSubdirectory: JudgeRubricStore.relativeDirectory,
                onChoose: { panel.judgeRubricFile = $0 },
                onProblem: { templateStatus = $0 }
            )
            .disabled(!isDraft)
            InfoButton(text: StudyInfo.rubricFile)
        }

        if selectedFile.isEmpty {
            noFileState
        } else {
            FileReferenceRow(
                label: "rubric",
                path: selectedFile,
                pinnedHash: manifest.judgeRubricHash,
                allowsOpenInEditor: true)
            Text(
                "editable in place — the pencil opens the in-app editor; "
                    + "edits to a pinned rubric surface as drift until Save "
                    + "Evaluation Settings re-pins the current bytes")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let templateStatus {
            Text(templateStatus)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    /// No rubric file chosen: the honest state line, the template
    /// scaffold, and the draft scratchpad with its write-to-file exit.
    @ViewBuilder
    private var noFileState: some View {
        Text(
            "no rubric file chosen — freezing a judge-evaluated study "
                + "requires one (pinned by hash); without judging declared, "
                + "none is needed")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        if isDraft, panel.rubricFileOptions.isEmpty {
            Button("Create rubric from template") { createFromTemplate() }
                .font(.caption)
                .help(
                    "copies the domain-neutral judging-criteria template "
                        + "to \(destination) (never overwrites) — edit "
                        + "the criteria, then Save Evaluation Settings "
                        + "pins it")
        }

        // The scratchpad: draft wording here, then write it out — the
        // FILE is the instrument (StudyInfo.rubricDraft states the rule).
        // Deliberately NOT draft-gated: Results' Run Paired Judge can use
        // inline text for an exploratory pass on a frozen file-less study.
        HStack(alignment: .top, spacing: 6) {
            TextField(
                "Rubric scratchpad (unpinned — write it to a file to keep it)",
                text: $panel.evaluationPrompt,
                axis: .vertical
            )
            .lineLimit(4 ... 10)
            .help(
                "draft-only scratchpad while iterating on rubric wording; "
                    + "the rubric FILE is the instrument and always wins "
                    + "once chosen")
            InfoButton(text: StudyInfo.rubricDraft)
        }
        if isDraft,
            !panel.evaluationPrompt
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            Button("Save scratchpad as rubric file") { saveScratchpadAsFile() }
                .font(.caption)
                .help(
                    "writes the scratchpad text to \(destination) and "
                        + "selects it (never overwrites a file with "
                        + "different contents) — Save Evaluation Settings "
                        + "then pins it by hash")
        }
    }

    private var destination: String {
        DataTemplates.judgeRubricDestination(experiment: manifest.name)
    }

    private func createFromTemplate() {
        // The readiness scaffold rule (copy template, refuse overwrite),
        // pointed at the rubric destination this study's readiness row uses.
        let requirement = DataRequirement(
            id: "judgeRubric", title: "judge rubric", kind: .judgeRubric,
            status: .missing, path: destination, detail: "",
            templateID: DataTemplates.judgeRubric.id)
        do {
            let created = try StudyDataReadiness.scaffold(
                requirement: requirement, in: VectorCatalog.projectRoot)
            panel.judgeRubricFile = destination
            templateStatus = "created \(created.path) — replace the example "
                + "criteria with your study's, then Save Evaluation Settings "
                + "pins it by hash"
        } catch {
            templateStatus = "could not create the rubric — \(error)"
        }
    }

    /// The scratchpad's write-to-file exit, under the house write rule:
    /// identical bytes are idempotent (just select), differing bytes
    /// refuse — a scratchpad save never silently replaces a file.
    private func saveScratchpadAsFile() {
        let text = panel.evaluationPrompt
            .trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let url = VectorCatalog.projectRoot.appending(path: destination)
        let data = Data(text.utf8)
        do {
            if let existing = try? Data(contentsOf: url) {
                if existing == data {
                    panel.judgeRubricFile = destination
                    templateStatus = "\(destination) already holds exactly "
                        + "this text — selected it; Save Evaluation Settings "
                        + "pins it by hash"
                } else {
                    templateStatus = "\(destination) already exists with "
                        + "DIFFERENT contents — scratchpad saves never "
                        + "overwrite. Select the file and use the pencil "
                        + "editor to change it deliberately"
                }
                return
            }
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
            panel.judgeRubricFile = destination
            templateStatus = "wrote \(destination) and selected it — Save "
                + "Evaluation Settings pins it by hash"
        } catch {
            templateStatus = "could not write the rubric file — \(error)"
        }
    }
}

/// Case family as an editable field with suggestions (the manifest accepts
/// any string — `ExperimentStore.setCaseFamily` does not validate against
/// the known list, which is offer-only vocabulary), plus the honest ⓘ and
/// plain menu descriptions (2026-07-21, issue 7): only 'sentencing'
/// activates a built-in endpoint parser; the other names are provenance
/// labels, and a declared registry parser wins over the family either way.
struct CaseFamilyField: View {
    let manifest: ExperimentManifest
    @Bindable var panel: ExperimentPanel

    private var isDraft: Bool { manifest.status == .draft }

    /// What each suggested family ACTUALLY does, verified against both
    /// engines' dispatch (`judicialParses` / `_execute_condition`): the field
    /// is a provenance label, and the ONE value that still switches parsing on
    /// does so through a deprecated implicit rule. A/B choice parsing reads
    /// from per-item 'options' regardless of family.
    private static let familyDescriptions: [String: String] = [
        "siliconFormalism":
            "legacy choice-of-law doctrine family (label only — no built-in "
            + "parser)",
        "katzZamir":
            "legacy rule-vs-justice vignette family (label only — the A/B "
            + "choice reads from each item's declared options)",
        "sentencing":
            "sentencing studies — DEPRECATED: with no numeric parser "
            + "declared this label still selects the built-in duration "
            + "endpoint (parsedMonths); declare the 'sentencing-months' "
            + "numeric parser below instead",
    ]

    var body: some View {
        HStack(spacing: 6) {
            LabeledContent("Case family") {
                TextField("not declared", text: $panel.caseFamilyField)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)
                Menu {
                    ForEach(ExperimentStore.knownCaseFamilies, id: \.self) { family in
                        Button(
                            Self.familyDescriptions[family].map {
                                "\(family) — \($0)"
                            } ?? family
                        ) { panel.caseFamilyField = family }
                    }
                    Button("clear (not declared)") { panel.caseFamilyField = "" }
                } label: {
                    Image(systemName: "chevron.down.circle")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("suggestions carried over from the shipped example "
                    + "study — free text is equally valid (saved as "
                    + "provenance)")
            }
            .disabled(!isDraft)
            .help(
                "free text, saved with Save Evaluation Settings — a "
                    + "provenance LABEL. Declare a Numeric parser (below) to "
                    + "say how answers are read into endpoints; a declared "
                    + "parser always wins. The one exception is deprecated: "
                    + "with no parser declared, 'sentencing' still selects "
                    + "the built-in duration-in-months endpoint")
            InfoButton(text: StudyInfo.caseFamily)
        }
    }
}
