import AppKit
import ExperimentKit
import SwiftUI

/// "What kind of study is this?" — asked ONCE, first on the page
/// (2026-07-19 second pass: replaces the former Study stage + Study Focus
/// duo). The picker writes the draft's studyKind and filters every section
/// below it; because it renders above everything it toggles, switching
/// types can no longer jump the scroll position. Also home to study.json
/// copy/paste — the manifest IS one JSON document, buildable by the app,
/// by hand, or by an LLM and pasted here (imports are always drafts and
/// verify() runs immediately; the firewall is identical either way).
struct StudyTypeSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    @State private var artifactStudy: DraftAuthoringSnapshot?
    @State private var showArtifactSheet = false
    @State private var showPasteSheet = false
    @State private var pasteText = ""
    @State private var packPreview: StudyPackAuthoring.Preview?
    @State private var packPreviewError: String?

    var body: some View {
        Section {
            HStack(spacing: 6) {
                Picker("Study type", selection: typeBinding) {
                    ForEach(StudyIntent.allCases) { intent in
                        Text(intent.displayName).tag(intent)
                    }
                }
                .help(
                    "what this study is trying to do — decides which sections "
                        + "show below and, on drafts, is saved into the study. "
                        + "Switching never deletes anything: data a type's view "
                        + "hides is called out right here")
                InfoButton(text: StudyInfo.studyType)
            }
            // On a non-draft the picker still moves, but `setStudyType` only
            // sets a view override — nothing is written. Say so inline rather
            // than only in the hover text (UI audit 2026-09-06).
            if manifest.status != .draft {
                Text(
                    "view only — the \(manifest.status.rawValue) study's "
                        + "declared type is \(declaredIntent.displayName); "
                        + "changing the picker only re-arranges this page"
                )
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
            }
            // One line here; the full guide (what it is, what you provide,
            // what it measures) renders large in the Selected Study viewer on
            // the right, beside the study's own manifest JSON.
            Label(
                panel.studyFocus.tagline + " Full guide in the Selected Study "
                    + "pane on the right (Guide).",
                systemImage: panel.studyFocus.systemImage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let hidden = panel.studyFocus.hiddenContentNote(for: manifest) {
                Label(hidden, systemImage: "eye.slash")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if manifest.status == .draft {
                Button("Attach vector artifact…") {
                    do {
                        artifactStudy = try panel.management.reviewStudy(named: manifest.name)
                        showArtifactSheet = true
                    } catch {
                        panel.note(
                            "Couldn't open the attach sheet — reload the study "
                                + "and try again. Details: "
                                + error.localizedDescription,
                            severity: .error)
                    }
                }
                .help(
                    "pin a steering vector that already exists in this "
                        + "workspace to one of this draft's concepts — checks "
                        + "the vector's provenance and compatibility; it never "
                        + "trains or alters the vector")
                .sheet(isPresented: $showArtifactSheet) {
                    if let artifactStudy { StudyArtifactAttachmentSheet(reviewed: artifactStudy, panel: panel) }
                }
            }
            HStack {
                // CopyButton, not a bare Button + note: the note renders in a
                // Status section at the very bottom of a long form, so at the
                // top of the page nothing changed on click (UI audit
                // 2026-09-06, headline 19).
                CopyButton(
                    "Copy Study JSON", systemImage: "curlybraces",
                    help: Self.copyStudyJSONHelp
                ) { panel.exportSelectedStudyJSON() }
                Button("Paste Study JSON…") {
                    pasteText = NSPasteboard.general
                        .string(forType: .string) ?? ""
                    showPasteSheet = true
                }
                .help(
                    "import a study JSON (hand-written or LLM-drafted) as a "
                        + "NEW DRAFT: freeze metadata is stripped — pasted "
                        + "text cannot mint a preregistered study — and "
                        + "verification runs immediately")
                CopyButton(
                    "Copy LLM Prompt", systemImage: "text.bubble",
                    help: Self.copyLLMPromptHelp
                ) { StudyCoauthoring.prompt(for: panel.studyFocus) }
            }
            // The round trip, visible: this used to live only in the notice
            // the two buttons posted at the bottom of the page.
            Text(
                "Copy either one into an LLM conversation, work the study out "
                    + "there, then Paste Study JSON imports the result as a "
                    + "new draft."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        } header: {
            Text("Study Type")
        }
        .sheet(isPresented: $showPasteSheet) { pasteSheet }
    }

    private var typeBinding: Binding<StudyIntent> {
        Binding(
            get: { panel.studyFocus },
            set: { panel.setStudyType($0) })
    }

    /// What the STUDY says it is, ignoring the page's view override — the
    /// thing a frozen study's caption has to name.
    private var declaredIntent: StudyIntent {
        StudyIntent.derive(from: manifest)
    }

    private var pasteSheet: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Paste Study JSON")
                .font(.headline)
            Text("Imports as a new DRAFT under the JSON's \"name\" (must not "
                + "already exist). Freeze metadata is stripped; the study is "
                + "checked immediately (verify) and any problems found arrive "
                + "as a loud notice.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            TextEditor(text: $pasteText)
                .font(.caption.monospaced())
                .frame(minWidth: 520, minHeight: 320)
                .border(.quaternary)
                .help(
                    "paste one study JSON document, or a study PACK (the "
                        + "manifest plus its data files) — the same "
                        + "experiment.json every engine reads")
            if let packPreview {
                Text("New draft: \(packPreview.name)").font(.subheadline)
                Text("\(packPreview.files.filter { $0.disposition == "create" }.count) new input files; "
                    + "\(packPreview.files.filter { $0.disposition == "reuse" }.count) identical files reused.")
                    .font(.caption)
                ScrollView {
                    VStack(alignment: .leading) {
                        ForEach(packPreview.files, id: \.path) { file in
                            Text("\(file.disposition): \(file.path)").font(.caption.monospaced())
                        }
                    }
                }.frame(maxHeight: 120)
                Text("Inputs are pinned and verification issues reported after import. This preview does not certify that the study is ready to run.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let packPreviewError {
                Label(packPreviewError, systemImage: "xmark.octagon.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            // The import refusal, IN the sheet. It used to reach only the
            // status line at the bottom of the form and the bell — both
            // behind this modal (UI audit 2026-09-06, headline 9).
            if let refusal = panel.draft.formErrors[.studyImport] {
                Label(refusal, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            // Why Import is grey before Preview — the researcher used to
            // paste, press Return, and watch nothing happen.
            if packPreview == nil, !pasteIsEmpty {
                Text("Preview first — Import as Draft turns on once the pack parses.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { showPasteSheet = false }
                    .keyboardShortcut(.cancelAction)
                    .help("close without importing; nothing is written")
                Button("Preview") {
                    do {
                        packPreview = try StudyPackAuthoring.preview(Data(pasteText.utf8), workspaceRoot: ExperimentStore.workspaceRoot)
                        packPreviewError = nil
                    } catch {
                        packPreview = nil
                        packPreviewError = error.localizedDescription
                    }
                }
                .disabled(pasteIsEmpty)
                .help(
                    "parses the pasted document and lists the input files the "
                        + "import would write or reuse — reads nothing into "
                        + "the workspace")
                Button("Import as Draft") {
                    guard let packPreview else { return }
                    if panel.importStudyJSON(pasteText, reviewed: packPreview) { showPasteSheet = false }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(packPreview == nil || pasteIsEmpty)
                .help(
                    "writes the previewed pack as a NEW draft under its own "
                        + "name, pins its files, and runs verification — "
                        + "enabled once Preview has parsed the document")
            }
        }
        .padding(16)
        // Constant frame: the optional preview block used to change the
        // sheet's content height after it was already up, resizing it under
        // the researcher (UI audit 2026-09-06).
        .frame(minWidth: 560, idealWidth: 620, minHeight: 620, idealHeight: 700)
        .onChange(of: pasteText) { _, _ in
            packPreview = nil
            packPreviewError = nil
            panel.clearFormError(.studyImport)
        }
    }

    private var pasteIsEmpty: Bool {
        pasteText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // Long strings live outside the body — interpolating them inline blows the
    // SwiftUI type-checker budget.

    private static let copyStudyJSONHelp =
        "the study's manifest as one JSON document — the same experiment.json "
        + "the CLI and server read: the recipe and its hash pins. Referenced "
        + "file BYTES (prompts, rubrics, scenarios) live in the workspace and "
        + "travel to the server in run bundles (evidence bundles bring results "
        + "home), not in this document"

    private static let copyLLMPromptHelp =
        "a prompt keyed to the selected study type: it teaches any capable LLM "
        + "the study-pack format (manifest + data files in one document) and "
        + "what to ask you — work the study out in conversation, then Paste "
        + "Study JSON imports the pack as a draft, writes its files, and pins "
        + "them"
}
