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

    @State private var showPasteSheet = false
    @State private var pasteText = ""

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
            // One line here; the full guide (what it is, what you provide,
            // what it measures) renders large in the Study Guide viewer on
            // the right.
            Label(
                panel.studyFocus.tagline + " Full guide in the Study Guide "
                    + "pane on the right.",
                systemImage: panel.studyFocus.systemImage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let hidden = panel.studyFocus.hiddenContentNote(for: manifest) {
                Label(hidden, systemImage: "eye.slash")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            HStack {
                Button("Copy Study JSON") {
                    guard let json = panel.exportSelectedStudyJSON() else { return }
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(json, forType: .string)
                    panel.note(
                        "study JSON copied — paste it into an editor or an "
                            + "LLM conversation; Paste Study JSON imports the "
                            + "result as a new draft",
                        severity: .success)
                }
                .help(
                    "the study's manifest as one JSON document — the same "
                        + "experiment.json the CLI and server read: the "
                        + "recipe and its hash pins. Referenced file BYTES "
                        + "(prompts, rubrics, scenarios) live in the "
                        + "workspace and travel to the server in run "
                        + "bundles (evidence bundles bring results home), "
                        + "not in this document")
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
                Button("Copy LLM Prompt") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(
                        StudyCoauthoring.prompt(for: panel.studyFocus),
                        forType: .string)
                    panel.note(
                        "co-authoring prompt for a "
                            + "\(panel.studyFocus.displayName) study copied — "
                            + "paste it into an LLM conversation; it "
                            + "interviews you and produces a study PACK "
                            + "(manifest + data files) to paste back here",
                        severity: .success)
                }
                .help(
                    "a prompt keyed to the selected study type: it teaches "
                        + "any capable LLM the study-pack format (manifest + "
                        + "data files in one document) and what to ask you — "
                        + "work out the study in conversation, then Paste "
                        + "Study JSON imports the pack as a draft, writes "
                        + "its files, and pins them")
            }
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
            HStack {
                Spacer()
                Button("Cancel") { showPasteSheet = false }
                Button("Import as Draft") {
                    panel.importStudyJSON(pasteText)
                    showPasteSheet = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(
                    pasteText.trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty)
            }
        }
        .padding(16)
    }
}
