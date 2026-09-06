import ExperimentKit
import SteeringKit
import SwiftUI

/// Prompt content and pin actions. Import presentation remains with the parent so collapse does not dismiss it.
struct StudyTaskPromptsEditor: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @Binding var showImportJSONL: Bool
    @Binding var importJSONLText: String

    /// The task-prompts CONTENT editor, rendered INSIDE Data & Prompts
    /// under the row that tracks the same file (2026-07-19: the standalone
    /// "Task Prompt Contents" section read as a second, mysterious copy).
    @ViewBuilder
    var body: some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        HStack(spacing: 6) {
            TextField("task prompts JSONL", text: $draft.taskPromptsFile)
                .disabled(manifest.status != .draft)
                .help(
                    "relative path to a {\"text\": ...}-per-line JSONL file. "
                        + "Save & Pin Prompts creates a new version and pins its hash into the study")
            // Phase 3 item 12: the no-typing route — pick the file, the
            // path lands workspace-relative, and the pin is made on
            // selection through the same validating pin path.
            WorkspacePathChooseButton(
                message: "Choose this study's task-prompts JSONL "
                    + "(workspace files only — the path pins on selection)",
                allowedTypes: WorkspaceFileChooser.jsonlTypes,
                startingSubdirectory: "prompts/tasks",
                onChoose: { panel.pinChosenTaskPromptsFile($0) },
                onProblem: { panel.note($0, severity: .error) }
            )
            .disabled(manifest.status != .draft)
        }
        HStack {
            Button("Load Prompts") { panel.loadTaskPrompts() }
                .help("read the JSONL file into the editor below")
            Button("Save & Pin Prompts") { panel.saveTaskPrompts() }
                .disabled(manifest.status != .draft)
                .help(StudyInfo.taskPromptsSavePin)
            Button("Import JSONL…") {
                importJSONLText = ""
                showImportJSONL = true
            }
            .disabled(manifest.status != .draft)
            .help(
                "paste or choose a raw JSONL records file (full "
                    + "records — options/target preserved, required "
                    + "by the answer-token instrument). Parsed with "
                    + "a preview; on import the file lands at the "
                    + "study's task-prompts destination, becomes "
                    + "this study's prompts file, and its hash is "
                    + "pinned — one action from paste to pinned")
            // Phase 3 item 13: spreadsheets (JSON array / CSV) enter
            // through a column-mapping sheet instead of hand-written
            // JSONL; same destination, same pin.
            TabularImportButton(
                target: .taskPrompts, panel: panel,
                disabled: manifest.status != .draft)
            // Phase 4 items 20–21: factorial/counterbalancing generation —
            // authoring-time data generation; the emitted file pins like
            // any hand-authored prompts (sheet lives in its own file).
            FactorialDesignButton(
                panel: panel, disabled: manifest.status != .draft)
        }
        // What Save & Pin actually DOES, visible — not hover-only
        // (2026-07-20 researcher round, item 2a).
        Text(
            "Save & Pin creates a new prompt version and updates this draft's path and hash. "
                + "The original file remains available to studies that already use it."
        )
        .font(.caption2)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
        TextEditor(text: $draft.taskPromptsText)
            .font(.system(.caption, design: .monospaced))
            .frame(minHeight: 220)
            .disabled(manifest.status != .draft)
            .help(
                "write one prompt per block. Long multi-paragraph prompts are fine; "
                    + "separate prompts with a line containing only ---")
        // Paste-detection guard: content in the PLAIN prompt editor that
        // looks like JSONL records would be saved as literal prompt text
        // (options/target lost). Offer the Import JSONL path — never
        // silently reinterpret.
        if manifest.status == .draft,
            TaskPromptsImport.looksLikeJSONL(panel.draft.taskPromptsText)
        {
            HStack(spacing: 8) {
                Label(
                    "this looks like JSONL records, not prompt text — "
                        + "Save & Pin would store the JSON itself as "
                        + "prompts and discard options/target",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption2)
                .foregroundStyle(.orange)
                Button("Import as JSONL…") {
                    importJSONLText = panel.draft.taskPromptsText
                    showImportJSONL = true
                }
                .controlSize(.small)
            }
        }
        if let instrumentSummary = panel.draft.taskPromptsInstrumentSummary {
            Label(instrumentSummary, systemImage: "list.bullet.rectangle")
                .font(.caption2)
                .foregroundStyle(.orange)
                .help(
                    "these items carry per-item instrument fields (options, "
                        + "target, …) that the text editor does not show — "
                        + "they are preserved byte-faithfully on save")
        }
        if let promptStatus = panel.draft.taskPromptsStatus {
            Text(promptStatus)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

}
