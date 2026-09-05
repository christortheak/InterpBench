import ExperimentKit
import SteeringKit
import SwiftUI

/// Pinned concept recipes and draft attachment controls. Extraction is not launched here.
struct StudyConceptsSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    /// One-step concept attachment in Studies: attached concepts with their
    /// pin status (stimulus hash, method, reading position, three-state
    /// validation pin) and a detach action, plus a draft-only picker —
    /// concept, method, reading position, grand-mean corpus — that writes
    /// through `ExperimentStore.attachConcept` exactly like the CLI attach.
    @ViewBuilder
    var body: some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        Section {
            if manifest.concepts.isEmpty {
                Text(
                    "No concepts pinned. Injection conditions steer along "
                        + "pinned concepts; attaching pins the stimulus files "
                        + "by hash (the recipe, not vector bytes)."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            ForEach(manifest.concepts, id: \.name) { ref in
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(ref.name)
                            .font(.callout)
                        Text(ExperimentPanel.conceptPinStatusLine(ref))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    if manifest.status == .draft {
                        Button {
                            panel.detachConcept(ref.name)
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.plain)
                        .help(
                            "detach '\(ref.name)' from this draft — refused "
                                + "while a declaration still names it")
                    }
                }
            }
            if manifest.status == .draft {
                attachPickerRows(panel: panel)
            }
        } header: {
            InfoSectionHeader(
                title: "Build & Validate Concept Vectors",
                text: StudyInfo.conceptVectors)
        }
    }

    @ViewBuilder
    private func attachPickerRows(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        let sources = panel.attachableConceptSources
        if sources.isEmpty {
            Text(
                "no unattached concepts on disk — author a paired stimulus set "
                    + "under prompts/concepts/<name>/ or grand-mean stories "
                    + "under prompts/emotions/<name>/stories.jsonl"
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        } else {
            let selectedSource = sources.first { $0.name == panel.attachConceptName }
            Picker("Attach Concept…", selection: $draft.attachConceptName) {
                Text("select…").tag("")
                ForEach(sources, id: \.name) { source in
                    Text(source.pickerLabel).tag(source.name)
                }
            }
            .help(
                "concepts found on disk that this study has not pinned yet — "
                    + "labels say what each can support (paired stimulus set, "
                    + "grand-mean stories)"
            )
            .onChange(of: panel.attachConceptName) {
                // Snap the method to something the chosen concept's data can
                // actually support.
                if let source = sources.first(where: { $0.name == panel.attachConceptName }),
                    !source.supportedMethods.contains(panel.attachMethod),
                    let first = source.supportedMethods.first
                {
                    panel.attachMethod = first
                }
            }
            HStack(spacing: 8) {
                Picker("", selection: $draft.attachMethod) {
                    ForEach(
                        selectedSource?.supportedMethods
                            ?? ExtractionMethod.allCases.filter(\.isRecipeMethod),
                        id: \.self
                    ) { method in
                        Text(method.label).tag(method)
                    }
                }
                .frame(maxWidth: 210)
                .help(
                    "extraction method pinned into the recipe — paired methods "
                        + "read positive/negative stimuli; grand mean reads the "
                        + "multi-concept story corpus")
                ReadingPositionField(
                    choice: $draft.attachReadingPositionChoice,
                    parameter: $draft.attachReadingPositionParameter,
                    defaultCaption: "method default",
                    help:
                        "WHERE the residual stream is read, pinned into the "
                        + "recipe. 'method default' declares nothing and "
                        + "keeps the method's own position (last token for "
                        + "paired; token 50 for grand mean and designated "
                        + "reference). The content-side roles only exist "
                        + "inside a rendered turn, so they need the chat "
                        + "template beside this")
                Button("Attach") { panel.attachConceptFromPicker() }
                    .disabled(panel.attachConceptName.isEmpty)
                    .help(
                        "pins the concept at its CURRENT stimulus hash plus the "
                            + "measurement-side validation pin — same write as "
                            + "'steerlab-cli experiment attach'")
            }
            HStack(spacing: 8) {
                Text("rendering")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ExtractionRenderingField(
                    choice: $draft.attachRendering,
                    help:
                        "HOW each stimulus reaches the model. 'raw' is the "
                        + "legacy rendering and declares nothing — the bare "
                        + "string through the tokenizer. 'chat template' "
                        + "renders it the way a measured generation does; "
                        + "the two produce DIFFERENT directions, which is "
                        + "why the choice is pinned rather than inferred")
                Spacer(minLength: 0)
            }
            if panel.attachMethod == .designatedReference {
                Picker("reference", selection: $draft.attachReferenceName) {
                    Text("select reference…").tag("")
                    ForEach(sources.filter(\.hasStories), id: \.name) { source in
                        Text(source.name).tag(source.name)
                    }
                }
                .font(.caption)
                .help(
                    "the DESIGNATED reference corpus: the vector is "
                        + "mean(concept stories) − mean(reference stories), both "
                        + "pooled from token 50 — the reference pins into the "
                        + "recipe beside the concept")
            }
            if panel.attachMethod == .emotionGrandMean {
                TextField(
                    "extra corpus members (comma-separated; targets are always members)",
                    text: $draft.attachCorpusText
                )
                .font(.caption)
                .help(
                    "grand-mean vectors are concept mean − corpus grand mean, "
                        + "so the pinned population is part of the recipe — "
                        + "name extra prompts/emotions/ concepts to widen it")
            }
            Text(
                "attach pins stimulus bytes by hash + the concept's "
                    + "validation.jsonl (or its absence) — the firewall's "
                    + "measurement-side pins, identical to the CLI attach"
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }
}
