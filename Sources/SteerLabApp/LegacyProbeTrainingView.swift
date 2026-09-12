import AppKit
import ExperimentKit
import SwiftUI

/// Preserves the existing Playground reader owner while the portable instruments
/// acquire their separate study measurement adapter.
struct LegacyProbeTrainingView: View {
    @Bindable var service: ChatService
    var body: some View {
        @Bindable var builder = service.concepts
        VStack(alignment: .leading, spacing: 10) {
            Text("Existing Playground reader").font(.headline)
            Text("This uses the original reader format and layer-selection procedure. Its validation accuracy chooses a layer; it is not final-test performance. The new portable classifiers above do not yet drive Playground highlighting.").font(.caption)
            Picker("Concept", selection: Binding(get: { builder.selectedExisting }, set: { value in
                if let value { _ = builder.selectConcept(value) }
            })) {
                Text("Choose a concept from Data").tag(String?.none)
                ForEach(builder.existingConcepts, id: \.self) { Text($0).tag(String?.some($0)) }
            }
            Text("\(builder.probePositiveCount) positive and \(builder.probeNegativeCount) control examples")
            Text("Pasted examples are saved locally. Server training uploads these local examples when present, then trains on the selected server. The original local reader uses the concept’s extraction recipe and reading settings from Data.").font(.caption)
            HStack {
                Stepper("Prompt examples: \(builder.probeGenerationCount)", value: $builder.probeGenerationCount, in: 20...600, step: 20)
                Button("Copy authoring prompt") {
                    if let prompt = builder.probeGenerationPrompt() {
                        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(prompt, forType: .string)
                    }
                }.disabled(builder.currentConceptName.isEmpty)
            }
            TextEditor(text: $builder.probeDraft).font(.callout.monospaced()).frame(height: 110)
                .help("Paste legacy JSONL with text, expresses (true or false), topic, and split. Import saves the examples; it does not train.")
            HStack {
                Button("Import examples") { Task { await builder.addProbeDrafts() } }
                    .disabled(builder.currentConceptName.isEmpty || builder.probeDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || builder.isWorking)
                if service.cluster.computeTarget == .server {
                    Button("Upload examples and train server reader") { Task { await builder.trainProbeOnActiveServer() } }
                        .disabled(service.selectedRemoteModelID == nil || builder.isWorking)
                } else {
                    Button("Train Playground reader") { Task { await builder.trainReadingProbe() } }
                        .disabled(builder.probePositiveCount < 4 || builder.probeNegativeCount < 4 || builder.isWorking)
                }
                if builder.isWorking { ProgressView() }
            }
            Text("Local training needs at least four examples of each class and a loaded local model. Server training needs a selected server model.").font(.caption)
            if let cost = builder.probeTrainingCostLine() { Text(cost).font(.caption) }
            if let failure = builder.lastBuildError { Text(failure).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            if let status = builder.status { Text(status).font(.caption).textSelection(.enabled) }
        }
    }
}
