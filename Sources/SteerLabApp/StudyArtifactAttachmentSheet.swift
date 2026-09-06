import ExperimentKit
import SwiftUI

/// The sheet keeps the study review from opening and a separate artifact review
/// from Inspect. Changing either input requires another review, never a fresh pin.
struct StudyArtifactAttachmentSheet: View {
    let reviewed: DraftAuthoringSnapshot
    let panel: ExperimentPanel
    @Environment(\.dismiss) private var dismiss
    @State private var path = ""
    @State private var concept = ""
    @State private var sourceConcept = ""
    @State private var evalRun = ""
    @State private var artifact: StudyArtifactAuthoring.Review?
    @State private var problem: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Attach vector to \(reviewed.manifest.name)").font(.headline)
            Text("Choose an existing vector in this workspace. Attachment checks its provenance and compatibility; it does not train or alter the vector.")
                .font(.callout).foregroundStyle(.secondary)
            TextField("Vector path (without extension)", text: $path)
                .onChange(of: path) { _, _ in artifact = nil; problem = nil }
            TextField("Concept name in this study", text: $concept)
            TextField("Source concept (only if the artifact requires it)", text: $sourceConcept)
            TextField("Evaluation run (optional)", text: $evalRun)
            Button("Inspect vector") {
                do {
                    artifact = try StudyArtifactAuthoring.inspect(path, workspaceRoot: reviewed.workspaceRoot)
                    problem = nil
                } catch { artifact = nil; problem = "\(error)" }
            }
            if let artifact {
                Text("Vector SHA-256: \(artifact.artifactSHA256)\nSidecar SHA-256: \(artifact.sidecarSHA256)")
                    .font(.caption.monospaced()).textSelection(.enabled)
                Text("Scientific admission runs when attaching. A missing prerequisite is reported with its repair.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let problem {
                Text(problem).foregroundStyle(.red).font(.callout).textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Attach reviewed vector") {
                    guard let artifact else { return }
                    do {
                        try panel.management.attachArtifact(concept, artifact: artifact, reviewed: reviewed,
                            sourceConcept: optional(sourceConcept), evalRun: optional(evalRun))
                        dismiss()
                    } catch { problem = "\(error)" }
                }.disabled(artifact == nil || concept.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.padding(20).frame(width: 640)
    }
    private func optional(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
