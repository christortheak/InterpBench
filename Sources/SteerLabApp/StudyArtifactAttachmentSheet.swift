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
            Text(Self.explainer)
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            TextField("Vector path (without extension)", text: $path)
                .font(.body.monospaced())
                .onChange(of: path) { _, _ in artifact = nil; problem = nil }
                .help(Self.pathHelp)
            TextField("Concept name in this study", text: $concept)
                .help(Self.conceptHelp)
            TextField("Source concept (only if the artifact requires it)", text: $sourceConcept)
                .help(Self.sourceConceptHelp)
            TextField("Evaluation run (optional)", text: $evalRun)
                .help(Self.evalRunHelp)
            // The gate, said before the click: Attach is off until Inspect has
            // read the artifact, because the hashes it prints are what gets
            // pinned (UI audit 2026-09-06).
            HStack(spacing: 8) {
                Button("Inspect vector") {
                    do {
                        artifact = try StudyArtifactAuthoring.inspect(path, workspaceRoot: reviewed.workspaceRoot)
                        problem = nil
                    } catch {
                        artifact = nil
                        problem = error.localizedDescription
                    }
                }
                .disabled(pathIsEmpty)
                .help(Self.inspectHelp)
                if artifact == nil {
                    Text(
                        pathIsEmpty
                            ? "enter a workspace-relative vector path to inspect"
                            : "Inspect first — Attach pins the hashes it reads"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
            }
            if let artifact {
                Text("Vector SHA-256: \(artifact.artifactSHA256)\nSidecar SHA-256: \(artifact.sidecarSHA256)")
                    .font(.caption.monospaced()).textSelection(.enabled)
                Text("Scientific admission runs when attaching. A missing prerequisite is reported with its repair.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let problem {
                Label(problem, systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.red).font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without attaching; nothing is pinned")
                Button("Attach vector") {
                    guard let artifact else { return }
                    do {
                        try panel.management.attachArtifact(concept, artifact: artifact, reviewed: reviewed,
                            sourceConcept: optional(sourceConcept), evalRun: optional(evalRun))
                        dismiss()
                    } catch { problem = error.localizedDescription }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!attachable)
                .help(
                    attachable
                        ? "pins the inspected vector into this draft under the "
                            + "concept name above, by both hashes"
                        : "inspect a vector and give it a concept name in this "
                            + "study first")
            }
        }
        // A height as well as a width: without one the sheet grew and shrank
        // as the hash block and the refusal appeared (UI audit 2026-09-06).
        // `width:` and `minHeight:` are different `frame` overloads and
        // cannot be mixed; pinning min = max width is the same fixed width.
        .padding(20).frame(minWidth: 640, maxWidth: 640, minHeight: 420)
    }

    private var pathIsEmpty: Bool {
        path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var attachable: Bool {
        artifact != nil
            && !concept.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func optional(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    // Long strings live outside the body.

    private static let explainer =
        "Choose an existing vector in this workspace. Attachment checks its "
        + "provenance and compatibility; it does not train or alter the vector."

    private static let pathHelp =
        "the vector's path relative to the workspace root, WITHOUT the "
        + "extension — the .safetensors tensor and its .json sidecar are read "
        + "as a pair. A path that is absolute, contains .., or resolves "
        + "outside the workspace is refused."

    private static let conceptHelp =
        "the name this study will know the vector by — an injection "
        + "condition's slot references exactly this name, so it is the "
        + "study's own label, not the file's"

    private static let sourceConceptHelp =
        "the concept the vector was DERIVED from, when that differs from "
        + "the name above — required only by artifacts whose sidecar names "
        + "a source (a designated-reference or grand-mean vector reused "
        + "under another label); leave it empty otherwise"

    private static let evalRunHelp =
        "a runs/ directory whose evidence backs this vector, recorded "
        + "beside the pin as provenance — it gates nothing and may be left "
        + "empty"

    private static let inspectHelp =
        "reads the vector and its sidecar at that path and shows their "
        + "SHA-256 — Attach pins exactly the bytes inspected here, so it "
        + "stays off until this succeeds"
}
