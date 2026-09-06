import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// A document-to-profile handoff over the same offline command as the CLI.
struct ClusterProfileCoauthoringSheet: View {
    let onImport: (Data) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var showingImporter = false
    @State private var review: ClusterProfileCoauthoring.Review?
    @State private var message: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configure from documentation").font(.title2)
            Text("Give your agent the cluster documentation and this prompt. It will return a profile with sources and questions about missing facts. Credentials stay in the Keychain.")
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Copy agent and reviewer prompts") {
                    do {
                        let guide = try ClusterProfileCoauthoring.guide()
                        let encoder = JSONEncoder()
                        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                        let example = String(decoding: try encoder.encode(guide.draftExample), as: UTF8.self)
                        let packet = guide.authorPrompt + "\n\nIndependent reviewer:\n" + guide.reviewerPrompt
                            + "\n\nCompanion format (incomplete example):\n" + example
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(packet, forType: .string)
                        message = "Prompts copied. Share them with your agent alongside the documentation."
                    } catch { message = error.localizedDescription }
                }
                Button("Review agent draft…") { showingImporter = true }
            }
            if let message { Text(message).font(.caption).textSelection(.enabled) }
            ScrollView {
                if let review {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(review.readyForImport ? "Ready for your factual review" : "Questions remain")
                            .font(.headline)
                        Text("This check verifies declarations and source references. Review the citations with your agent before accepting the configuration. It does not test connectivity or authorize cluster work.")
                            .font(.caption)
                        switch review.profile.transport {
                        case .ssh(let host, let jump, let port, _):
                            Text("SSH destination: " + host + " · port " + String(port)
                                + (jump.map { " · via " + $0 } ?? "")).font(.caption)
                        case .direct(let url):
                            Text("Server: " + url.absoluteString).font(.caption)
                        }
                        Text(SiteEditorModel.topologyExplanation(review.profile.topology)).font(.caption)
                        DisclosureGroup("Sources and declarations") {
                            ForEach(Array(review.sources.enumerated()), id: \.offset) { _, source in
                                Text(source.reference).font(.caption)
                            }
                            ForEach(Array(review.facts.enumerated()), id: \.offset) { _, fact in
                                VStack(alignment: .leading) {
                                    Text(fact.path).font(.caption.monospaced())
                                    Text(fact.explanation).font(.caption)
                                    Text(fact.sourceID + " · " + fact.locator).font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                        ForEach(review.advisories, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                        ForEach(review.blockers, id: \.self) { Text($0).foregroundStyle(.orange) }
                        ForEach(Array(review.questions.enumerated()), id: \.offset) { _, question in
                            Text(question.question).foregroundStyle(.orange)
                        }
                        ClusterSitePreviewPanes(preview: review.preview, paneHeight: 160, expandsEnvironment: false)
                    }
                    .textSelection(.enabled)
                }
            }
            HStack {
                Spacer()
                Button("Close") { dismiss() }
                Button("Import reviewed profile") {
                    guard let review, review.readyForImport else { return }
                    do {
                        let data = try review.profile.encoded()
                        dismiss()
                        onImport(data)
                    } catch { message = error.localizedDescription }
                }
                .disabled(review?.readyForImport != true)
            }
        }
        .padding(20)
        .frame(minWidth: 760, minHeight: 600)
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.json]) { result in
            do {
                let url = try result.get()
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                review = try ClusterProfileCoauthoring.review(data: Data(contentsOf: url))
                message = nil
            } catch {
                review = nil
                message = "Could not review the draft: \(error.localizedDescription)"
            }
        }
    }
}
