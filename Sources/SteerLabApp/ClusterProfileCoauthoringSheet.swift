import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// A document-to-profile handoff over the same offline command as the CLI.
struct ClusterProfileCoauthoringSheet: View {
    let repository: ClusterSiteRepository
    let onImport: (ClusterSiteRecord) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var showingImporter = false
    @State private var review: ClusterProfileCoauthoring.Review?
    @State private var message: String?
    @State private var reviewedData: Data?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configure from documentation").font(.title2)
            Text("Give your agent the cluster documentation and this prompt. It will return a profile with sources and questions about missing facts. Credentials stay in the Keychain.")
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                CopyButton(
                    help: "copies the authoring prompt, the independent "
                        + "reviewer's prompt, and an example of the JSON shape "
                        + "the agent must answer with",
                    text: { promptPacket() }
                ) {
                    Text("Copy agent and reviewer prompts")
                }
                Button("Review agent draft…") { showingImporter = true }
                    .help("reads the JSON your agent produced and checks its "
                        + "declarations and source references — it never "
                        + "contacts the cluster")
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
                            // Same ordering the reviewer emitted; offsets as
                            // ids because two advisories may read alike.
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
                        .help("every fact the draft claims, with the document "
                            + "and locator it was taken from — read these "
                            + "against the site's own documentation")
                        ForEach(Array(review.advisories.enumerated()), id: \.offset) { _, advisory in
                            Text(advisory).font(.caption).foregroundStyle(.secondary)
                        }
                        ForEach(Array(review.blockers.enumerated()), id: \.offset) { _, blocker in
                            Text(blocker).foregroundStyle(.orange)
                        }
                        ForEach(Array(review.questions.enumerated()), id: \.offset) { _, question in
                            Text(question.question).foregroundStyle(.orange)
                        }
                        ClusterSitePreviewPanes(preview: review.preview, paneHeight: 160, expandsEnvironment: false)
                    }
                    .textSelection(.enabled)
                }
            }
            HStack {
                if let reason = importDisabledReason {
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Button("Close", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("closes without adding anything to your Sites registry")
                Button("Import reviewed profile") {
                    guard let review, let reviewedData, review.readyForImport else { return }
                    do {
                        let accepted = try ClusterProfileAcceptance.accept(data: reviewedData,
                            expectedSHA256: review.draftSHA256, repository: repository)
                        onImport(accepted.site)
                        dismiss()
                    } catch { message = error.localizedDescription }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(review?.readyForImport != true)
                .help(importDisabledReason
                    ?? "copies the reviewed profile into your Sites registry "
                        + "and selects it — the same checks the command line "
                        + "runs, against the draft's own checksum")
            }
        }
        .padding(20)
        .frame(minWidth: 760, minHeight: 600)
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.json]) { result in
            do {
                let url = try result.get()
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                let data = try Data(contentsOf: url)
                review = try ClusterProfileCoauthoring.review(data: data)
                reviewedData = data
                message = nil
            } catch {
                review = nil
                reviewedData = nil
                message = "Could not review the draft: \(error.localizedDescription)"
            }
        }
    }

    /// Why "Import reviewed profile" cannot be pressed yet — visible beside
    /// the button, not only in its tooltip.
    private var importDisabledReason: String? {
        guard let review else {
            return "no draft reviewed yet — copy the prompts, then open your "
                + "agent's JSON with Review agent draft…"
        }
        guard review.readyForImport else {
            return "the review left questions or blockers above — answer them "
                + "with your agent and review the corrected draft"
        }
        return nil
    }

    /// The packet the agent needs, or nil when it could not be built (the
    /// reason lands in the sheet's message line).
    private func promptPacket() -> String? {
        do {
            let guide = try ClusterProfileCoauthoring.guide()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let example = String(
                decoding: try encoder.encode(guide.draftExample), as: UTF8.self)
            message = "Share the copied prompts with your agent alongside the "
                + "documentation."
            return guide.authorPrompt + "\n\nIndependent reviewer:\n"
                + guide.reviewerPrompt
                + "\n\nCompanion format (incomplete example):\n" + example
        } catch {
            message = "Could not build the prompts: \(error.localizedDescription)"
            return nil
        }
    }
}
