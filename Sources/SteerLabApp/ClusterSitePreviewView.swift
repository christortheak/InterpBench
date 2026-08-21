import ExperimentKit
import SwiftUI

/// The WP5 §3.3 preview panes, as views. The site editor shows all of them; the
/// setup wizard shows the same panes for the site it is about to provision.
///
/// **These views format nothing.** Every string comes from `ClusterSitePreview`
/// (ExperimentKit), which is unit-tested against the committed cross-engine
/// goldens — so what an admin reads here, what `steerlab-cli cluster preview`
/// prints, and what the renderer will push are the same bytes.
///
/// Layout note (macOS 27 beta, project memory): pane bodies live in
/// `DisclosureGroup`s whose contents are height-capped and internally
/// scrollable, so expanding one never changes an enclosing column's minimum.
struct ClusterSitePreviewPanes: View {
    let preview: ClusterSitePreview
    /// Height cap for a pane body. The wizard's inline copy is shorter than the
    /// editor's sheet.
    var paneHeight: CGFloat = 220
    /// Whether the panes start open. The editor opens the environment pane
    /// (that is what the section is FOR); the wizard keeps everything closed so
    /// step 1 stays a step and not a wall of text.
    var expandsEnvironment: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(preview.defaultSetSummary)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            DisclosureGroup(isExpanded: $environmentExpanded) {
                monospacedPane(preview.envFile)
            } label: {
                paneLabel(
                    "Environment file",
                    detail: "complete, verbatim — secrets stay $(cat …) indirections")
            }
            DisclosureGroup {
                monospacedPane(preview.headerDocument)
            } label: {
                paneLabel(
                    "Scheduler headers",
                    detail: "#SBATCH block per job class (\(preview.headers.count))")
            }
            DisclosureGroup {
                monospacedPane(preview.schedulerCommandDocument)
            } label: {
                paneLabel("Scheduler commands", detail: "binaries this site invokes")
            }
            DisclosureGroup {
                monospacedPane(preview.gpuDocument)
            } label: {
                paneLabel(
                    "GPU vocabulary",
                    detail: preview.gpuVocabulary.isEmpty
                        ? "none emitted"
                        : "\(preview.gpuVocabulary.entries.count) type(s) + VRAM table")
            }
            DisclosureGroup {
                unresolvedPane
            } label: {
                paneLabel(
                    "Unresolved facts",
                    detail: preview.unresolvedFacts.isEmpty
                        ? "none — the profile states everything"
                        : "\(preview.unresolvedFacts.count) fell back to a default")
            }
        }
    }

    @State private var environmentExpanded: Bool = false

    init(
        preview: ClusterSitePreview, paneHeight: CGFloat = 220,
        expandsEnvironment: Bool = true
    ) {
        self.preview = preview
        self.paneHeight = paneHeight
        self.expandsEnvironment = expandsEnvironment
        _environmentExpanded = State(initialValue: expandsEnvironment)
    }

    private func paneLabel(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.callout)
            Text(detail)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Selectable monospaced text in its own scroller: the pane is READ-ONLY
    /// and COPYABLE, which is the whole requirement — an admin must be able to
    /// lift these bytes into a review.
    private func monospacedPane(_ text: String) -> some View {
        ScrollView([.vertical, .horizontal]) {
            Text(text)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
        }
        .frame(maxHeight: paneHeight)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
    }

    /// The unresolved pane reads as a list rather than a document: each row is
    /// a decision a site admin may want to go make, not text to copy.
    @ViewBuilder
    private var unresolvedPane: some View {
        if preview.unresolvedFacts.isEmpty {
            Text("every field the renderer needs is stated by this profile")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(preview.unresolvedFacts, id: \.key) { fact in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(fact.key)
                                .font(.caption.monospaced())
                            Text(fact.detail)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(6)
            }
            .frame(maxHeight: paneHeight)
            .textSelection(.enabled)
        }
    }
}
