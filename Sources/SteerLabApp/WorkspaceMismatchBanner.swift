import ExperimentKit
import SwiftUI

/// Prominent standing banner for a CONFIRMED workspace mismatch: the app's
/// selected data workspace is not the tree the active server serves, so
/// server-scoped panels (agents, optimization runs, robustness targets,
/// vector catalogs) are showing — and server actions write to — the SERVER's
/// own workspace. Rendered at the top of every server-scoped panel; the text
/// is the one rule (`WorkspaceScoping.workspaceMismatchBanner` via the
/// store), never per-view prose. Renders nothing when paired/unknown/local.
struct WorkspaceMismatchBanner: View {
    let cluster: ClusterConnectionStore
    /// Repointing a serving root is a server round trip with no indicator of
    /// its own here; without this flag a second click queued a second switch
    /// (2026-09-06 audit: server-route actions need a busy state and a
    /// re-entry guard).
    @State private var isSwitching = false

    var body: some View {
        if let message = cluster.workspaceMismatchBanner {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Label {
                    Text(message)
                        .font(.callout.weight(.medium))
                        .textSelection(.enabled)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                }
                Spacer(minLength: 0)
                // One-click repair when the server supports runtime workspace
                // switching (capability-gated; older servers keep the
                // text-only banner). The affordance rule is the tested
                // `WorkspaceScoping.workspaceSwitchAffordance`: a same-machine
                // server is offered the app's own workspace path; a remote
                // server is offered only SERVER-side roots — a Mac path is
                // never sent across a tunnel.
                switchAffordance
            }
            .foregroundStyle(.orange)
            .padding(.vertical, 6)
            .padding(.horizontal, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.orange.opacity(0.12)))
            .help(
                "the server's serving root (serve --root) is authoritative for "
                    + "everything listed under this server workspace; the app's "
                    + "selected workspace is a different folder — use Workspace → "
                    + "Open Workspace to switch to the server's root, point the "
                    + "server at this workspace (button, when supported), or "
                    + "restart the server pointing at yours")
        }
    }

    @ViewBuilder
    private var switchAffordance: some View {
        switch cluster.workspaceSwitchAffordance {
        case .pointServerAtLocalWorkspace(let localRoot):
            Button(isSwitching ? "Repointing…" : "Point server at this workspace") {
                Task { await switchRoot(to: localRoot) }
            }
            .disabled(isSwitching)
            .help(
                "the server runs on this Mac — repoint its serving root at "
                    + "\(localRoot) (no restart; refused while server jobs "
                    + "are running)")
        case .offerServerSideRoots(let roots):
            Menu(isSwitching ? "Repointing…" : "Point server at…") {
                ForEach(roots, id: \.self) { root in
                    Button(root) { Task { await switchRoot(to: root) } }
                        .help(
                            "make \(root) this server's serving root — no "
                                + "restart; refused while server jobs are running")
                }
            }
            .fixedSize()
            .disabled(isSwitching)
            .help(
                "repoint the server's serving root at one of its OWN known "
                    + "workspace roots (site profile + recents) — the app's "
                    + "local folder is not offered because this server is on "
                    + "another machine")
        case .unavailable:
            EmptyView()
        }
    }

    /// One in flight at a time. The store surfaces the outcome (the banner
    /// itself disappears on success; a refusal lands in the status line).
    private func switchRoot(to root: String) async {
        guard !isSwitching else { return }
        isSwitching = true
        defer { isSwitching = false }
        await cluster.switchServerWorkspace(to: root)
    }
}
