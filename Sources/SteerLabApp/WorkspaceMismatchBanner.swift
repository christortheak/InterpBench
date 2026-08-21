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
            Button("Point server at this workspace") {
                Task { await cluster.switchServerWorkspace(to: localRoot) }
            }
            .help(
                "the server runs on this Mac — repoint its serving root at "
                    + "\(localRoot) (no restart; refused while server jobs "
                    + "are running)")
        case .offerServerSideRoots(let roots):
            Menu("Point server at…") {
                ForEach(roots, id: \.self) { root in
                    Button(root) {
                        Task { await cluster.switchServerWorkspace(to: root) }
                    }
                }
            }
            .fixedSize()
            .help(
                "repoint the server's serving root at one of its OWN known "
                    + "workspace roots (site profile + recents) — the app's "
                    + "local folder is not offered because this server is on "
                    + "another machine")
        case .unavailable:
            EmptyView()
        }
    }
}
