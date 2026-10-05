import AppKit
import ExperimentKit
import SwiftUI

/// "Open Demo Workspace…" from Research Setup: the Demo Workspaces this build
/// carries, one row each with what it needs and what it shows, and a button
/// that asks where to put the copy.
///
/// Its sentences live in `DemoWorkspaceCopy` and `DemoWorkspace` (ExperimentKit),
/// where they are unit-tested; this view only lays them out. A build that
/// carries no demo says so plainly and offers nothing else.
struct DemoWorkspaceSheet: View {
    /// The demos this build carries, read once when the sheet opens.
    let demos: [DemoWorkspace.Entry]
    /// Asks for a folder, copies, and opens. Nil when the panel is cancelled.
    let open: (DemoWorkspace.Entry) throws -> WorkspaceStore.DemoOpening?
    @Environment(\.dismiss) private var dismiss
    @State private var opening: WorkspaceStore.DemoOpening?
    @State private var failure: DemoWorkspace.Refusal?
    @State private var otherFailure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(DemoWorkspaceCopy.title).font(.title2.bold())
            if let opening {
                opened(opening)
            } else {
                Text(DemoWorkspaceCopy.introduction).foregroundStyle(.secondary)
                if demos.isEmpty {
                    Text(DemoWorkspaceCopy.noneCarried)
                } else {
                    ForEach(demos) { demo in row(demo) }
                }
                if let failure {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(failure.reason).foregroundStyle(.red)
                        Text(failure.repair)
                    }
                    .font(.caption).textSelection(.enabled)
                } else if let otherFailure {
                    Text(otherFailure).foregroundStyle(.red).font(.caption)
                        .textSelection(.enabled)
                }
            }
            HStack {
                Spacer()
                Button(opening == nil ? "Cancel" : "Done") { dismiss() }
                    .keyboardShortcut(opening == nil ? .cancelAction : .defaultAction)
            }
        }
        .padding(24).frame(width: 560)
    }

    /// One demo: its name and where it runs, what it needs, what it shows.
    private func row(_ demo: DemoWorkspace.Entry) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                Text(DemoWorkspaceCopy.rowTitle(demo.description)).font(.headline)
                Text(DemoWorkspace.showsLine(demo.description))
                Text(DemoWorkspace.needsLine(demo.description))
                    .font(.caption).foregroundStyle(.secondary)
                Button(DemoWorkspaceCopy.openButton) { choose(demo) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// After the copy is open: where it is, and where to start reading.
    private func opened(_ opening: WorkspaceStore.DemoOpening) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(opening.opened.root.path).textSelection(.enabled)
            Text(DemoWorkspaceCopy.opened)
            if !opening.unverified.isEmpty {
                Text(DemoWorkspaceCopy.unverified(opening.unverified))
                    .foregroundStyle(.red).textSelection(.enabled)
            }
            Button(DemoWorkspaceCopy.showGuide) {
                NSWorkspace.shared.open(opening.opened.readme)
            }
        }
    }

    private func choose(_ demo: DemoWorkspace.Entry) {
        failure = nil
        otherFailure = nil
        do {
            opening = try open(demo)
        } catch let refusal as DemoWorkspace.Refusal {
            failure = refusal
        } catch {
            otherFailure = error.localizedDescription
        }
    }
}
