import ExperimentKit
import SwiftUI

/// What a section shows before any workspace exists.
///
/// Home gets the welcome; every other section gets the same prompt with its
/// own name in it. Either way the section's real view is never built, so
/// nothing scans, reads, or writes a workspace that is not there — and no
/// path is shown, because there is none to show.
///
/// macOS 27 layout rule (the split-view minimum crash): this sits in a
/// split-view column, so the text lives in a `ScrollView` and the column's
/// incompressible minimum never depends on how the sentences wrap.
struct NoWorkspaceView: View {
    let actions: WorkspaceActions
    let section: WorkbenchSection
    @State private var showingDemos = false

    private var isHome: Bool { section == .home }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text(isHome ? FirstLaunchCopy.welcomeTitle : section.rawValue)
                    .font(.title2.bold())
                Text(
                    isHome
                        ? FirstLaunchCopy.welcomeBody
                        : FirstLaunchCopy.sectionPrompt(section: section.rawValue))
                    .fixedSize(horizontal: false, vertical: true)
                // Side by side where they fit, stacked at Home's 420 pt
                // floor where three do not.
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 10) { workspaceButtons }
                    VStack(alignment: .leading, spacing: 8) { workspaceButtons }
                }
                if isHome {
                    Text(FirstLaunchCopy.afterwards)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button(FirstLaunchCopy.researchSetupButton) {
                        actions.showingResearchSetup = true
                    }
                    .controlSize(.small)
                    .help(
                        "the three first steps in one place: choose a "
                            + "workspace, set up the study-design helper, and "
                            + "begin with your question")
                    // The whole path to a first result, before any of it is
                    // done. A preview only: the ways in are the buttons
                    // above, and the steps get their own buttons on Home
                    // once a workspace is open.
                    Divider().padding(.vertical, 4)
                    Text(FirstStudyChecklist.title).font(.headline)
                    FirstStudyChecklistRows(
                        items: FirstStudyChecklist.items(.noWorkspace),
                        hasWorkspace: false, showsButtons: false, perform: { _ in })
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .sheet(isPresented: $showingDemos, onDismiss: { actions.demoSheetClosed() }) {
            DemoWorkspaceSheet(
                demos: DemoWorkspace.available(), open: { try actions.openDemoWorkspace($0) })
        }
    }

    @ViewBuilder
    private var workspaceButtons: some View {
        Button(FirstLaunchCopy.createButton) { actions.newWorkspace() }
            .buttonStyle(.borderedProminent)
            .help(
                "choose a name and a place for a new workspace "
                    + "folder; SteerLab fills it with its starter "
                    + "files and opens it")
        Button(FirstLaunchCopy.openButton) { actions.openWorkspace() }
            .help("choose a workspace folder you already have")
        if isHome {
            // The third way in, on Home (2026-10 release review): a copy of
            // a finished study, the same sheet Research Setup opens.
            Button(DemoWorkspaceCopy.button) { showingDemos = true }
                .disabled(actions.workspace.isEnvironmentPinned)
                .help(
                    actions.workspace.isEnvironmentPinned
                        ? DemoWorkspaceCopy.unavailableWhilePinned
                        : "open a copy of a finished study, with a draft to run, "
                            + "in a folder you choose")
        }
    }
}

/// The right-hand pane before any workspace exists: one quiet sentence, in
/// the same compressible container as the prompt beside it.
struct NoWorkspaceViewerPlaceholder: View {
    var body: some View {
        ScrollView {
            Text(FirstLaunchCopy.viewerPlaceholder)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
