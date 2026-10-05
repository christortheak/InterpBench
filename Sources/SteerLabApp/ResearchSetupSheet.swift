import AppKit
import ExperimentKit
import SwiftUI

/// The first-run sheet. Its sentences live in `ResearchSetupCopy`
/// (ExperimentKit), where they are unit-tested to stay free of commands,
/// flags, environment variables, and build instructions — this view only
/// lays them out.
struct ResearchSetupSheet: View {
    @Bindable var model: ResearchSetupModel
    @Bindable var workspace: WorkspaceStore
    /// The three compute choices and the actions behind picking one.
    let compute: ComputeChoiceCoordinator
    let service: ChatService
    let localServer: LocalServerController
    let createWorkspace: () -> Void
    let openWorkspace: () -> Void
    /// Asks where a Demo Workspace's copy goes, copies it, and opens the copy.
    let openDemo: (DemoWorkspace.Entry) throws -> WorkspaceStore.DemoOpening?
    /// Called when the demo sheet closes.
    let demoSheetClosed: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false
    @State private var showingDemos = false

    /// The folder the researcher chose — nil with no workspace yet, and nil
    /// for a developer build standing on its own checkout.
    private var selectedRoot: URL? { workspace.chosenRootURL }

    private var helperTitle: String { model.helperTitle }

    private var planButtonTitle: String {
        if model.clientReady { return "Review Repair Plan" }
        return model.basicClientReady ? "Review Update Plan" : "Review Setup Plan"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(ResearchSetupCopy.title).font(.title2.bold())
            Text(ResearchSetupCopy.introduction)
                .foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    workspaceStep
                    computeStep
                    helperStep
                    beginStep
                }
            }
            if model.busy {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(model.message ?? "Checking setup…").font(.caption)
                    // An installation can be stopped: the installer removes
                    // what it staged and the earlier helper stays in place.
                    if model.installing {
                        Spacer()
                        Button(ResearchSetupCopy.cancelInstallButton) { model.cancelInstall() }
                            .controlSize(.small).disabled(model.cancelling)
                    }
                }
            }
            else if let message = model.message { Text(message).font(.caption).textSelection(.enabled) }
            if let error = model.error {
                VStack(alignment: .leading, spacing: 2) {
                    Text(error).foregroundStyle(.red).font(.caption).textSelection(.enabled)
                    // The repair travels with the reason: a failure that only
                    // says what went wrong leaves the researcher with nothing
                    // to do about it.
                    if let repair = model.errorRepair {
                        Text(repair).font(.caption).textSelection(.enabled)
                    }
                    if let details = model.errorDetails {
                        detailsDisclosure(details)
                    }
                }
            }
            HStack {
                Text(model.authoringReady ? ResearchSetupCopy.readyFooter : ResearchSetupCopy.returnFooter)
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                // Return dismisses only once there is a workspace to return
                // to. Before that, Return creates one (see `workspaceStep`).
                // Only an installation holds the sheet open, and it has its
                // own Cancel; a readiness check or a plan never does.
                Button("Done") { dismiss() }.disabled(model.installing)
                    .keyboardShortcut(selectedRoot == nil ? nil : .defaultAction)
            }
        }
        .padding(24).frame(width: 660, height: 700)
        .interactiveDismissDisabled(model.installing)
        .task(id: selectedRoot) { copied = false; await model.refresh(workspace: selectedRoot) }
        // This sheet is itself a sheet, so the engine setup and the "what
        // runs where" view it can ask for are presented ON it.
        .modifier(
            ComputeSheets(
                compute: compute, service: service, localServer: localServer,
                isActive: true))
    }

    /// The three plainly named choices. Picking one records it for the
    /// workspace and switches the app to it; for the engine on this Mac that
    /// opens its setup, where nothing is installed until it is approved.
    private var computeStep: some View {
        GroupBox(ResearchSetupCopy.computeStepTitle) {
            VStack(alignment: .leading, spacing: 8) {
                // Offered for the folder the researcher chose — not for a
                // developer build standing on its own checkout.
                if selectedRoot != nil {
                    ComputeChoiceList(
                        selection: workspace.isComputeDeclared
                            ? compute.workspaceChoice : nil,
                        choose: { compute.choose($0) })
                    if !workspace.isComputeDeclared {
                        Text(ComputeChoice.undeclaredNote(treatingAs: compute.workspaceChoice))
                            .font(.caption)
                    } else if compute.workspaceChoice == .macFullCapabilities {
                        Text(ComputeChoice.fullCapabilitiesSetup).font(.caption)
                    } else if compute.workspaceChoice == .anotherMachine,
                        compute.cluster.otherMachines.isEmpty
                    {
                        Text(ComputeChoice.connectAnotherMachine).font(.caption)
                    }
                    Button(ComputeGuide.guideButton) { compute.showingGuide = true }
                        .controlSize(.small)
                    Text(ResearchSetupCopy.computeCaption)
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text(ResearchSetupCopy.computeNeedsWorkspace)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .disabled(model.busy)
        }
    }

    private var workspaceStep: some View {
        GroupBox(ResearchSetupCopy.workspaceStepTitle) {
            VStack(alignment: .leading, spacing: 8) {
                Text(selectedRoot?.path ?? ResearchSetupCopy.workspacePrompt)
                    .textSelection(.enabled)
                HStack {
                    if selectedRoot == nil {
                        Button("New Workspace…", action: createWorkspace)
                            .buttonStyle(.borderedProminent)
                            .keyboardShortcut(.defaultAction)
                    } else {
                        Button("New Workspace…", action: createWorkspace)
                    }
                    Button("Open Workspace…", action: openWorkspace)
                    // A worked example, opened as a copy in a folder the
                    // researcher chooses. Offered beside the other two ways
                    // to get a workspace, and listed by what this build
                    // carries — which may be nothing, and the sheet says so.
                    Button(DemoWorkspaceCopy.button) { showingDemos = true }
                }.disabled(model.busy || workspace.isEnvironmentPinned)
                Text(ResearchSetupCopy.workspaceCaption)
                    .font(.caption).foregroundStyle(.secondary)
                Text(ResearchSetupCopy.demoCaption)
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
        .sheet(isPresented: $showingDemos, onDismiss: demoSheetClosed) {
            DemoWorkspaceSheet(demos: DemoWorkspace.available(), open: openDemo)
        }
    }

    private var helperStep: some View {
        GroupBox(ResearchSetupCopy.helperStepTitle) {
            VStack(alignment: .leading, spacing: 8) {
                Label(helperTitle, systemImage: model.clientReady ? "checkmark.circle" : "arrow.down.circle")
                Text(ResearchSetupCopy.helperExplanation)
                // One plain step the researcher can take here. The readiness
                // report's own reason and repair are written for the command
                // line and are not shown in this sheet.
                if let guidance = model.helperGuidance {
                    Text(guidance).font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // The identity failure's own detail — both versions, where
                // the files are, and what the helper said — for whoever is
                // asked to help. Hidden until opened.
                if let failure = model.identityFailure, !model.clientReady {
                    detailsDisclosure(failure.details)
                }
                HStack {
                    Button(planButtonTitle) { Task { await model.preview() } }
                    Button("Check Again") { Task { await model.refresh(workspace: selectedRoot) } }
                }.disabled(model.busy)
                if model.planHash != nil {
                    Text("Install location: " + model.planDestination).font(.caption).textSelection(.enabled)
                    ForEach(model.planActions, id: \.self) { Text("• " + $0).font(.callout) }
                    Text(ResearchSetupCopy.planCaption)
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Approve and Install") { Task { await model.install(workspace: selectedRoot) } }
                        .buttonStyle(.borderedProminent).disabled(model.busy)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// A failure's technical detail, selectable so it can be copied into a
    /// message to whoever is helping.
    private func detailsDisclosure(_ details: String) -> some View {
        DisclosureGroup(ResearchSetupCopy.detailsLabel) {
            Text(details)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.caption)
    }

    private var beginStep: some View {
        GroupBox(ResearchSetupCopy.beginStepTitle) {
            VStack(alignment: .leading, spacing: 8) {
                Text(ResearchSetupCopy.beginExplanation)
                Button(copied ? ResearchSetupCopy.instructionsCopied : ResearchSetupCopy.copyInstructions) {
                    if let handoff = model.handoff {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(handoff, forType: .string)
                        copied = true
                    }
                }.disabled(model.handoff == nil || model.busy)
                Text(ResearchSetupCopy.workInTheApp)
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
