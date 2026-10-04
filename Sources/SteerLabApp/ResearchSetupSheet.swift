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
    let createWorkspace: () -> Void
    let openWorkspace: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    /// The folder the researcher chose — nil with no workspace yet, and nil
    /// for a developer build standing on its own checkout.
    private var selectedRoot: URL? { workspace.chosenRootURL }

    private var helperTitle: String {
        if model.clientReady { return ResearchSetupCopy.helperReady }
        return model.basicClientReady
            ? ResearchSetupCopy.helperUpdateTitle : ResearchSetupCopy.helperSetupTitle
    }

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
                    helperStep
                    beginStep
                }
            }
            if model.busy { HStack { ProgressView().controlSize(.small); Text(model.message ?? "Checking setup…").font(.caption) } }
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
                }
            }
            HStack {
                Text(model.authoringReady ? ResearchSetupCopy.readyFooter : ResearchSetupCopy.returnFooter)
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                // Return dismisses only once there is a workspace to return
                // to. Before that, Return creates one (see `workspaceStep`).
                Button("Done") { dismiss() }.disabled(model.busy)
                    .keyboardShortcut(selectedRoot == nil ? nil : .defaultAction)
            }
        }
        .padding(24).frame(width: 660, height: 700)
        .interactiveDismissDisabled(model.busy)
        .task(id: selectedRoot) { copied = false; await model.refresh(workspace: selectedRoot) }
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
                }.disabled(model.busy || workspace.isEnvironmentPinned)
                Text(ResearchSetupCopy.workspaceCaption)
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
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
