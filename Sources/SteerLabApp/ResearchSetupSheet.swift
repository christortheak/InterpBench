import AppKit
import ExperimentKit
import SwiftUI

struct ResearchSetupSheet: View {
    @Bindable var model: ResearchSetupModel
    @Bindable var workspace: WorkspaceStore
    let createWorkspace: () -> Void
    let openWorkspace: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    private var selectedRoot: URL? { workspace.isLegacyRepoRoot ? nil : workspace.rootURL }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Start your research workspace").font(.title2.bold())
            Text("Bring a research question. SteerLab and your agent help turn it into a study you can review, run and inspect.")
                .foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    GroupBox("1. Choose where your study lives") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(selectedRoot?.path ?? "Choose a local folder for your prompts, study designs and results.")
                                .textSelection(.enabled)
                            HStack {
                                Button("New Workspace…", action: createWorkspace)
                                Button("Open Workspace…", action: openWorkspace)
                            }.disabled(model.busy || workspace.isEnvironmentPinned)
                            Text("The same workspace can be opened by the app and your agent. Execution copies can be sent to a server later.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    GroupBox("2. Prepare study authoring") {
                        VStack(alignment: .leading, spacing: 8) {
                            Label(model.clientReady ? "Client ready" : model.basicClientReady ? "Client update needed for corpus tools" : "Client setup needed", systemImage: model.clientReady ? "checkmark.circle" : "arrow.down.circle")
                            Text("This CPU setup supplies Python and the tools for study interviews, evidence import, and corpus preparation. Model downloads and server setup are separate.")
                            if model.basicClientReady && !model.clientReady {
                                Text("Basic study authoring is available. Update the client to enable all corpus tools.").font(.caption)
                            }
                            HStack {
                                Button(model.clientReady ? "Review Repair Plan" : model.basicClientReady ? "Review Update Plan" : "Review Setup Plan") { Task { await model.preview() } }
                                Button("Check Again") { Task { await model.refresh(workspace: selectedRoot) } }
                            }.disabled(model.busy)
                            if model.planHash != nil {
                                Text("Install location: " + model.planDestination).font(.caption).textSelection(.enabled)
                                ForEach(model.planActions, id: \.self) { Text("• " + $0).font(.callout) }
                                Text("Requires internet access. Existing managed environments are retained; no study files are changed.")
                                    .font(.caption).foregroundStyle(.secondary)
                                Button("Approve and Install Client") { Task { await model.install(workspace: selectedRoot) } }
                                    .buttonStyle(.borderedProminent).disabled(model.busy)
                            }
                            if !model.clientReady, case .string(let repair) = model.readiness["repairAction"] {
                                Text(repair).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            }
                            if !model.clientReady, case .string(let reason) = model.readiness["reason"] {
                                Text(reason).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    GroupBox("3. Begin with your question") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Give your agent the workspace instructions and describe what you want to understand. It can discover the methods, help prepare datasets and propose a study for your review.")
                            Button(copied ? "Agent Handoff Copied" : "Copy Agent Handoff") {
                                if let handoff = model.handoff {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(handoff, forType: .string)
                                    copied = true
                                }
                            }.disabled(model.handoff == nil || model.busy)
                            Text("To work directly in the app, open Studies or Templates after closing this screen. When you are ready to run, use Compute to select and check local or remote hardware, then prepare your model.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            if model.busy { HStack { ProgressView().controlSize(.small); Text(model.message ?? "Checking setup…").font(.caption) } }
            else if let message = model.message { Text(message).font(.caption).textSelection(.enabled) }
            if let error = model.error { Text(error).foregroundStyle(.red).font(.caption).textSelection(.enabled) }
            HStack {
                Text(model.authoringReady ? "Ready to author studies" : "You can return here from Workspace → Research Setup.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Done") { dismiss() }.disabled(model.busy).keyboardShortcut(.defaultAction)
            }
        }
        .padding(24).frame(width: 660, height: 700)
        .interactiveDismissDisabled(model.busy)
        .task(id: selectedRoot) { copied = false; await model.refresh(workspace: selectedRoot) }
    }
}
