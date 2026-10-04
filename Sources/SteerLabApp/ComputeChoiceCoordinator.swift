import ExperimentKit
import SwiftUI

/// Acting on one of the three compute choices, in one place.
///
/// The choice is offered in the new-workspace panel, Research Setup, the
/// Workspace menu, the Compute menu, and wherever a method needs the Python
/// engine. Each of those used to be a different control over a different
/// piece of state; they all call this now, so "This Mac, full capabilities"
/// means the same thing — and leads to the same setup — from every one.
///
/// Two verbs, deliberately:
///
/// - `choose` DECLARES the choice for the workspace and switches the app to
///   it. This is the Workspace menu, Research Setup, and a new workspace.
/// - `use` switches the app only. This is the Compute menu, where a
///   researcher may be trying another engine for an afternoon; it declares
///   the choice only for a workspace that has declared nothing yet and whose
///   own runs do not point the other way
///   (`ComputeChoice.declarationAfterUsing`). The declaration decides whose
///   vectors and evidence count for the workspace's studies, and is never
///   rewritten as a side effect.
///
/// Nothing here installs anything. "This Mac, full capabilities" checks what
/// is already set up — touching nothing — and then either connects to the
/// engine or opens the existing setup sheet, where the researcher presses
/// Set Up.
///
/// Owned by the App (one instance), like the stores it drives.
@MainActor @Observable
final class ComputeChoiceCoordinator {
    let workspace: WorkspaceStore
    let cluster: ClusterConnectionStore
    let service: ChatService
    let localEngine: LocalEngineProvisioner
    let actions: WorkspaceActions

    /// The local-engine setup sheet. One flag for every entry point, so the
    /// sheet exists once and reopening shows where the setup got to.
    var showingEngineSetup = false
    /// The "what runs where" view.
    var showingGuide = false
    /// The cluster setup wizard (presented by the connection menu's view).
    var showingClusterWizard = false

    /// True from the moment the researcher asks for the engine on this Mac
    /// until the app is connected to it — so that when a setup they started
    /// finishes, the app switches over without a second click.
    private(set) var wantsThisMacEngine = false

    init(
        workspace: WorkspaceStore, cluster: ClusterConnectionStore,
        service: ChatService, localEngine: LocalEngineProvisioner,
        actions: WorkspaceActions
    ) {
        self.workspace = workspace
        self.cluster = cluster
        self.service = service
        self.localEngine = localEngine
        self.actions = actions
    }

    // MARK: What is set, and what is in use

    /// The choice the app is using right now.
    var inUse: ComputeChoice { cluster.activeComputeChoice }

    /// The choice this workspace is set to.
    var workspaceChoice: ComputeChoice {
        workspace.computeChoice(
            activeEngineIsThisMac: cluster.activeServerSharesLocalFilesystem)
    }

    /// The sentence for "the workspace is set to one engine and the app is
    /// using the other", or nil when they agree.
    var mismatchNote: String? {
        guard workspace.hasWorkspace else { return nil }
        return ComputeChoice.mismatchNote(workspace: workspaceChoice, inUse: inUse)
    }

    // MARK: Choosing

    /// Declare `choice` for this workspace and switch the app to it.
    func choose(_ choice: ComputeChoice) {
        guard workspace.hasWorkspace else {
            actions.errorTitle = "Create or open a workspace first"
            actions.errorMessage = WorkspaceRoot.noWorkspaceReason
            return
        }
        do {
            try workspace.declareComputeChoice(choice)
        } catch {
            actions.report("Could not record where this workspace runs", error)
            return
        }
        switchTo(choice)
    }

    /// A workspace has just been created with `choice` already declared in
    /// it: switch the app to match. For the engine on this Mac that opens
    /// its setup, where nothing is installed until the researcher approves.
    func applyToNewWorkspace(_ choice: ComputeChoice) {
        switchTo(choice)
    }

    /// Switch the app to `choice` without re-declaring a workspace that has
    /// already declared one.
    func use(_ choice: ComputeChoice) {
        // Only for a folder the researcher chose: a developer build standing
        // on its own checkout is never written to as a side effect.
        if let root = workspace.chosenRootURL,
            let toDeclare = ComputeChoice.declarationAfterUsing(
                choice,
                declared: WorkspaceCompute.declaredChoice(root: root),
                inferredBinding: WorkspaceCompute.inferred(root: root))
        {
            do {
                try workspace.declareComputeChoice(toDeclare)
            } catch {
                actions.report("Could not record where this workspace runs", error)
            }
        }
        switchTo(choice)
    }

    private func switchTo(_ choice: ComputeChoice) {
        switch choice {
        case .macQuickStart:
            wantsThisMacEngine = false
            cluster.activeWorkspace = .local
        case .macFullCapabilities:
            guard workspace.hasWorkspace else {
                // The engine on this Mac serves a workspace; without one
                // there is nothing for it to serve.
                actions.errorTitle = "Create or open a workspace first"
                actions.errorMessage =
                    "The Python engine on this Mac works inside a workspace. "
                    + "Create or open one, then choose "
                    + "\(ComputeChoice.macFullCapabilities.title) again."
                return
            }
            wantsThisMacEngine = true
            Task { await prepareThisMacEngine() }
        case .anotherMachine:
            wantsThisMacEngine = false
            guard let machine = preferredOtherMachine else {
                // Nothing to switch to yet. Say where one is added rather
                // than opening a form the researcher did not ask for.
                // Research Setup says the same sentence in place, under the
                // choice — an alert on the window behind it would wait.
                if !actions.showingResearchSetup {
                    actions.errorTitle = "Connect another machine"
                    actions.errorMessage = ComputeChoice.connectAnotherMachine
                }
                return
            }
            useMachine(machine)
        }
    }

    /// Switch the app to one saved machine and connect.
    func useMachine(_ machine: ClusterConnectionStore.ServerEntry) {
        wantsThisMacEngine = false
        cluster.activeWorkspace = .server(machine.id)
        Task { await service.connectCluster() }
    }

    /// The other machine to switch to: the one in use if it is one, else the
    /// first saved.
    private var preferredOtherMachine: ClusterConnectionStore.ServerEntry? {
        let machines = cluster.otherMachines
        if let active = cluster.activeServer,
            machines.contains(where: { $0.id == active.id })
        {
            return active
        }
        return machines.first
    }

    // MARK: The engine on this Mac

    /// Look at what is already set up — touching nothing — then connect if
    /// the engine is running, or open its setup sheet if it is not. A setup
    /// already in flight just gets its sheet back.
    private func prepareThisMacEngine() async {
        if localEngine.phase.isRunning {
            showingEngineSetup = true
            return
        }
        await localEngine.refreshPlan()
        // The researcher may have chosen something else while that ran, or
        // the engine's own state change may already have connected.
        guard wantsThisMacEngine else { return }
        if case .ready = localEngine.phase {
            connectToThisMacEngine()
        } else {
            showingEngineSetup = true
        }
    }

    /// Called when the engine's setup state changes. If the researcher asked
    /// for the engine on this Mac and it has just become ready, switch to it.
    func engineStateChanged() {
        guard wantsThisMacEngine, case .ready = localEngine.phase else { return }
        connectToThisMacEngine()
    }

    private func connectToThisMacEngine() {
        wantsThisMacEngine = false
        let hostLabel = "127.0.0.1:\(localEngine.port)"
        let entry = cluster.servers.first { $0.hostLabel == hostLabel }
            ?? cluster.addServer(
                name: ClusterConnectionStore.thisMacEngineName,
                urlString: "http://\(hostLabel)")
        cluster.activeWorkspace = .server(entry.id)
        Task {
            await service.connectCluster()
            // Point the engine at THIS workspace if it was serving another
            // (same-machine servers follow the app's workspace).
            cluster.synchronizeServerToLocalWorkspace()
        }
    }
}
