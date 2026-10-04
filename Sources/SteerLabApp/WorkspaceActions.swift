import AppKit
import ExperimentKit
import SwiftUI

/// Creating and opening a workspace, in one place.
///
/// These actions used to be private to the toolbar's Workspace menu, which
/// was fine while the menu was the only way in. A first launch has no
/// workspace at all, and then Home, every other section, and Research Setup
/// each need the same two buttons — so the panels, the catalog reset, and the
/// failure alert live here and every surface calls the same code.
///
/// Owned by the App (one instance), like the stores it drives.
@MainActor @Observable
final class WorkspaceActions {
    let workspace: WorkspaceStore
    let service: ChatService
    let catalog: SubstrateCatalog

    /// A failure title names what failed; "Workspace" alone is a noun, not a
    /// report (2026-09-06 audit).
    var errorTitle = "Workspace"
    var errorMessage: String?
    /// Research Setup's presentation, shared so Home's welcome and the
    /// Workspace menu open the same sheet.
    var showingResearchSetup = false

    init(workspace: WorkspaceStore, service: ChatService, catalog: SubstrateCatalog) {
        self.workspace = workspace
        self.service = service
        self.catalog = catalog
    }

    /// One place to raise a failure, so every path names what failed and
    /// renders the error's own message (`localizedDescription`, which every
    /// ExperimentKit error type answers through `ErrorMessages`) instead of
    /// its raw Swift description.
    func report(_ title: String, _ error: Error) {
        errorTitle = title
        errorMessage = error.localizedDescription
    }

    func newWorkspace(deferCompute: Bool = false) {
        let panel = NSSavePanel()
        panel.title = "New SteerLab Workspace"
        panel.prompt = "Create"
        panel.nameFieldStringValue = "SteerLab Workspace"
        panel.canCreateDirectories = true
        panel.showsTagField = false
        // Creation is the one moment the answer is never ambiguous, so ask
        // here rather than leaving a new workspace to be inferred later.
        // Defaults to Cluster: real studies compute there and MLX is for toy
        // runs and shakedowns.
        let chooser = ComputeChoiceAccessory(selected: .cluster)
        if !deferCompute { panel.accessoryView = chooser.view }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try workspace.createAndSwitch(to: url, computing: deferCompute ? nil : chooser.selected)
            resetCatalogs()
        } catch {
            report("Could not create the workspace", error)
        }
    }

    func openWorkspace() {
        let panel = NSOpenPanel()
        panel.title = "Open SteerLab Workspace"
        panel.prompt = "Open"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try workspace.switchTo(url)
            resetCatalogs()
        } catch {
            report("Could not open that workspace", error)
        }
    }

    /// The existing refresh entry points, called once after a switch so
    /// panels drop state scanned from the previous root. Anything a panel
    /// caches outside these paths refreshes on its next interaction.
    func resetCatalogs() {
        // Section-specific DISPLAY state first: the catalog refreshes below
        // re-scan lists, but none of them retired the viewer's selection, so
        // after a switch the Results viewer still showed the previous
        // workspace's run (Finder button and all) and Analysis still showed
        // its cosine tables. Each viewer has an empty state; it just needed
        // its selection dropped. The workspace-wide Activity log is kept.
        service.resetSectionViewers()
        service.experiments.refresh()
        service.datasetInventory.refresh()
        service.concepts.refreshConceptList()
        service.concepts.refreshStaleness()
        service.concepts.refreshReaderTemplates()
        service.concepts.refreshReaderArtifacts()
        service.fineTuning.refresh()
        service.refreshVectors()
        service.refreshNeutralCorpora()
        service.refreshNeutralPCBases()
        catalog.refreshLocalVectors()
    }

    // Same-machine server auto-switching is NOT a view concern: it lives on
    // `ClusterConnectionStore.synchronizeServerToLocalWorkspace()`, triggered
    // once at the workspace-root-change seam (`WorkspaceStore.onRootChange`,
    // wired in `SteerLabApp`), so every root-change path — including ones
    // this type never sees — gets the same serialized, surfaced behavior.
}
