import Foundation

extension ChatService {
    /// Re-read the workspace lists a coding assistant or a command line may
    /// have changed while the researcher was in another window: studies
    /// (including ones that cannot be read), templates, the selected study's
    /// runs, the agent library, datasets, and concept names.
    ///
    /// Called when the app becomes active and by the Refresh controls. It
    /// discards nothing a researcher typed: the study editor keeps its own
    /// retained review and answers a changed file with "Discard edits and
    /// reload", the agent editor reloads only when the selection changes, and
    /// the two slow scans (agents, datasets) run off the main actor.
    public func reloadWorkspaceListsFromDisk() {
        experiments.refresh()
        fineTuning.refreshAgentLibraryAsync()
        datasetInventory.refresh()
        concepts.refreshConceptList()
    }
}
