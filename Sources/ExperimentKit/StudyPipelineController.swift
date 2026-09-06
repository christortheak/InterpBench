import Foundation
import Observation

/// Pipeline declaration writes and selection-scoped local/server ledger listings.
@Observable @MainActor
public final class StudyPipelineController {
    public private(set) var pipelineRuns: [ClusterClient.PipelineRunSummary] = []
    public private(set) var localPipelineRuns: [ClusterClient.PipelineRunSummary] = []
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored var presentation = StudyPipelinePresentation()
    public init() {}
    private func note(_ text: String, severity: PanelNotice.Severity) {
        presentation.note(text, severity)
    }
    public func resetSelection() {
        generation = UUID()
        pipelineRuns = []
        localPipelineRuns = []
    }
    func refresh(name: String?, client: ClusterClient?, isCurrent: @escaping @MainActor () -> Bool)
        async
    {
        await refresh(
            name: name, local: { LocalPipelineCatalog.summaries(experiment: $0) },
            remote: client.map { client in
                { name in try await client.pipelineRuns(experiment: name) }
            },
            isCurrent: isCurrent)
    }
    func refresh(
        name: String?,
        local: (String) -> [ClusterClient.PipelineRunSummary],
        remote: ((String) async throws -> [ClusterClient.PipelineRunSummary])?,
        isCurrent: @escaping @MainActor () -> Bool = { true }
    ) async {
        let request = UUID()
        generation = request
        guard let name else {
            resetSelection()
            return
        }
        guard isCurrent() else { return }
        localPipelineRuns = local(name)
        guard let remote else {
            pipelineRuns = []
            return
        }
        do {
            let runs = try await remote(name)
            guard generation == request, isCurrent(), !Task.isCancelled else { return }
            pipelineRuns = runs
        } catch {
            guard generation == request, isCurrent(), !Task.isCancelled else { return }
            pipelineRuns = []
        }
    }

    /// Save the Pipeline Composer's declaration into the manifest (stage 5,
    /// sixth round — the app must AUTHOR the chain it runs, not just submit
    /// it). `nil` removes the block. Draft-only, like every declaration.
    @discardableResult
    public func saveDeclaration(_ draft: PipelineDraft?, reviewed: DraftAuthoringSnapshot) -> DraftAuthoringSnapshot? {
        do {
            let saved = try StudyPipelineAuthoring.save(draft, reviewed: reviewed)
            presentation.refresh()
            note(
                draft == nil
                    ? "pipeline declaration removed"
                    : "pipeline declared — submit it with Run Pipeline",
                severity: .success)
            return saved
        } catch {
            note(
                "Couldn't save the pipeline declaration — the study must "
                    + "still be a draft and its file writable. "
                    + "Details: \(error). Reload and review the declaration before retrying.",
                severity: .error)
            return nil
        }
    }
}

@MainActor
struct StudyPipelinePresentation {
    var note: (String, PanelNotice.Severity) -> Void = { _, _ in }
    var refresh: () -> Void = {}
}
