import Foundation

/// Compatibility entry points for the currently selected workspace. New callers
/// that already have a workspace should retain a StudyResultRepository instead.
public enum StudyResultStore {
    public static func list(experimentName: String) -> [StudyRunListItem] {
        StudyResultRepository(workspaceRoot: ExperimentStore.workspaceRoot)
            .list(experimentName: experimentName)
    }

    public static func detail(for item: StudyRunListItem) -> StudyRunDetail {
        StudyResultRepository(workspaceRoot: ExperimentStore.workspaceRoot)
            .detail(for: item)
    }
}
