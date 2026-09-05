import Foundation

/// The model choices resolved by the workspace host before creating a draft.
public struct StudyCreationContext: Sendable {
    public let workspaceDefaultModelID: String
    public let modelOptions: [String]

    public init(workspaceDefaultModelID: String, modelOptions: [String]) {
        self.workspaceDefaultModelID = workspaceDefaultModelID
        self.modelOptions = modelOptions
    }
}

/// Coordinator effects that are outside study/design management.
@MainActor
struct StudyManagementPresentation {
    var note: (String, PanelNotice.Severity) -> Void = { _, _ in }
    var selectionChanged: () -> Void = {}
    var refreshed: () -> Void = {}
}
