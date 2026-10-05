import Foundation

/// How the app's sidebar is grouped: the basics a first study uses, and the
/// Advanced methods beside them.
///
/// The sections themselves are the app's `WorkbenchSection` (whose raw values
/// these names are) and keep their identities; this type only says which
/// group each one is listed under, so the grouping can be tested here. A
/// section added to the app without a place in this list is still shown, at
/// the end of the basics, and `WorkbenchNavigationTests` fails until it is
/// placed.
///
/// Advanced is a heading, not a hiding place: it is always listed, one click
/// away, and no setting removes it.
public enum WorkbenchNavigation {

    public enum Group: String, CaseIterable, Sendable, Identifiable {
        case basics = "Basics"
        case advanced = "Advanced"

        public var id: String { rawValue }

        /// Shown as the group's help in the sidebar.
        public var help: String {
            switch self {
            case .basics:
                "everything a study needs, from the workspace to exported results"
            case .advanced:
                "methods beyond a first study: probes, scenarios several agents "
                    + "take turns in, and how vectors relate to each other"
            }
        }
    }

    /// Each group's sections, by their sidebar names, in sidebar order.
    public static let sections: [(group: Group, names: [String])] = [
        (.basics, ["Home", "Playground", "Data", "Agents", "Templates", "Studies", "Results", "Compute"]),
        (.advanced, ["Probes", "Multi-Agent", "Analysis"]),
    ]

    /// The group a section is listed under, or nil for a name this list does
    /// not place.
    public static func group(ofSection name: String) -> Group? {
        sections.first { $0.names.contains(name) }?.group
    }

    /// The Data section's tools, by their names, split the same way: the two
    /// a first study uses, and the two expert builders.
    public static let basicDataTools = ["Inventory", "Concepts & Vectors"]
    public static let advancedDataTools = ["Adapter Training", "OptVec"]

    public static func isAdvancedDataTool(_ name: String) -> Bool {
        advancedDataTools.contains(name)
    }

    /// The label beside the Data section's advanced tools.
    public static let advancedDataToolsLabel = "Advanced"
}
