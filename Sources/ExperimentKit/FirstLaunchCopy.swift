import Foundation

/// What the app says before a workspace exists, for a researcher with no
/// machine-learning background. Kept here, beside the state it describes
/// (`WorkspaceRoot.Source.none`), so the wording is unit-tested: it must never
/// show a path, name a command, or point at the source tree.
public enum FirstLaunchCopy {
    public static let welcomeTitle = "Welcome to SteerLab"

    /// Home, with no workspace yet.
    public static let welcomeBody =
        "SteerLab keeps each project in a workspace: one folder on this Mac "
        + "that holds your prompts, study designs, and results. Create a "
        + "workspace to begin, or open one you already have."

    /// Every other section, with no workspace yet.
    public static func sectionPrompt(section: String) -> String {
        "\(section) works inside a workspace: the folder that holds your "
            + "prompts, study designs, and results. Create a workspace to "
            + "begin, or open one you already have."
    }

    public static let createButton = "Create a Workspace…"
    public static let openButton = "Open a Workspace…"
    public static let researchSetupButton = "Research Setup…"

    public static let afterwards =
        "Nothing is downloaded when you create a workspace. Research Setup "
        + "walks through the remaining steps, and you can return to it at any "
        + "time from the Workspace menu."

    /// The right-hand pane, with no workspace yet.
    public static let viewerPlaceholder =
        "Activity, results, and chat appear here once a workspace is open."

    /// The toolbar's Workspace menu, with no workspace yet.
    public static let menuHelp =
        "No workspace yet. A workspace is the folder that holds your "
        + "prompts, study designs, and results. Choose New Workspace… or "
        + "Open Workspace… to begin."
}
