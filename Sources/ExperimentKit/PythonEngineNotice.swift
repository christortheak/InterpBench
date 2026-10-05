import Foundation

/// What a place that needs the Python engine shows, for the compute the app
/// is using right now.
///
/// Wave 1 put "switch to full capabilities" at two sites; a dozen more still
/// refused or only instructed ("Needs a server connection", "Select Python
/// Compute", "switch Compute to …"), and the instructions disagreed about
/// what to do. Every one of them now asks this type, so the decision and its
/// wording are the same everywhere and unit-tested once:
///
/// - on the quick start, offer the switch (a sentence and a button);
/// - on the Python engine but not connected to it, say so and offer to
///   connect — switching would be no help;
/// - on a connected Python engine, show nothing.
public enum PythonEngineNotice: Equatable, Sendable {
    /// The app is on the quick start: the sentence and the switch.
    case offerSwitch
    /// The app is set to the Python engine and is not connected to it.
    case connect
    /// The Python engine is in use and connected: nothing to say.
    case none

    public init(inUse: ComputeChoice, connected: Bool) {
        if !inUse.runsEveryMethod {
            self = .offerSwitch
        } else {
            self = connected ? .none : .connect
        }
    }

    // MARK: Words

    /// Where a method needs the Python engine and the app is set to it but
    /// not connected. `subject` is what needs it, as `needsPythonEngine`
    /// takes it. `buttonHere` is false for a status line, which has no
    /// Connect button beside it.
    public static func notConnected(
        _ subject: String, plural: Bool, buttonHere: Bool = true
    ) -> String {
        "\(subject) \(plural ? "need" : "needs") the Python engine, and the "
            + "app is not connected to it right now. "
            + (buttonHere
                ? "Choose Connect here, or in the connection menu in the toolbar."
                : "Choose Connect in the connection menu in the toolbar, then "
                    + "try again.")
    }

    public static let connectButton = "Connect"

    /// The two places the Python engine runs, by the names the app shows.
    public static let pythonChoices =
        "\u{201C}\(ComputeChoice.macFullCapabilities.title)\u{201D} or "
        + "\u{201C}\(ComputeChoice.anotherMachine.title)\u{201D}"

    /// For a sentence with no room for a button — a status line, a run-start
    /// message, a tooltip: where the Python engine is chosen. `what` finishes
    /// "The Python engine …" ("records probe measurements").
    public static func whereToRun(_ what: String) -> String {
        "The Python engine \(what): in the Workspace menu, choose "
            + pythonChoices + " as where this workspace runs studies, and "
            + "run the study there."
    }

    /// For a status note where an action that runs on the Python engine was
    /// asked for while the app is on the quick start. It replaces "switch
    /// the substrate selector first", which named a control the app no
    /// longer has.
    public static func switchFirst(_ subject: String, plural: Bool) -> String {
        "\(subject) \(plural ? "need" : "needs") the Python engine, and the "
            + "app is using \u{201C}\(ComputeChoice.macQuickStart.title)\u{201D} "
            + "right now. Choose " + pythonChoices + " in the Compute menu "
            + "first."
    }

    /// A short form for a tooltip or a one-line status slot.
    public static func needsPythonEngineBriefly(_ subject: String, plural: Bool) -> String {
        "\(subject) \(plural ? "need" : "needs") the Python engine: "
            + pythonChoices + "."
    }

    /// The tooltip for a control that waits on a connection to the Python
    /// engine.
    public static let notConnectedBriefly =
        "the Python engine is not connected — choose Connect in the connection "
        + "menu in the toolbar"
}
