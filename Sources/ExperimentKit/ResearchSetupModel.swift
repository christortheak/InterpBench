import Foundation
import Observation

/// First-run presentation state. Scientific owners and the installer retain
/// all validation and mutation responsibilities.
@MainActor @Observable public final class ResearchSetupModel {
    public private(set) var readiness: [String: JSONValue] = [:]
    public private(set) var plan: [String: JSONValue] = [:]
    public private(set) var busy = false
    public private(set) var message: String?
    public private(set) var error: String?
    /// What to do about `error`, in the app's own words. A plan or install
    /// failure carries its repair; it is kept beside the reason, not dropped.
    public private(set) var errorRepair: String?
    public private(set) var handoff: String?
    public init() {}

    public var clientReady: Bool { readiness["clientReady"] == .bool(true) }
    public var basicClientReady: Bool { readiness["basicClientReady"] == .bool(true) }
    public var authoringReady: Bool { readiness["authoringReady"] == .bool(true) }
    public var planHash: String? { if case .string(let value) = plan["planSHA256"] { value } else { nil } }
    public var planDestination: String { if case .string(let value) = plan["runtime"] { value } else { "" } }
    public var planActions: [String] {
        guard case .array(let values) = plan["actions"] else { return [] }
        return values.compactMap { if case .string(let text) = $0 { text } else { nil } }
    }

    /// What the sheet says while the helper is not ready, or nil once it is.
    /// The readiness report's own `reason` and `repairAction` are written for
    /// the command line (verbs, an environment variable, a rebuild); the app
    /// says the same thing as one step the researcher can take in this sheet.
    public var helperGuidance: String? {
        if clientReady { return nil }
        return basicClientReady
            ? ResearchSetupCopy.helperUpdateNeeded : ResearchSetupCopy.helperNeeded
    }

    /// Whether Research Setup opens by itself at launch.
    ///
    /// With no workspace yet it opens EVERY launch — the "already shown" flag
    /// is not consulted, because a newcomer who dismissed the sheet once must
    /// not be left on a Home with nothing to act on. Once a workspace exists
    /// it opens at most once, and only while study design is not ready.
    public nonisolated static func opensAtLaunch(
        hasWorkspace: Bool, alreadyPresented: Bool, authoringReady: Bool
    ) -> Bool {
        guard hasWorkspace else { return true }
        return !alreadyPresented && !authoringReady
    }

    public func refresh(workspace: URL?) async {
        guard !busy else { return }
        busy = true
        readiness = await ClientSetup.inspect(workspace: workspace)
        handoff = workspace.flatMap { root in
            guard let value = try? WorkspaceBootstrap.handoff(root),
                  let bytes = try? JSONEncoder().encode(JSONValue.object(value)) else { return nil }
            return String(decoding: bytes, as: UTF8.self)
        }
        busy = false
    }

    public func preview() async {
        guard !busy else { return }
        busy = true; error = nil; errorRepair = nil; message = nil; plan = [:]
        do { plan = try await ClientSetup.provision("plan") }
        catch { record(error) }
        busy = false
    }

    public func install(workspace: URL?) async {
        guard !busy, let expected = planHash else { return }
        busy = true; error = nil; errorRepair = nil; message = ResearchSetupCopy.installing
        defer { busy = false; plan = [:] }
        do {
            let result = try await ClientSetup.provision(clientReady ? "repair" : "apply", expected: expected, approved: true)
            readiness = await ClientSetup.inspect(workspace: workspace)
            message = clientReady ? ResearchSetupCopy.installed : ResearchSetupCopy.installedButNotReady
            if case .string(let log) = result["logPath"] { message = (message ?? "") + " Setup log: " + log }
        } catch { record(error); message = nil }
    }

    private func record(_ failure: any Error) {
        let described = ResearchSetupCopy.failure(failure)
        error = described.reason
        errorRepair = described.repair
    }
}

/// Research Setup's wording, for a researcher with no machine-learning
/// background. The command line keeps `ScientificPythonRuntime.setupHint`;
/// nothing here names a command, a flag, an environment variable, or a build.
public enum ResearchSetupCopy {
    public static let title = "Start your research workspace"

    public static let introduction =
        "Bring a research question. SteerLab and your coding assistant help "
        + "turn it into a study you can review, run, and inspect."

    // MARK: Step 1 — the workspace

    public static let workspaceStepTitle = "1. Choose where your study lives"

    public static let workspacePrompt =
        "Choose a folder on this Mac for your prompts, study designs, and results."

    public static let workspaceCaption =
        "The app and your coding assistant can both open the same workspace. "
        + "Your study files stay in this folder."

    /// Under the step's three buttons: what the third one is for.
    public static let demoCaption =
        "New here? A Demo Workspace is a finished study to read and a draft "
        + "to run. It opens as a copy in a folder you choose."

    // MARK: Step 2 — where studies run

    public static let computeStepTitle = "2. Choose where studies run"

    public static let computeNeedsWorkspace =
        "Choose a workspace first. This setting is kept with the workspace."

    public static let computeCaption =
        "You can change this later from the Workspace menu. Choosing a "
        + "place to run installs nothing by itself."

    // MARK: Step 3 — the helper

    public static let helperStepTitle = "3. Set up the study-design helper"

    public static let helperExplanation =
        "SteerLab uses a small helper to guide study design, bring results "
        + "back into your workspace, and prepare text collections. Setting "
        + "it up does not download a model, and it does not set up a server."

    public static let helperReady = "Helper ready"
    public static let helperUpdateTitle = "Helper update available"
    public static let helperSetupTitle = "Helper setup needed"

    /// The helper is missing: the ordinary first-launch state.
    public static let helperNeeded =
        "The helper is not installed yet. Choose Review Setup Plan to see "
        + "exactly what will be installed, then approve it. It takes a few "
        + "minutes and needs an internet connection."

    /// The helper works for basic study design and is behind for the rest.
    public static let helperUpdateNeeded =
        "You can design studies now. Choose Review Update Plan to add the "
        + "tools for preparing text collections."

    public static let planCaption =
        "Needs an internet connection. Earlier installs are kept, and none of "
        + "your study files change."

    public static let installing =
        "Installing the study-design helper. Downloads may take a few minutes."

    public static let installed =
        "The helper is ready. You can design studies."

    public static let installedButNotReady =
        "Installation finished, but the helper is still not ready. Choose "
        + "Check Again. If this message stays, reinstall SteerLab and open "
        + "Research Setup once more."

    // MARK: Step 4 — begin

    public static let beginStepTitle = "4. Begin with your question"

    public static let beginExplanation =
        "Give your coding assistant the workspace instructions, and describe "
        + "what you want to understand. It can find the available methods, "
        + "help prepare datasets, and propose a study for your review."

    public static let copyInstructions = "Copy Instructions for Your Coding Assistant"
    public static let instructionsCopied = "Instructions Copied"

    public static let workInTheApp =
        "To work in the app instead, close this screen and open Studies or "
        + "Templates. When you are ready to run a study, prepare your model "
        + "in the Playground or in Compute."

    public static let readyFooter = "Ready to design studies"
    public static let returnFooter =
        "You can return here from the Workspace menu: Research Setup."

    // MARK: Failures

    /// Shown when this copy of the app cannot use its own installer. The
    /// command line's version of this names a build script.
    public static let installerMissing =
        "This copy of SteerLab cannot find the helper installer that belongs "
        + "to it."
    public static let reinstallRepair =
        "Reinstall SteerLab, then open Research Setup again."

    /// The repair shown where the command line's hint would appear.
    public static let helperRepair =
        "Choose Review Setup Plan in this sheet, then approve the plan."

    /// A plan or install failure as the sheet shows it: the reason, and the
    /// repair it carried — in the app's words where the carried repair was
    /// written for the command line.
    public static func failure(_ error: any Error) -> (reason: String, repair: String?) {
        guard let typed = error as? ExperimentError else {
            return (error.localizedDescription, nil)
        }
        // The helper's files and this build are different versions, or the
        // helper could not be asked: the app's own words, not the command
        // line's. Only the helper failing to start is a setup problem, and it
        // keeps this sheet's setup sentence.
        if let identity = typed.clientIdentityFailure {
            return (identity.appSummary,
                    identity.cause == .noAnswer ? helperRepair : identity.appNextStep)
        }
        guard let repair = typed.malformedInvocation?.repairAction else {
            return (typed.reason, nil)
        }
        if repair == ScientificPythonRuntime.setupHint {
            return (typed.reason, helperRepair)
        }
        if repair == ClientSetup.installerMissingRepair {
            return (installerMissing, reinstallRepair)
        }
        return (typed.reason, repair)
    }
}
