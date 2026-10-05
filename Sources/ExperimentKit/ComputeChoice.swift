import Foundation

/// Where a workspace's studies run, as the three choices a researcher sees.
///
/// The persisted fact is still `WorkspaceCompute` — two bindings, because the
/// lifecycle only needs to know whose artifacts are native to a workspace:
/// the engine built into this app (MLX), or the Python engine. But "the
/// Python engine" hid the choice that matters to someone deciding how to
/// work: it runs on THIS Mac's GPU just as well as on another machine, and
/// nothing in the old two-way picker ("Cluster" / "Local (MLX)") said so. A
/// researcher with only a laptop read "Cluster" as "not for me" and was left
/// with the engine the caption called "toy models and pipeline checks".
///
/// So the choice has three names, in plain words, and the second and third
/// both bind today's `cluster` value. Which of the two was chosen is
/// remembered in one optional key (`WorkspaceCompute.Location`) beside the
/// binding; nothing in the lifecycle reads it.
public enum ComputeChoice: String, CaseIterable, Sendable, Codable, Identifiable {
    /// The engine built into this app (MLX).
    case macQuickStart = "mac-quick-start"
    /// The Python engine, on this Mac's GPU.
    case macFullCapabilities = "mac-full-capabilities"
    /// The Python engine, on a workstation or a cluster.
    case anotherMachine = "another-machine"

    public var id: String { rawValue }

    /// What a new workspace starts with, from every entry point: the choice
    /// that needs nothing installed beyond a model.
    public static let newWorkspaceDefault: ComputeChoice = .macQuickStart

    // MARK: Words

    public var title: String {
        switch self {
        case .macQuickStart: "This Mac, quick start"
        case .macFullCapabilities: "This Mac, full capabilities"
        case .anotherMachine: "Another machine"
        }
    }

    /// One or two sentences under the title: what it is for and what it costs.
    public var summary: String {
        switch self {
        case .macQuickStart:
            "Core steering studies on small models, using the engine built "
                + "into this app. Nothing to install beyond a model."
        case .macFullCapabilities:
            "Every method, using the Python engine on this Mac's own "
                + "graphics processor. Needs a one-time setup of several "
                + "gigabytes."
        case .anotherMachine:
            "Every method, on a workstation or a cluster you connect to. "
                + "Your study files and results stay on this Mac."
        }
    }

    /// The engine, named once for the reader who meets its name elsewhere.
    public var engineNote: String {
        switch self {
        case .macQuickStart: "Engine: MLX, built into this app."
        case .macFullCapabilities: "Engine: Python, on this Mac."
        case .anotherMachine: "Engine: Python, on the machine you connect to."
        }
    }

    /// Whether this choice runs every method (the Python engine) or the core
    /// set (the engine built into the app).
    public var runsEveryMethod: Bool { self != .macQuickStart }

    // MARK: The persisted binding

    /// The binding this choice writes — unchanged from before the three
    /// names existed.
    public var binding: WorkspaceCompute {
        switch self {
        case .macQuickStart: .localMLX
        case .macFullCapabilities, .anotherMachine: .cluster
        }
    }

    /// What is remembered beside a `cluster` binding. Nil for the quick
    /// start, whose binding already says everything.
    public var location: WorkspaceCompute.Location? {
        switch self {
        case .macQuickStart: nil
        case .macFullCapabilities: .thisMac
        case .anotherMachine: .anotherMachine
        }
    }

    /// Back from what is on disk — and from nothing else when the location
    /// was recorded.
    ///
    /// A `cluster` binding with NO recorded location is every workspace
    /// declared before the key existed, and every one an older build has
    /// re-declared since. The file cannot say which of the two it means, so
    /// the caller supplies what it can see: whether the engine the app is
    /// using right now is this Mac's own. With no such evidence it reads as
    /// another machine — what "Cluster" always meant.
    public init(
        binding: WorkspaceCompute, location: WorkspaceCompute.Location?,
        activeEngineIsThisMac: Bool = false
    ) {
        switch (binding, location) {
        case (.localMLX, _): self = .macQuickStart
        case (.cluster, .thisMac?): self = .macFullCapabilities
        case (.cluster, .anotherMachine?): self = .anotherMachine
        case (.cluster, nil):
            self = activeEngineIsThisMac ? .macFullCapabilities : .anotherMachine
        }
    }
}

// MARK: - What runs where

/// The one compact account of what each choice can run, and what switching
/// costs. Data rather than view text, so the claims are unit-tested against
/// the choices they describe and every surface says the same thing.
public enum ComputeGuide {

    /// One line of the "what runs where" table.
    public struct Row: Sendable, Equatable, Identifiable {
        public let id: String
        /// The activity, in the researcher's words.
        public let activity: String
        /// Runs on the engine built into this app (the quick start).
        public let quickStart: Bool
        /// Runs on the Python engine — on this Mac or on another machine.
        public let pythonEngine: Bool

        public init(_ id: String, _ activity: String, quickStart: Bool, pythonEngine: Bool = true) {
            self.id = id
            self.activity = activity
            self.quickStart = quickStart
            self.pythonEngine = pythonEngine
        }

        public func runs(on choice: ComputeChoice) -> Bool {
            choice == .macQuickStart ? quickStart : pythonEngine
        }
    }

    public static let title = "What Runs Where"

    public static let introduction =
        "SteerLab can run a study in three places. All three use the same "
        + "workspace on this Mac; they differ in which methods are available "
        + "and what has to be set up first."

    /// The table. The quick start covers the core of a steering study; the
    /// Python engine — on this Mac or on another machine — covers every row.
    public static let rows: [Row] = [
        Row("chat", "Chat with a model and try a steering vector", quickStart: true),
        Row("vector", "Build a concept vector from example texts", quickStart: true),
        Row("study", "Run a study that compares a model with and without steering",
            quickStart: true),
        Row("multi-agent", "Run a scenario in which several agents take turns",
            quickStart: true),
        Row("optimize", "Search for the best place and strength to steer",
            quickStart: true),
        Row("adapter", "Train an adapter", quickStart: true),
        Row("probes", "Probes and intervention policies", quickStart: false),
        Row("optvec", "Trained steering vectors (OptVec)", quickStart: false),
        Row("jlens", "Jacobian lens: import or fit a lens, and derive token directions",
            quickStart: false),
        Row("sae", "Import a feature from a sparse autoencoder (SAE)", quickStart: false),
        Row("battery", "Standalone capability checks", quickStart: false),
    ]

    /// The limits, said up front rather than discovered.
    public static let limits: [String] = [
        "Vectors and results do not carry between the quick start and the "
            + "Python engine. If you switch, build your vectors again and run "
            + "the study again on the engine you switched to.",
        "On this Mac, memory and model size are the limits. A model has to "
            + "fit in this Mac's memory with room to work, and very long "
            + "prompts need more memory still.",
        "Results from different engines or machines are not numerically "
            + "interchangeable. Report one study from one of them, and name it.",
    ]

    /// Where a method needs the Python engine and the app is on the quick
    /// start: one sentence — shown with the button that offers the switch,
    /// in place of a refusal. `subject` is what needs it ("Probes", "Importing
    /// an SAE feature"); `plural` picks "need" or "needs".
    public static func needsPythonEngine(_ subject: String, plural: Bool) -> String {
        "\(subject) \(plural ? "need" : "needs") the Python engine, and the "
            + "app is using \(ComputeChoice.macQuickStart.title) right now. "
            + "The Python engine runs on this Mac too."
    }

    public static let switchButton =
        "Switch to \(ComputeChoice.macFullCapabilities.title)…"

    public static let switchCaption =
        "Switching opens the one-time setup, and nothing is installed until "
        + "you approve it. Vectors built in the quick start stay there, so "
        + "build them again after switching."

    public static let guideButton = "What Runs Where…"
}

// MARK: - Declaring versus using

extension ComputeChoice {

    /// Short enough for a menu row.
    public var menuCaption: String {
        switch self {
        case .macQuickStart: "Core steering studies on small models"
        case .macFullCapabilities: "Every method, after a one-time setup of several gigabytes"
        case .anotherMachine: "Every method, on a workstation or a cluster"
        }
    }

    /// What picking a choice in the COMPUTE menu — "use this now" — does to
    /// the workspace's declaration.
    ///
    /// The declaration is a statement about the workspace, read by the
    /// lifecycle to decide whose vectors and evidence are native to it. It
    /// must never be rewritten as a side effect of trying another engine for
    /// an afternoon. So using a choice declares it ONLY when there is nothing
    /// to contradict: the workspace has declared nothing, and its own runs
    /// either say nothing or already say the same engine. A workspace whose
    /// runs point at the OTHER engine stays undeclared — fifty Python runs
    /// are not outvoted by one chat on the quick start. Returns the choice
    /// to record, or nil to leave the declaration alone.
    public static func declarationAfterUsing(
        _ choice: ComputeChoice, declared: ComputeChoice?,
        inferredBinding: WorkspaceCompute?
    ) -> ComputeChoice? {
        guard declared == nil else { return nil }
        guard inferredBinding == nil || inferredBinding == choice.binding else {
            return nil
        }
        return choice
    }

    /// The sentence shown when the workspace is set to one engine and the app
    /// is using the other — the state in which vectors and results land where
    /// this workspace's studies will not count them. Nil when the two agree
    /// on the engine; "this Mac" versus "another machine" is not a mismatch,
    /// because both are the Python engine.
    public static func mismatchNote(
        workspace: ComputeChoice, inUse: ComputeChoice
    ) -> String? {
        guard workspace.binding != inUse.binding else { return nil }
        let workspaceEngine =
            workspace.runsEveryMethod ? "the Python engine" : workspace.title
        let inUseEngine = inUse.runsEveryMethod ? "the Python engine" : inUse.title
        return "This workspace is set to \(workspaceEngine), but the app is "
            + "using \(inUseEngine) right now. Vectors and results made there "
            + "do not count for this workspace's studies. Choose one in the "
            + "Workspace menu so the two agree."
    }

    /// Said when nothing has been chosen for the workspace and the app is
    /// going by its earlier runs (or, with none, by the default). A reading
    /// must never look like a decision the researcher made.
    public static func undeclaredNote(treatingAs choice: ComputeChoice) -> String {
        "Not chosen yet. For now SteerLab treats this workspace as "
            + "\(choice.title). Choose one to confirm."
    }

    /// Said after "Another machine" is chosen and none is connected yet.
    public static let connectAnotherMachine =
        "No other machine is connected yet. Open the Compute menu in the "
        + "toolbar and choose Add a Machine by Address…, or Set Up a "
        + "Cluster…"

    /// Said beside "This Mac, full capabilities" before its setup has run.
    public static let fullCapabilitiesSetup =
        "The setup lists what it will download before anything is fetched, "
        + "and nothing is installed until you approve it."
}

// MARK: - The connection's side

extension ClusterConnectionStore {

    /// The name the Python engine on this Mac is saved under.
    public nonisolated static let thisMacEngineName = "Python engine on this Mac"

    /// Whether a saved server is this Mac's own engine: direct transport to
    /// a loopback address. (A cluster reached through an SSH tunnel also
    /// answers on a loopback address, which is why the transport is checked
    /// and not just the host.)
    public func runsOnThisMac(_ entry: ServerEntry) -> Bool {
        Self.serverSharesLocalFilesystem(
            site: entry.resolvedSite, urlString: entry.urlString)
    }

    /// Saved machines other than this Mac's own engine, in menu order.
    public var otherMachines: [ServerEntry] {
        servers.filter { !runsOnThisMac($0) }
    }

    /// Which of the three choices the app is USING right now — a fact about
    /// the connection, not about the workspace.
    public var activeComputeChoice: ComputeChoice {
        switch activeWorkspace {
        case .local: .macQuickStart
        case .server:
            activeServerSharesLocalFilesystem ? .macFullCapabilities : .anotherMachine
        }
    }

    /// The toolbar's words for what the app is using: the choice's own name
    /// on this Mac, and the machine's name for another machine.
    public var activeComputeTitle: String {
        switch activeComputeChoice {
        case .macQuickStart, .macFullCapabilities: activeComputeChoice.title
        case .anotherMachine: substrateLabel
        }
    }
}
