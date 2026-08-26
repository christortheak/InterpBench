import ExperimentKit
import SwiftUI

/// Primary navigation sections, ordered by the research lifecycle (design
/// brief: docs/UI_REDESIGN_AGENT_WORKBENCH_EXPERIENCE.md), not implementation
/// history. Existing panel TYPES keep their names — only user-facing labels
/// change (Steering→Playground, Variants→Agents, Concept Lab→Data,
/// Geometry→Analysis).
enum WorkbenchSection: String, CaseIterable, Identifiable {
    case home = "Home"
    // Playground above Data (researcher request 2026-07-18): interactive
    // chat is the daily entry point.
    case playground = "Playground"
    // Data before Agents (researcher request 2026-08-18): datasets are what
    // agents are derived from, so the inventory precedes the derivations;
    // Multi-Agent follows Agents as the composition of them.
    case data = "Data"
    case agents = "Agents"
    case multiAgent = "Multi-Agent"
    // Templates immediately above Studies (researcher request 2026-08-06): a
    // design is what a study MEANS minus its agents, so the library that holds
    // designs sits beside — and before — the place they are cast and run.
    case templates = "Templates"
    case studies = "Studies"
    case results = "Results"
    case analysis = "Analysis"
    case compute = "Compute"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .home: "house"
        case .agents: "person.crop.square.on.square.angled"
        case .playground: "bubble.left.and.text.bubble.right"
        case .data: "square.stack.3d.up"
        case .templates: "square.on.square"
        case .studies: "checkmark.seal"
        case .multiAgent: "person.3.sequence"
        case .results: "archivebox"
        case .analysis: "waveform.path.ecg"
        case .compute: "server.rack"
        }
    }

    var help: String {
        switch self {
        case .home: "workspace dashboard: where am I, what exists, what next"
        case .agents: "the agent workbench — library, creation (manual or optimize from a concept vector), and optimization runs"
        case .playground: "interactive chat and exploratory steering (formerly Steering)"
        case .data: "the workspace's data: inventory of every dataset and derived artifact, the one New Dataset flow, and the editors that author and build them (concepts, corpora, vectors, adapter training)"
        case .templates: "the design library: a study's task, instruments, sampling and judges — with no agents and no compute"
        case .studies: "evidence-grade study protocols (draft → freeze → run)"
        case .multiAgent: "scenario and protocol builder"
        case .results: "browse immutable run directories"
        case .analysis: "vector geometry and mechanistic analysis (formerly Geometry)"
        case .compute: "server connections, jobs, logs, and installs"
        }
    }

    /// Minimum width for the main content pane (the dense panels need room).
    var minimumContentWidth: CGFloat {
        switch self {
        case .playground: 340
        case .home: 420
        default: 560
        }
    }
}

/// Regions within the Agents section (design brief:
/// docs/AGENT_CREATION_SWEEP_UI_RECOMMENDATION.md — Screens is no longer a
/// top-level section; the sweep/promote workflow is agent creation, and the
/// Optimizations region is the returning-user path to in-flight and
/// completed optimization runs).
enum AgentsRegion: String, CaseIterable, Identifiable {
    case library = "Library"
    case create = "New Agent"
    case optimizations = "Optimizations"

    var id: String { rawValue }

    var help: String {
        switch self {
        case .library: "browse, filter, and check saved agents"
        case .create: "create an agent — manually or by optimizing a concept vector"
        case .optimizations: "declared layer×alpha optimization runs: grids, recommendations, Create Agent"
        }
    }
}

/// What the right-hand display pane (the viewer) shows. It follows the
/// selected section unless pinned (so a long job log can stay visible while
/// browsing elsewhere). Chat transcript controls appear ONLY in `.chat`.
enum ActivityViewerMode: Equatable, Sendable {
    case chat
    case multiAgentRun
    case agentRobustness
    /// Results section: the selected run's focused FILE rendered as a bounded
    /// preview — the run list, stamps, and file list live in the Results
    /// main pane (contents in the viewer, listing in the controls pane).
    case resultsRun
    /// Analysis section: the computed cosine/RSA tables — the vector
    /// selection and Compute controls live in the Analysis main pane.
    case analysisGeometry
    /// Data section, Inventory tool: the DETAIL of the selected dataset or
    /// derived artifact — its facts, provenance, and the routes into the
    /// tool that owns it. Same split as Results: the tables, filters, and
    /// selection live in the Data main pane, the selection's contents here
    /// (2026-08-26: the detail used to render inline BELOW the table).
    case dataInventory
    /// Studies section: the selected study itself — its manifest JSON
    /// (2026-08-26) and the rich guide for its type (what it is, what the
    /// researcher provides, what it measures) in the viewer where there is
    /// room to read them (2026-07-19: the tiny in-form caption was not
    /// enough).
    case studyOverview
    case activity(title: String)

    static func mode(for section: WorkbenchSection) -> ActivityViewerMode {
        switch section {
        case .playground: .chat
        case .multiAgent: .multiAgentRun
        case .agents: .agentRobustness
        case .results: .resultsRun
        case .analysis: .analysisGeometry
        case .studies: .studyOverview
        case .home: .activity(title: "Recent Activity")
        default: .activity(title: "Activity")
        }
    }

    /// Region-aware variant: within Agents, only the Library region wants
    /// the robustness viewer; New Agent and Optimizations show the live
    /// Activity feed so a sweep started there streams its logs right beside
    /// the controls (live-testing finding: an optimization run from Agents
    /// previously showed nothing until the researcher visited Compute).
    /// Within Data the same rule holds for the same reason: only the
    /// Inventory tool has a SELECTION to render, and the builders beside it
    /// (Concepts & Vectors, Adapter Training, OptVec) stream their builds
    /// into the shared Activity feed.
    /// Pinning is unaffected — a pinned viewer always wins over this.
    static func mode(
        for section: WorkbenchSection, agentsRegion: AgentsRegion,
        dataTool: DataSectionView.Tool
    ) -> ActivityViewerMode {
        switch section {
        case .agents:
            switch agentsRegion {
            case .library: return .agentRobustness
            case .create, .optimizations: return .activity(title: "Activity")
            }
        case .data:
            return dataTool == .inventory ? .dataInventory : mode(for: section)
        default:
            return mode(for: section)
        }
    }

    var title: String {
        switch self {
        case .chat: "Chat"
        case .multiAgentRun: "Multi-Agent Run"
        case .agentRobustness: "Agent Robustness"
        case .resultsRun: "Selected Run"
        case .analysisGeometry: "Vector Geometry"
        case .dataInventory: "Selected Data"
        case .studyOverview: "Selected Study"
        case .activity(let title): title
        }
    }

    /// Whose data this viewer renders (viewer-state hygiene, 2026-08-19).
    /// Every mode but `.activity` renders ONE section's own state — its
    /// selected run, its computed tables, its transcript — and may not
    /// appear under another section's selection without a pin label.
    /// `.activity` is the single workspace-wide feed that `home`,
    /// `templates`, `compute`, and the authoring regions of Agents and Data
    /// share; its persistence across sections is correct, not leakage.
    var ownership: WorkbenchViewerOwnership {
        switch self {
        case .activity: .workspaceWide
        case .chat, .multiAgentRun, .agentRobustness, .resultsRun,
            .analysisGeometry, .dataInventory, .studyOverview:
            .section
        }
    }

    /// Hygiene identity: title + the section that owns this viewer. Passed
    /// to `WorkbenchViewerPin` (ExperimentKit), which owns the pin/reset/
    /// labelling decisions so they are unit-testable.
    func identity(ownedBy section: WorkbenchSection) -> WorkbenchViewerIdentity {
        WorkbenchViewerIdentity(
            title: title, sectionLabel: section.rawValue, ownership: ownership)
    }
}
