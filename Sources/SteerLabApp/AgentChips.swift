import ExperimentKit
import SwiftUI

/// Small capsule for one readiness chip computed by
/// `AgentLibrary.chips` (ExperimentKit). Views only render — the rule that
/// produced the chip is unit-tested in the kit.
struct AgentChipView: View {
    let chip: AgentLibrary.Chip

    var body: some View {
        Text(chip.label)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(background))
            .help(chip.help)
    }

    private var background: Color {
        switch chip.tone {
        case .positive: .green.opacity(0.18)
        case .neutral: .secondary.opacity(0.14)
        case .warning: .orange.opacity(0.2)
        }
    }
}

/// Kind badge for an agent row (exploratory / sweep-promoted / override /
/// adapter / vector-only / baseline[virtual]).
struct AgentKindBadge: View {
    let kind: AgentLibrary.Kind

    var body: some View {
        Text(kind.label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(background))
            .help(help)
    }

    /// Where this agent CAME FROM, in a sentence — the badge word alone
    /// ("override-promoted", "vector-only") says nothing to a reader who has
    /// not run that path yet (2026-09-06 audit).
    private var help: String {
        switch kind {
        case .baseline:
            "the unsteered base model itself — not a saved file; it is listed "
                + "so a study can name it as an arm"
        case .sweepPromoted:
            "promoted from an optimization run at the point that run "
                + "recommended — its layer, strength, and evidence are pinned"
        case .overridePromoted:
            "promoted from an optimization run at a point a researcher chose "
                + "instead of the recommended one; the override is recorded "
                + "in the agent"
        case .adapter:
            "carries a trained LoRA adapter (Data → Adapter Training)"
        case .vectorOnly:
            "steers with a concept vector at a fixed layer and strength; no "
                + "trained weights"
        case .exploratory:
            "saved by hand in the Playground or the editor — no optimization "
                + "run stands behind its settings"
        }
    }

    private var background: Color {
        switch kind {
        case .sweepPromoted: .green.opacity(0.18)
        case .overridePromoted: .orange.opacity(0.2)
        case .baseline: .blue.opacity(0.14)
        case .adapter, .vectorOnly, .exploratory: .secondary.opacity(0.14)
        }
    }
}
