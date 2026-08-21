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
