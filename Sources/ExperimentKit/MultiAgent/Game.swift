/// A multi-agent game: payoff structure, turn order, message visibility.
///
/// **Status (2026-07-27): an unbuilt placeholder, deliberately.** This has
/// been a two-property stub since 2026-06-09 and nothing depends on it. The
/// cooperation / public-goods application ("Experiment A": prisoner's-dilemma
/// and public-goods panels) has no machinery behind it — payoffs, scoring,
/// and the turn contract are all unwritten.
///
/// It is kept rather than deleted because the SCENARIO machinery it would sit
/// on is now real: scripted turns, routing and visibility, per-agent
/// variants, replicates, transcript-level statistics, and panel-effect
/// decomposition all exist and are what a game harness would compose. When
/// the cooperation study is scheduled, the work is payoffs and scoring on top
/// of `MultiAgentScenario`, not a new engine.
///
/// Do not read the existence of this protocol as coverage. See
/// `docs/STATUS.md` and `docs/MULTI-AGENT-SUBSTRATE-PARITY-PLAN.md` § F5.
public protocol Game: Sendable {
    var name: String { get }
    var playerCount: Int { get }
}

/// Placeholder conformance keeping the `Game` interface honest. Carries no
/// payoff matrix and plays no rounds — see the note on `Game`.
public struct PrisonersDilemma: Game {
    public let name = "prisoners-dilemma"
    public let playerCount = 2
    public init() {}
}
