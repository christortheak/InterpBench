/// An agent is a model plus optional injected steering vectors and a
/// transcript. Experiment B's "judge" is a single `Agent`; Experiment A
/// composes several. Run orchestration must not assume a single agent.
/// (Extension point per CLAUDE.md › Experiment A extension points —
/// interface lands with Phase 0 plumbing, implementations later.)
public protocol Agent: Sendable {
    var id: String { get }
}
