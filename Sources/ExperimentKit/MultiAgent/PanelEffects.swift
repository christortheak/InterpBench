import Foundation

/// Panel-effect decomposition (plan C3) — the Swift twin of the server's
/// `panel_effects.py`, byte-compatible in its CSV output so
/// `RunResults.panelEffects(fromCSV:)` reads either engine's file.
///
/// The panel arm's estimands, from a study's paired configured/baseline
/// transcripts (same scenario, interventions stripped in the baseline). Turns
/// pair by turn id; a seat is TREATED when its agent carries a variant
/// artifact, which is exactly what stripping removes:
///
/// - `direct` — treated seats' endpoint shift against their own baseline turns
/// - `spillover` — untreated seats' shift while sharing a panel with treated
///   ones; they never received an injection, only routed context from agents
///   that did, so this is the propagation channel
/// - `group` — the shift of the designated group-outcome turn (default: the
///   last turn, the final disposition)
/// - `transmissionRatio` = spillover / direct — how much of the injected
///   stance survives deliberation
/// - `amplification` = group / direct — whether deliberating amplifies or damps
///   the treated seat's shift
///
/// Endpoints are pluggable text parsers. A turn whose endpoint fails to parse
/// in EITHER condition is dropped from that estimand and counted — never
/// coerced to zero, which would silently pull every mean toward no-effect.
public enum PanelEffects {

    public struct Row: Sendable, Equatable {
        public let endpoint: String
        public let direct: Double
        public let directN: Int
        public let spillover: Double
        public let spilloverN: Int
        public let group: Double
        public let groupN: Int
        public let droppedTurns: Int
        /// Untreated seats speaking BEFORE any treated output could reach
        /// them — a placebo channel that should sit at ~zero. See
        /// `exposureByTurn`.
        public let unexposed: Double
        public let unexposedN: Int

        /// Undefined when either side is unmeasured or `direct` is zero —
        /// a ratio against no direct effect describes nothing.
        public var transmissionRatio: Double {
            guard !direct.isNaN, !spillover.isNaN, direct != 0 else { return .nan }
            return spillover / direct
        }

        public var amplification: Double {
            guard !direct.isNaN, !group.isNaN, direct != 0 else { return .nan }
            return group / direct
        }
    }

    /// Whether each turn's SPEAKER had, at that point in the script,
    /// received routed context tracing back to a treated seat.
    ///
    /// Spillover is meant to measure transmission: an untreated agent moving
    /// because it READ something a steered agent wrote. Classifying every
    /// untreated speaker as spillover counts turns that cannot carry the
    /// intervention at all — the private-notes turns at the head of a
    /// typical deliberation template have empty context, so their prompts are
    /// byte-identical in both arms and their contribution is a structural
    /// zero. Averaging those zeros in dilutes the estimate toward "no
    /// transmission".
    ///
    /// Exposure is transitive: an agent is exposed once it receives routed
    /// output from a treated seat, OR from an already-exposed agent. That
    /// second-order case (A speaks to B, B speaks to C) is real propagation
    /// and arguably the most interesting kind.
    public static func exposureByTurn(
        scenario: MultiAgentScenario, treated: Set<String>
    ) -> [String: Bool] {
        // Two channels reach a prompt, and BOTH count, because exposure is
        // about what a turn actually READS — not about what routing recorded:
        //
        //   * routed context, which a turn only reads when
        //     `includeSpeakerContext` is true. A turn that switches it off
        //     sees none of the deliberation, however much has accumulated.
        //   * `{{outputs.<label>}}` interpolation, which pulls a named turn's
        //     output straight into the prompt with no routing involved.
        //     Routing-only accounting misses this and marks a genuinely
        //     exposed turn unexposed.
        var carrierLabels = Set<String>()
        var contextExposed = Set<String>()
        var out: [String: Bool] = [:]
        for (index, turn) in scenario.turns.enumerated() {
            let speaker = turn.speakerAgentID
            let viaContext = turn.includeSpeakerContext && contextExposed.contains(speaker)
            let viaOutputs = !carrierLabels.isDisjoint(
                with: MultiAgentRunner.outputReferences(in: turn.promptTemplate))
            let exposedHere = viaContext || viaOutputs
            out[turn.id] = exposedHere

            // This turn's output carries the treatment when its speaker is
            // treated, or when the speaker read treated material to write it.
            if treated.contains(speaker) || exposedHere {
                let explicit = turn.outputLabel
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                carrierLabels.insert(explicit.isEmpty ? "turn_\(index + 1)" : explicit)
                for listener in routedAgentIDs(for: turn, agents: scenario.agents)
                where listener != speaker {
                    contextExposed.insert(listener)
                }
            }
        }
        return out
    }

    /// Routing resolution, matching `MultiAgentRunner.routedAgentIDs`.
    private static func routedAgentIDs(
        for turn: MultiAgentScenario.Turn, agents: [MultiAgentScenario.Agent]
    ) -> [String] {
        switch turn.routing {
        case .all: return agents.map(\.id)
        case .speakerOnly: return [turn.speakerAgentID]
        case .selected: return turn.routedAgentIDs
        case .none: return []
        }
    }

    /// Seats whose agents carry a variant artifact.
    public static func treatedAgentIDs(in scenario: MultiAgentScenario) -> Set<String> {
        Set(
            scenario.agents
                .filter { !($0.variantArtifactPath ?? "").isEmpty }
                .map(\.id))
    }

    /// Decompose one endpoint over paired turns.
    ///
    /// `groupTurnID` names the panel-outcome turn; nil defaults to the last
    /// configured turn.
    public static func compute(
        configured: [MultiAgentTurnResult],
        baseline: [MultiAgentTurnResult],
        treated: Set<String>,
        endpoint: String,
        parse: (String) -> Double?,
        groupTurnID: String? = nil,
        exposedTurns: [String: Bool]? = nil
    ) -> Row {
        var baselineByID: [String: MultiAgentTurnResult] = [:]
        for turn in baseline { baselineByID[turn.turnID] = turn }
        let groupID = groupTurnID ?? configured.last?.turnID

        var directDiffs: [Double] = []
        var spilloverDiffs: [Double] = []
        var groupDiffs: [Double] = []
        var unexposedDiffs: [Double] = []
        var dropped = 0

        for turn in configured {
            guard let counterpart = baselineByID[turn.turnID] else {
                dropped += 1
                continue
            }
            guard
                let treatedValue = parse(turn.output),
                let baselineValue = parse(counterpart.output)
            else {
                dropped += 1
                continue
            }
            let diff = treatedValue - baselineValue
            if turn.turnID == groupID { groupDiffs.append(diff) }
            if treated.contains(turn.speakerAgentID) {
                directDiffs.append(diff)
            } else if exposedTurns?[turn.turnID] ?? true {
                // Untreated AND exposed: the transmission channel.
                spilloverDiffs.append(diff)
            } else {
                // Untreated and not yet exposed — could not have carried the
                // intervention. Kept as a placebo channel rather than
                // discarded: it should sit at ~zero, and if it does not,
                // something leaked.
                unexposedDiffs.append(diff)
            }
        }
        return Row(
            endpoint: endpoint,
            direct: mean(directDiffs), directN: directDiffs.count,
            spillover: mean(spilloverDiffs), spilloverN: spilloverDiffs.count,
            group: mean(groupDiffs), groupN: groupDiffs.count,
            droppedTurns: dropped,
            unexposed: mean(unexposedDiffs), unexposedN: unexposedDiffs.count)
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? .nan : values.reduce(0, +) / Double(values.count)
    }

    /// `%.6g`-equivalent, with NaN as an empty field — the server's `_fmt`.
    static func format(_ value: Double) -> String {
        value.isNaN ? "" : String(format: "%.6g", value)
    }

    /// panel-effects.csv. Header and column order are the cross-engine
    /// contract (server `write_panel_effects` fieldnames).
    public static func csv(_ rows: [Row]) -> String {
        var lines = [
            "endpoint,direct,directN,spillover,spilloverN,group,groupN,"
                + "transmissionRatio,amplification,unexposed,unexposedN,"
                + "droppedTurns"
        ]
        for row in rows {
            lines.append(
                [
                    row.endpoint,
                    format(row.direct), String(row.directN),
                    format(row.spillover), String(row.spilloverN),
                    format(row.group), String(row.groupN),
                    format(row.transmissionRatio),
                    format(row.amplification),
                    format(row.unexposed), String(row.unexposedN),
                    String(row.droppedTurns),
                ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }
}
