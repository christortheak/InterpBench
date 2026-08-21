"""Panel-effect decomposition for multi-agent (appellate-panel) studies.

The panel arm's estimands, computed from a study's paired ``configured`` /
``baseline`` scenario runs (same scenario, interventions stripped in the
baseline). Turns pair by ``turnID``; seats are treated when their agent
carries a variant artifact (that is what stripping removes):

- **direct**   — treated seats' endpoint shift vs their own baseline turns
- **spillover** — untreated seats' shift when sharing a panel with treated
  seats (they receive routed context from steered agents)
- **group**    — the shift of the designated group-outcome turn (default: the
  final turn), the panel-level disposition
- **transmissionRatio** — spillover / direct (how much of the injected stance
  leaks through deliberation)
- **amplification** — group / direct (does deliberation amplify or damp the
  treated seat's shift?)

Endpoints are pluggable text parsers (months, choice-target, word count…);
turns whose endpoint fails to parse in either condition are dropped from that
estimand and counted, never coerced.

Sibling artifact: ``voice_lint.csv_rows`` writes ``panel-voice-lint.csv`` (one
row per speaker × condition). It is deliberately not a column here — this
file's row is an ENDPOINT of a paired decomposition and its header is a
cross-engine contract — and it is written for every panel run, including the
multi-replicate ones this decomposition skips.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Callable


import re

_OUTPUT_REFS = re.compile(r"\{\{outputs\.([^}]+)\}\}")


@dataclass
class PanelEffects:
    endpoint: str
    direct: float
    direct_n: int
    spillover: float
    spillover_n: int
    group: float
    group_n: int
    dropped_turns: int
    #: Untreated seats speaking BEFORE any treated output could reach them —
    #: a placebo channel that should sit at ~zero (see exposure_by_turn).
    unexposed: float = float("nan")
    unexposed_n: int = 0

    @property
    def transmission_ratio(self) -> float:
        if math.isnan(self.direct) or math.isnan(self.spillover) or self.direct == 0:
            return float("nan")
        return self.spillover / self.direct

    @property
    def amplification(self) -> float:
        if math.isnan(self.direct) or math.isnan(self.group) or self.direct == 0:
            return float("nan")
        return self.group / self.direct

    def as_row(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "direct": _fmt(self.direct), "directN": self.direct_n,
            "spillover": _fmt(self.spillover), "spilloverN": self.spillover_n,
            "group": _fmt(self.group), "groupN": self.group_n,
            "transmissionRatio": _fmt(self.transmission_ratio),
            "amplification": _fmt(self.amplification),
            "unexposed": _fmt(self.unexposed), "unexposedN": self.unexposed_n,
            "droppedTurns": self.dropped_turns,
        }


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6g}"


def treated_agent_ids(scenario) -> set[str]:
    """Seats whose agents carry a variant artifact — exactly the interventions
    the baseline condition strips. Accepts a scenario JSON dict or the loaded
    ``multi_agent.Scenario`` object."""
    agents = scenario.get("agents", []) if isinstance(scenario, dict) \
        else getattr(scenario, "agents", [])
    treated: set[str] = set()
    for agent in agents:
        if isinstance(agent, dict):
            if agent.get("variantArtifactPath"):
                treated.add(agent.get("id", ""))
        elif getattr(agent, "variant_artifact_path", None):
            treated.add(agent.id)
    return treated


def exposure_by_turn(scenario, treated: set[str]) -> dict[str, bool]:
    """Whether each turn's SPEAKER had, at that point in the script, received
    routed context tracing back to a treated seat.

    Spillover is meant to measure transmission: an untreated agent moving
    because it READ something a steered agent wrote. Classifying every
    untreated speaker as spillover counts turns that cannot carry the
    intervention at all — the private-notes turns at the head of a typical
    deliberation template have empty context, so their prompts are
    byte-identical in both
    arms and their contribution is a structural zero. Averaging those zeros in
    dilutes the estimate toward "no transmission".

    Exposure is transitive: an agent is exposed once it receives routed output
    from a treated seat, OR from an agent that is already exposed. The
    second-order case (A speaks to B, B speaks to C) is real propagation and
    arguably the most interesting kind.
    """
    from . import multi_agent

    agents = getattr(scenario, "agents", [])
    turns = getattr(scenario, "turns", [])
    # Two channels reach a prompt, and BOTH must count, because exposure is
    # about what the turn actually READS — not about what routing recorded:
    #
    #   * routed context, which a turn only reads when
    #     include_speaker_context is true. A turn that switches it off does
    #     not see the deliberation at all, however much has accumulated.
    #   * {{outputs.<label>}} interpolation, which pulls a named turn's output
    #     straight into the prompt with no routing involved. Routing-only
    #     accounting misses this entirely and marks a genuinely exposed turn
    #     unexposed.
    carrier_labels: set[str] = set()   # labels whose output carries treatment
    context_exposed: set[str] = set()  # agents whose ROUTED context carries it
    out: dict[str, bool] = {}
    for index, turn in enumerate(turns):
        speaker = turn.speaker_agent_id
        via_context = turn.include_speaker_context and speaker in context_exposed
        via_outputs = bool(carrier_labels & set(
            _OUTPUT_REFS.findall(turn.prompt_template)))
        exposed_here = via_context or via_outputs
        out[turn.id] = exposed_here

        # This turn's own output carries the treatment when its speaker is
        # treated, or when the speaker read treated material to produce it.
        if speaker in treated or exposed_here:
            label = turn.output_label.strip() or f"turn_{index + 1}"
            carrier_labels.add(label)
            for listener in multi_agent._routed_ids(turn, agents):
                if listener != speaker:
                    context_exposed.add(listener)
    return out


def compute(configured_turns: list[dict], baseline_turns: list[dict],
            treated: set[str], *, endpoint_name: str,
            parse: Callable[[str], float | None],
            group_turn_id: str | None = None,
            exposed_turns: dict[str, bool] | None = None) -> PanelEffects:
    """Decompose one endpoint over paired turns.

    ``group_turn_id`` names the panel-outcome turn; default is the last turn
    of the configured transcript (the final disposition in the appellate
    template)."""
    baseline_by_id = {turn.get("turnID"): turn for turn in baseline_turns}
    if group_turn_id is None and configured_turns:
        group_turn_id = configured_turns[-1].get("turnID")

    direct_diffs: list[float] = []
    spillover_diffs: list[float] = []
    group_diffs: list[float] = []
    unexposed_diffs: list[float] = []
    dropped = 0
    for turn in configured_turns:
        counterpart = baseline_by_id.get(turn.get("turnID"))
        if counterpart is None:
            dropped += 1
            continue
        treated_value = parse(turn.get("output", ""))
        baseline_value = parse(counterpart.get("output", ""))
        if treated_value is None or baseline_value is None:
            dropped += 1
            continue
        diff = treated_value - baseline_value
        if turn.get("turnID") == group_turn_id:
            group_diffs.append(diff)
        if turn.get("speakerAgentID") in treated:
            direct_diffs.append(diff)
        elif exposed_turns is None or exposed_turns.get(turn.get("turnID"), True):
            # Untreated AND exposed: the transmission channel.
            spillover_diffs.append(diff)
        else:
            # Untreated and NOT yet exposed. Not spillover — this turn could
            # not have carried the intervention. Kept as a placebo channel
            # rather than discarded: these should sit at ~zero, and if they do
            # not, something leaked.
            unexposed_diffs.append(diff)
    return PanelEffects(
        endpoint=endpoint_name,
        direct=_mean(direct_diffs), direct_n=len(direct_diffs),
        spillover=_mean(spillover_diffs), spillover_n=len(spillover_diffs),
        group=_mean(group_diffs), group_n=len(group_diffs),
        unexposed=_mean(unexposed_diffs), unexposed_n=len(unexposed_diffs),
        dropped_turns=dropped)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_panel_effects(run_directory: str, scenario, *,
                        endpoints: dict[str, Callable[[str], float | None]],
                        group_turn_id: str | None = None) -> list[PanelEffects]:
    """Read the run's configured/baseline turn files and write
    panel-effects.csv (one row per endpoint). No-op (empty list) when either
    condition is missing — a panel study without a baseline arm has no paired
    estimand, which the caller should surface, not fake."""
    rows: list[PanelEffects] = []
    configured = _read_turns(os.path.join(run_directory, "configured", "turns.jsonl"))
    baseline = _read_turns(os.path.join(run_directory, "baseline", "turns.jsonl"))
    if not configured or not baseline:
        return rows
    treated = treated_agent_ids(scenario)
    exposed = exposure_by_turn(scenario, treated)
    for name, parse in endpoints.items():
        rows.append(compute(configured, baseline, treated, endpoint_name=name,
                            parse=parse, group_turn_id=group_turn_id,
                            exposed_turns=exposed))
    path = os.path.join(run_directory, "panel-effects.csv")
    fieldnames = ["endpoint", "direct", "directN", "spillover", "spilloverN",
                  "group", "groupN", "transmissionRatio", "amplification",
                  "unexposed", "unexposedN", "droppedTurns"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
    return rows


def _read_turns(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    turns: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns
