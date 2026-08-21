"""Panel-effect decomposition: direct / spillover / group / transmission."""

import csv
import json
import math
import os

from steerlab_server.experiment import judicial, panel_effects


def _turn(turn_id, speaker, output):
    return {"turnID": turn_id, "speakerAgentID": speaker, "output": output}


CONFIGURED = [
    _turn("t1", "treated-judge", "I would impose 30 months."),
    _turn("t2", "peer-a", "I lean toward 24 months."),
    _turn("t3", "peer-b", "A term of 22 months."),
    _turn("t4", "treated-judge", "The panel imposes 26 months."),
]
BASELINE = [
    _turn("t1", "treated-judge", "I would impose 18 months."),
    _turn("t2", "peer-a", "I lean toward 20 months."),
    _turn("t3", "peer-b", "A term of 20 months."),
    _turn("t4", "treated-judge", "The panel imposes 20 months."),
]


def test_compute_decomposition():
    effects = panel_effects.compute(
        CONFIGURED, BASELINE, {"treated-judge"},
        endpoint_name="months", parse=judicial.parse_months)
    # direct: treated turns t1 (+12) and t4 (+6) → 9
    assert effects.direct == 9.0 and effects.direct_n == 2
    # spillover: t2 (+4), t3 (+2) → 3
    assert effects.spillover == 3.0 and effects.spillover_n == 2
    # group: last turn t4 (+6)
    assert effects.group == 6.0 and effects.group_n == 1
    assert math.isclose(effects.transmission_ratio, 3.0 / 9.0)
    assert math.isclose(effects.amplification, 6.0 / 9.0)
    assert effects.dropped_turns == 0


def test_compute_drops_unparseable_and_unpaired():
    configured = CONFIGURED + [_turn("t5", "peer-a", "no number here")]
    baseline = BASELINE[:3]  # t4, t5 unpaired/unparseable
    effects = panel_effects.compute(
        configured, baseline, {"treated-judge"},
        endpoint_name="months", parse=judicial.parse_months,
        group_turn_id="t1")
    assert effects.dropped_turns == 2
    assert effects.direct_n == 1  # only t1 pairs
    assert effects.group == 12.0


def test_treated_agent_ids_from_dict_and_object():
    scenario = {"agents": [
        {"id": "a", "variantArtifactPath": "runs/x.json"},
        {"id": "b"},
    ]}
    assert panel_effects.treated_agent_ids(scenario) == {"a"}

    class Agent:
        def __init__(self, id, path):
            self.id = id
            self.variant_artifact_path = path

    class Scenario:
        agents = [Agent("c", "p.json"), Agent("d", None)]

    assert panel_effects.treated_agent_ids(Scenario()) == {"c"}


def test_write_panel_effects_csv(tmp_path):
    for condition, turns in (("configured", CONFIGURED), ("baseline", BASELINE)):
        directory = tmp_path / condition
        directory.mkdir()
        with open(directory / "turns.jsonl", "w") as handle:
            for turn in turns:
                handle.write(json.dumps(turn) + "\n")
    scenario = {"agents": [{"id": "treated-judge", "variantArtifactPath": "v.json"},
                           {"id": "peer-a"}, {"id": "peer-b"}]}
    rows = panel_effects.write_panel_effects(
        str(tmp_path), scenario, endpoints={"months": judicial.parse_months})
    assert len(rows) == 1
    with open(tmp_path / "panel-effects.csv", encoding="utf-8") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["endpoint"] == "months"
    assert float(row["direct"]) == 9.0
    assert math.isclose(float(row["transmissionRatio"]), 3.0 / 9.0, rel_tol=1e-5)


def test_write_panel_effects_requires_both_arms(tmp_path):
    (tmp_path / "configured").mkdir()
    with open(tmp_path / "configured" / "turns.jsonl", "w") as handle:
        handle.write(json.dumps(CONFIGURED[0]) + "\n")
    rows = panel_effects.write_panel_effects(
        str(tmp_path), {"agents": []}, endpoints={"months": judicial.parse_months})
    assert rows == []
    assert not os.path.exists(tmp_path / "panel-effects.csv")


# --- exposure-based spillover (review finding 4) -----------------------------

def _exposure_scenario():
    from steerlab_server.experiment import multi_agent as ma
    return ma.Scenario(
        name="panel",
        agents=[ma.Agent(id="a", name="A", variant_artifact_path="v.json"),
                ma.Agent(id="b", name="B"),
                ma.Agent(id="c", name="C")],
        turns=[
            # Private notes: speakerOnly, so nobody hears anybody yet.
            ma.Turn(id="p_a", title="notes A", speaker_agent_id="a",
                    prompt_template="x", routing="speakerOnly"),
            ma.Turn(id="p_b", title="notes B", speaker_agent_id="b",
                    prompt_template="x", routing="speakerOnly"),
            # A (treated) now speaks to B only.
            ma.Turn(id="memo_a", title="memo A", speaker_agent_id="a",
                    prompt_template="x", routing="selected",
                    routed_agent_ids=["b"]),
            # B has heard A: exposed.
            ma.Turn(id="memo_b", title="memo B", speaker_agent_id="b",
                    prompt_template="x", routing="selected",
                    routed_agent_ids=["c"]),
            # C heard B, who heard A: exposed transitively.
            ma.Turn(id="memo_c", title="memo C", speaker_agent_id="c",
                    prompt_template="x", routing="all"),
        ])


def test_exposure_tracks_routing_not_just_seat():
    from steerlab_server.experiment import panel_effects as pe

    exposed = pe.exposure_by_turn(_exposure_scenario(), {"a"})

    # B speaks before hearing anything from A — cannot carry the intervention.
    assert exposed["p_b"] is False
    # After A's memo reaches B, B is exposed.
    assert exposed["memo_b"] is True
    # C never heard A directly, only via B — second-order propagation counts.
    assert exposed["memo_c"] is True


def test_unexposed_turns_are_not_counted_as_spillover():
    """Counting a pre-exposure turn as spillover dilutes the estimate toward
    'no transmission': its prompt is byte-identical in both arms, so it is a
    structural zero."""
    from steerlab_server.experiment import panel_effects as pe

    scenario = _exposure_scenario()
    exposed = pe.exposure_by_turn(scenario, {"a"})
    configured = [{"turnID": t.id, "speakerAgentID": t.speaker_agent_id,
                   "output": "10" if t.id == "memo_b" else "0"}
                  for t in scenario.turns]
    baseline = [{"turnID": t.id, "speakerAgentID": t.speaker_agent_id,
                 "output": "0"} for t in scenario.turns]

    row = pe.compute(configured, baseline, {"a"}, endpoint_name="v",
                     parse=lambda t: float(t), exposed_turns=exposed)

    # Only the two exposed untreated turns feed spillover (memo_b, memo_c);
    # B's pre-exposure private note goes to the placebo channel instead.
    assert row.spillover_n == 2
    assert row.unexposed_n == 1
    assert row.spillover == 5.0          # (10 + 0) / 2
    # Diluted answer under the old rule would have been (10+0+0)/3 = 3.33.


def test_the_unexposed_channel_is_a_placebo_check():
    """Pre-exposure turns should sit at ~zero; a non-zero value there means
    something leaked, which is worth surfacing rather than discarding."""
    from steerlab_server.experiment import panel_effects as pe

    scenario = _exposure_scenario()
    exposed = pe.exposure_by_turn(scenario, {"a"})
    configured = [{"turnID": t.id, "speakerAgentID": t.speaker_agent_id,
                   "output": "0"} for t in scenario.turns]
    baseline = list(configured)

    row = pe.compute(configured, baseline, {"a"}, endpoint_name="v",
                     parse=lambda t: float(t), exposed_turns=exposed)

    assert row.unexposed_n == 1 and row.unexposed == 0.0


def test_exposure_follows_what_the_prompt_actually_reads():
    """Routing records who COULD hear; the prompt decides who DOES. Both
    directions were wrong when only routing was consulted."""
    from steerlab_server.experiment import multi_agent as ma
    from steerlab_server.experiment import panel_effects as pe

    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", variant_artifact_path="v"),
                ma.Agent(id="b", name="B")],
        turns=[
            ma.Turn(id="t_a", title="A speaks", speaker_agent_id="a",
                    prompt_template="x", output_label="amemo", routing="all"),
            # Routed to, but reads no context: NOT exposed.
            ma.Turn(id="blind", title="B ignores context", speaker_agent_id="b",
                    prompt_template="x", routing="all",
                    include_speaker_context=False),
            # Reads the routed context: exposed.
            ma.Turn(id="reads", title="B reads", speaker_agent_id="b",
                    prompt_template="x", routing="all"),
            # No routing at all, but interpolates a treated output: exposed.
            ma.Turn(id="interp", title="B interpolates", speaker_agent_id="b",
                    prompt_template="see {{outputs.amemo}}", routing="none",
                    include_speaker_context=False),
        ])

    exposed = pe.exposure_by_turn(scenario, {"a"})
    assert exposed["blind"] is False
    assert exposed["reads"] is True
    assert exposed["interp"] is True
