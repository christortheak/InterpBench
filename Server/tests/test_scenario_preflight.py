"""Scenario-aware token preflight (plan A4).

A panel accumulates context across turns, so it is the study kind most likely
to overflow — and the only backstop was the in-generation check, i.e. dying
part-way through a late turn after a queue wait and a 27B load. These pin the
two-tier verdict: refuse only what cannot fit whatever the model writes, warn
about what the accumulating deliberation is projected to break.
"""

import pytest

from steerlab_server.experiment import multi_agent, scenario_preflight


def _panel(turn_max_tokens=64, materials="rules", routing="all", turns=4):
    return multi_agent.Scenario(
        name="panel", base_model_id="m", shared_materials=materials,
        max_tokens=turn_max_tokens,
        agents=[multi_agent.Agent(id="a", name="A", base_model_id="m"),
                multi_agent.Agent(id="b", name="B", base_model_id="m")],
        turns=[multi_agent.Turn(id=f"t{i}", title=f"Turn {i}",
                                speaker_agent_id="a" if i % 2 == 0 else "b",
                                prompt_template="Speak. {{agent.context}}",
                                output_label=f"o{i}", routing=routing)
               for i in range(turns)])


def _report(monkeypatch, scenario, window, floor=10):
    """Preflight with a stub tokenizer: `floor` tokens for every rendered turn."""
    monkeypatch.setattr(scenario_preflight.token_preflight, "_tokenizer",
                        lambda *a, **k: object())
    monkeypatch.setattr(scenario_preflight.token_preflight, "context_window",
                        lambda *a, **k: window)
    monkeypatch.setattr(
        scenario_preflight.prompt_render, "render",
        lambda *a, **k: type("R", (), {"prompt_token_count": floor})())
    return scenario_preflight.preflight(scenario, model_id="m")


def test_context_accumulates_across_routed_turns(monkeypatch):
    """routing=all means every turn lands in every agent's context, so the
    projection must grow turn over turn — the whole reason a panel overflows."""
    report = _report(monkeypatch, _panel(), window=100_000)
    projected = [t["projectedPromptTokens"] for t in report["turns"]]

    assert projected == sorted(projected), projected
    assert projected[-1] > projected[0]


def test_speaker_only_routing_accumulates_far_less(monkeypatch):
    """Narrowing routing is one of the levers the message names, so it had
    better actually reduce the projection."""
    wide = _report(monkeypatch, _panel(routing="all"), window=100_000)
    narrow = _report(monkeypatch, _panel(routing="speakerOnly"), window=100_000)

    assert (narrow["turns"][-1]["projectedPromptTokens"]
            < wide["turns"][-1]["projectedPromptTokens"])


def test_a_turn_that_cannot_fit_at_all_refuses(monkeypatch):
    """Floor over budget is certain regardless of what the model generates."""
    report = _report(monkeypatch, _panel(), window=200, floor=500)

    assert report["certainOverflow"]
    message = scenario_preflight.refusal(report)
    assert message is not None
    # Names the turn, the numbers, and what to change.
    assert "turn 1" in message and "500 tokens" in message
    assert "Lower that turn's Max tokens" in message


def test_projected_overflow_warns_but_never_blocks(monkeypatch):
    """The projection charges every prior turn its FULL Max tokens, so it is a
    bound, not a prediction — blocking on it would refuse runs that fit."""
    # Floor fits comfortably; accumulation is what breaks the later turns.
    report = _report(monkeypatch, _panel(turn_max_tokens=200, turns=8),
                     window=700, floor=10)

    assert not report["certainOverflow"], "floor must fit for this to be a projection"
    assert report["projectedOverflow"]
    assert scenario_preflight.refusal(report) is None
    advisory = scenario_preflight.advisory(report)
    assert "worst-case projection" in advisory and "NOT blocked" in advisory


def test_a_panel_that_fits_says_nothing(monkeypatch):
    report = _report(monkeypatch, _panel(), window=100_000)

    assert not report["certainOverflow"] and not report["projectedOverflow"]
    assert scenario_preflight.refusal(report) is None
    assert scenario_preflight.advisory(report) is None


def test_unreadable_tokenizer_never_blocks_a_run(monkeypatch):
    """A preflight that cannot RUN yields to the in-generation backstop rather
    than becoming a new way to fail."""
    def boom(*a, **k):
        raise scenario_preflight.token_preflight.PreflightError("no cached tokenizer")

    monkeypatch.setattr(scenario_preflight.token_preflight, "_tokenizer", boom)
    with pytest.raises(scenario_preflight.token_preflight.PreflightError):
        scenario_preflight.preflight(_panel(), model_id="m")


# --- E1: transcripts are the shardable unit ---------------------------------

def test_transcripts_partition_across_shards_exactly_once():
    """Replicates are independent play-throughs sharing no state, which is the
    property sharding needs. Every transcript must land on exactly one worker
    — none dropped, none duplicated."""
    from steerlab_server.experiment import sharding

    conditions = ["configured", "baseline"]
    replicates = 8
    all_keys = [(c, r) for c in conditions for r in range(replicates)]

    seen = []
    for index in range(4):
        plan = sharding.plan_shard(
            sharding.ShardSpec(index=index, count=4),
            all_keys=all_keys, condition_names=conditions)
        seen.extend(plan.keys)

    assert sorted(seen) == sorted(all_keys)
    assert len(seen) == len(set(seen)), "a transcript ran on two workers"


def test_sharding_a_single_transcript_panel_refuses_with_the_remedy():
    """Turns cannot be split — that refusal stands. But it has to name the
    axis that CAN be sharded, or the researcher is just stuck."""
    from steerlab_server.experiment import tasks

    source = tasks.run.__doc__ or ""
    # The refusal text lives in run(); assert the guidance is present in the
    # module so it cannot be silently reduced back to "does not apply".
    import inspect
    body = inspect.getsource(tasks.run)
    assert "cannot shard a single-transcript panel study" in body
    assert "samplesPerItem 8 with --shard 0/4" in body


# --- E2 / F1 ---------------------------------------------------------------

def test_a_panel_submission_can_be_sized():
    """Preflight used to be blind for panels — sizing returned None for exactly
    the runs hardest to size, and replicates multiplied that."""
    import json as _json
    from steerlab_server.api import submissions
    from steerlab_server.experiment.manifest import Manifest

    scenario = _json.dumps({"name": "p", "turns": [{"id": f"t{i}"} for i in range(6)]})
    manifest = Manifest.from_dict({
        "name": "p", "modelID": "m", "studyKind": "multiAgent",
        "multiAgentIncludeBaseline": True, "samplesPerItem": 4,
        "multiAgentScenarioPath": "prompts/panels/p.json"})

    # 6 turns x 2 conditions x 4 replicates
    assert submissions._planned_records(manifest, scenario) == 48


def test_sizing_a_panel_without_a_baseline_halves_it():
    import json as _json
    from steerlab_server.api import submissions
    from steerlab_server.experiment.manifest import Manifest

    scenario = _json.dumps({"name": "p", "turns": [{"id": f"t{i}"} for i in range(6)]})
    manifest = Manifest.from_dict({
        "name": "p", "modelID": "m", "studyKind": "multiAgent",
        "multiAgentIncludeBaseline": False, "samplesPerItem": 1,
        "multiAgentScenarioPath": "prompts/panels/p.json"})

    assert submissions._planned_records(manifest, scenario) == 6


def test_authoring_advisories_catch_the_silent_failures():
    """All three fail silently at run time — the run completes and the prompts
    are quietly wrong. Advisory, not a gate."""
    from steerlab_server.experiment import multi_agent as ma

    scenario = ma.Scenario(
        name="p",
        # baseModelID because the tail of this test calls validate(), which
        # refuses a seat with no model of its own.
        agents=[ma.Agent(id="a", name="A", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="dup"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="a",
                    prompt_template="see {{outputs.nope}}", output_label="dup"),
        ])
    notes = ma.advisories(scenario)

    assert any("no earlier turn produces the label 'nope'" in n for n in notes)
    assert any("reuses the output label 'dup'" in n for n in notes)
    # Advisory only: validate() still passes, so a draft is never blocked.
    ma.validate(scenario)


def test_a_clean_panel_draws_no_advisories():
    from steerlab_server.experiment import multi_agent as ma

    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="first"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="a",
                    prompt_template="see {{outputs.first}}", output_label="second"),
        ])
    assert ma.advisories(scenario) == []


def test_a_default_label_collision_is_caught_too():
    """An explicit label that collides with the turn_<n> default the runner
    assigns to an unlabelled turn silently wins its interpolations."""
    from steerlab_server.experiment import multi_agent as ma

    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go"),  # unlabelled -> turn_1
            ma.Turn(id="t2", title="Two", speaker_agent_id="a",
                    prompt_template="go", output_label="turn_1"),
        ])
    assert any("turn_1" in n for n in ma.advisories(scenario))
