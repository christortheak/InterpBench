"""Multi-agent runner: validation, routing, context propagation, and output-
label interpolation. Uses a stub generate() — no model."""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import multi_agent


def _scenario():
    return multi_agent.Scenario(
        name="pd", base_model_id="m", shared_materials="The rules.",
        agents=[multi_agent.Agent(id="a", name="Alice", base_model_id="m"),
                multi_agent.Agent(id="b", name="Bob", base_model_id="m")],
        turns=[
            multi_agent.Turn(id="t1", title="Alice opens", speaker_agent_id="a",
                             prompt_template="Say something. {{scenario.materials}}",
                             output_label="alice1", routing="all"),
            multi_agent.Turn(id="t2", title="Bob replies", speaker_agent_id="b",
                             prompt_template="Reply to: {{outputs.alice1}}\nContext: {{agent.context}}",
                             output_label="bob1", routing="speakerOnly"),
        ])


def test_validate_rejects_bad_speaker():
    s = _scenario()
    s.turns[0].speaker_agent_id = "ghost"
    with pytest.raises(multi_agent.ScenarioError):
        multi_agent.validate(s)


def test_routing_modes():
    s = _scenario()
    assert multi_agent._routed_ids(s.turns[0], s.agents) == ["a", "b"]   # all
    assert multi_agent._routed_ids(s.turns[1], s.agents) == ["b"]        # speakerOnly


def test_run_propagates_context_and_labels(tmp_path, monkeypatch):
    calls = []

    def stub_generate(model, prompt, **kwargs):
        calls.append(prompt)
        return f"OUT{len(calls)}"

    monkeypatch.setattr(multi_agent, "generate", stub_generate)
    model = SimpleNamespace(model_id="m", revision="r")
    run_dir = str(tmp_path)
    multi_agent.run_scenario(model, _scenario(), run_dir=run_dir, scenario_hash="h")

    # Turn 1 prompt included the shared materials.
    assert "The rules." in calls[0]
    # Turn 2 prompt interpolated Alice's output (OUT1) and saw it in Bob's context
    # (routing="all" on turn 1 added it to Bob's context).
    assert "OUT1" in calls[1]
    assert "Alice opens" in calls[1]  # context entry header

    turns = [json.loads(l) for l in open(os.path.join(run_dir, "turns.jsonl"))]
    assert [t["speakerName"] for t in turns] == ["Alice", "Bob"]
    assert turns[0]["outputLabel"] == "alice1"
    assert os.path.exists(os.path.join(run_dir, "transcript.md"))
    report = json.load(open(os.path.join(run_dir, "report.json")))
    assert report["turnCount"] == 2 and report["scenarioHash"] == "h"


def test_positive_temperature_is_supported_and_seeded(tmp_path, monkeypatch):
    """A5: warm panel runs are supported, and reproducible on this substrate —
    each turn is seeded from (experimentHash, condition, turn id, replicate)."""
    seen = []

    def stub_generate(model, prompt, **kwargs):
        seen.append(kwargs.get("temperature"))
        return "out"

    monkeypatch.setattr(multi_agent, "generate", stub_generate)
    s = _scenario()
    s.temperature = 0.7
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"), s,
                             run_dir=str(tmp_path), experiment_hash="eh")

    assert seen == [0.7, 0.7], "the scenario temperature must reach the sampler"
    turns = [json.loads(l) for l in open(os.path.join(tmp_path, "turns.jsonl"))]
    assert all(t["temperature"] == 0.7 for t in turns)
    # Distinct per-turn seeds, and each one actually recorded (a warm turn that
    # stamped a null seed would be unreproducible without saying so).
    seeds = [t["seed"] for t in turns]
    assert all(isinstance(x, int) for x in seeds) and len(set(seeds)) == len(seeds)
    report = json.load(open(os.path.join(tmp_path, "report.json")))
    assert report["seedPolicy"] == "derivedSHA256"


def test_greedy_turns_record_no_seed(tmp_path, monkeypatch):
    """Temperature 0 draws no RNG, so naming a seed would imply a stream that
    was never used."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"), _scenario(),
                             run_dir=str(tmp_path), experiment_hash="eh")

    turns = [json.loads(l) for l in open(os.path.join(tmp_path, "turns.jsonl"))]
    assert all(t["seed"] is None for t in turns)
    assert json.load(open(os.path.join(tmp_path, "report.json")))["seedPolicy"] == "greedy"


def test_replicates_draw_distinct_seeds(tmp_path, monkeypatch):
    """Replicate index is part of the seed derivation, so two play-throughs of
    the same scenario are independent — the property that makes replicates
    (not turns) the shardable unit."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    s = _scenario()
    s.temperature = 0.7

    def seeds_for(replicate):
        d = tmp_path / f"rep{replicate}"
        d.mkdir()
        multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"), s,
                                 run_dir=str(d), experiment_hash="eh",
                                 replicate_index=replicate)
        return [json.loads(l)["seed"] for l in open(os.path.join(d, "turns.jsonl"))]

    assert not set(seeds_for(0)) & set(seeds_for(1))


def test_manifest_temperature_overrides_the_scenario(tmp_path, monkeypatch):
    """The STUDY manifest owns measured-run sampling policy; the scenario's own
    temperature is the authoring convenience."""
    seen = []
    monkeypatch.setattr(multi_agent, "generate",
                        lambda m, p, **k: (seen.append(k.get("temperature")), "out")[1])
    s = _scenario()
    s.temperature = 0.7
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"), s,
                             run_dir=str(tmp_path), temperature=0.0,
                             experiment_hash="eh")
    assert seen == [0.0, 0.0]


def test_warm_turns_generate_inside_the_seeded_fork(tmp_path, monkeypatch):
    """The seed must be APPLIED, not merely derived and stamped. Asserts the
    generate call happens inside _seeded_generation's RNG fork, carrying the
    same seed the turn record reports."""
    from steerlab_server.experiment import tasks as tasks_mod

    active, wrapped = [], []

    @contextmanager
    def spy_seeded(temperature, seed):
        active.append(seed)
        yield
        active.pop()

    def stub_generate(model, prompt, **kwargs):
        # Non-empty `active` proves we are lexically inside the fork.
        wrapped.append(active[-1] if active else None)
        return "out"

    monkeypatch.setattr(tasks_mod, "_seeded_generation", spy_seeded)
    monkeypatch.setattr(multi_agent, "generate", stub_generate)
    s = _scenario()
    s.temperature = 0.7
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"), s,
                             run_dir=str(tmp_path), experiment_hash="eh")

    turns = [json.loads(l) for l in open(os.path.join(tmp_path, "turns.jsonl"))]
    assert wrapped == [t["seed"] for t in turns] and all(x is not None for x in wrapped)


def test_negative_temperature_refuses_with_the_value(tmp_path):
    with pytest.raises(multi_agent.ScenarioError, match="-1"):
        multi_agent.run_scenario(SimpleNamespace(model_id="m"), _scenario(),
                                 run_dir=str(tmp_path), temperature=-1.0)


def _slug_fixture():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(here, "prompts", "fixtures", "panel-slug", "slugs.json"),
              encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def test_panel_slug_matches_the_cross_engine_fixture():
    """B3: both engines name files in prompts/panels/, so the slug rule must be
    ONE rule. The Swift twin reads the same fixture."""
    from steerlab_server.experiment.experiment_store import _slugify

    for case in _slug_fixture():
        assert _slugify(case["name"]) == case["slug"], case["name"]


def test_saving_the_same_panel_twice_updates_in_place(tmp_path):
    """Same name is an update, not a new panel — and byte-identical, so a
    pinned input's hash does not move for a no-op save."""
    s = _scenario()
    first = multi_agent.save_scenario(s, root=str(tmp_path))
    second = multi_agent.save_scenario(s, root=str(tmp_path))

    assert first["path"] == second["path"]
    assert first["hash"] == second["hash"]
    assert not second["disambiguated"]
    assert len(os.listdir(os.path.join(tmp_path, "prompts", "panels"))) == 1


def test_distinct_panels_that_slugify_alike_do_not_clobber(tmp_path):
    """'Judicial Panel' and 'Judicial: Panel' both slugify to judicial-panel.
    The newcomer is disambiguated rather than silently overwriting."""
    a = _scenario()
    a.name = "Judicial Panel"
    b = _scenario()
    b.name = "Judicial: Panel"

    first = multi_agent.save_scenario(a, root=str(tmp_path))
    second = multi_agent.save_scenario(b, root=str(tmp_path))

    assert first["path"] != second["path"]
    assert not first["disambiguated"] and second["disambiguated"]
    assert second["path"].endswith("judicial-panel-2.json")
    # Both survive, and both are listed.
    names = {p["name"] for p in multi_agent.list_scenarios(root=str(tmp_path))}
    assert names == {"Judicial Panel", "Judicial: Panel"}


def test_unreadable_neighbour_is_written_alongside_not_over(tmp_path):
    """A corrupt file cannot be identified, so it is treated as a different
    panel — writing beside it beats destroying it."""
    base = os.path.join(tmp_path, "prompts", "panels")
    os.makedirs(base, exist_ok=True)
    corrupt = os.path.join(base, "panel.json")
    with open(corrupt, "w", encoding="utf-8") as handle:
        handle.write("{not json")

    s = _scenario()
    s.name = "panel"
    result = multi_agent.save_scenario(s, root=str(tmp_path))

    assert result["disambiguated"] and result["path"].endswith("panel-2.json")
    with open(corrupt, encoding="utf-8") as handle:
        assert handle.read() == "{not json"


def test_wrong_base_model_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "x")
    s = _scenario()
    s.agents[0].base_model_id = "other/model"
    with pytest.raises(multi_agent.ScenarioError):
        multi_agent.run_scenario(SimpleNamespace(model_id="m"), s, run_dir=str(tmp_path))


def test_provider_allows_multiple_base_models(tmp_path, monkeypatch):
    calls = []

    def stub_generate(model, prompt, **kwargs):
        calls.append((model.model_id, kwargs.get("model_id")))
        return f"from {model.model_id}"

    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "r", device="cpu")

    monkeypatch.setattr(multi_agent, "generate", stub_generate)
    s = _scenario()
    s.agents[0].base_model_id = "model/a"
    s.agents[1].base_model_id = "model/b"
    multi_agent.run_scenario(None, s, run_dir=str(tmp_path), model_provider=provider)

    assert calls == [("model/a", "model/a"), ("model/b", "model/b")]
    turns = [json.loads(l) for l in open(os.path.join(tmp_path, "turns.jsonl"))]
    assert [t["modelID"] for t in turns] == ["model/a", "model/b"]


# --- D1: transcript is the unit of analysis ---------------------------------

def test_transcript_level_diffs_aggregates_within_a_play_through():
    """Turn diffs collapse to one value per transcript, so downstream `n`
    counts transcripts rather than dependent turns."""
    from steerlab_server.experiment.tasks import _transcript_level_diffs

    # Two transcripts, three turns each. Transcript 0 mean +2, transcript 1 +8.
    values = {"t1@0": 3.0, "t2@0": 4.0, "t3@0": 5.0,
              "t1@1": 9.0, "t2@1": 10.0, "t3@1": 11.0}
    base = {"t1@0": 1.0, "t2@0": 2.0, "t3@0": 3.0,
            "t1@1": 1.0, "t2@1": 2.0, "t3@1": 3.0}

    assert _transcript_level_diffs(values, base) == [2.0, 8.0]


def test_replicate_keying_keeps_transcripts_apart():
    """Without the replicate in the key, the same turn id from two
    play-throughs lands in one cell and the between-transcript variation the
    estimator exists to measure is averaged away before it is ever seen."""
    from steerlab_server.experiment.tasks import _key_records_by_transcript

    keyed = _key_records_by_transcript([
        {"promptID": "t1", "replicateIndex": 0},
        {"promptID": "t1", "replicateIndex": 1},
        {"promptID": "t1"},  # pre-replicate record normalizes to 0
    ])

    assert [r["promptID"] for r in keyed] == ["t1@0", "t1@1", "t1@0"]


def test_clustering_does_not_understate_uncertainty():
    """The point of D1. Turns within a transcript agree with each other far
    more than transcripts agree with each other; treating the turns as
    independent draws reports an interval the design has not earned."""
    from steerlab_server.experiment import study_stats
    from steerlab_server.experiment.tasks import _transcript_level_diffs

    # Two transcripts that disagree sharply; turns within each agree closely.
    values, base = {}, {}
    for turn in range(6):
        values[f"t{turn}@0"] = 1.0
        base[f"t{turn}@0"] = 0.0
        values[f"t{turn}@1"] = 9.0
        base[f"t{turn}@1"] = 0.0

    naive = [values[k] - base[k] for k in sorted(values)]      # 12 "observations"
    clustered = _transcript_level_diffs(values, base)          # 2 transcripts

    naive_row = study_stats.effect_row("configured", "wordCount", naive)
    clustered_row = study_stats.effect_row("configured", "wordCount", clustered)

    assert naive_row.ci.n == 12 and clustered_row.ci.n == 2
    # Same central estimate, honest width.
    assert abs(naive_row.ci.mean - clustered_row.ci.mean) < 1e-9
    naive_width = naive_row.ci.ci_upper - naive_row.ci.ci_lower
    clustered_width = clustered_row.ci.ci_upper - clustered_row.ci.ci_lower
    assert clustered_width > naive_width


# --- E1: turn-level checkpoint --------------------------------------------

def test_a_killed_transcript_resumes_from_its_last_completed_turn(tmp_path, monkeypatch):
    """The point of E1. A panel that dies on a late turn must not restart at
    turn 1 — on a cluster that is a whole queue wait and model load thrown
    away."""
    calls = []

    def die_on_second(model, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("walltime")
        return f"OUT{len(calls)}"

    monkeypatch.setattr(multi_agent, "generate", die_on_second)
    model = SimpleNamespace(model_id="m", revision="r")

    with pytest.raises(RuntimeError, match="walltime"):
        multi_agent.run_scenario(model, _scenario(), run_dir=str(tmp_path))

    # Turn 1 survived the kill because it was flushed as it landed.
    done = multi_agent._completed_turns(os.path.join(tmp_path, "turns.jsonl"))
    assert list(done) == ["t1"]

    # Re-run: turn 1 is replayed from disk, only turn 2 generates.
    calls.clear()
    monkeypatch.setattr(multi_agent, "generate",
                        lambda m, p, **k: (calls.append(p), "OUT2-redone")[1])
    multi_agent.run_scenario(model, _scenario(), run_dir=str(tmp_path))

    assert len(calls) == 1, "only the unfinished turn should regenerate"
    turns = [json.loads(l) for l in open(os.path.join(tmp_path, "turns.jsonl"))]
    assert [t["turnID"] for t in turns] == ["t1", "t2"]
    assert turns[0]["output"] == "OUT1", "the surviving turn must not be rewritten"


def test_resumed_turns_still_feed_downstream_context(tmp_path, monkeypatch):
    """A replayed turn has to route into context and output labels exactly as
    a generated one would, or the resumed transcript diverges from the
    uninterrupted one."""
    calls = []

    def die_on_second(model, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("walltime")
        return "ALICE-SAID"

    monkeypatch.setattr(multi_agent, "generate", die_on_second)
    model = SimpleNamespace(model_id="m", revision="r")
    with pytest.raises(RuntimeError):
        multi_agent.run_scenario(model, _scenario(), run_dir=str(tmp_path))

    calls.clear()
    monkeypatch.setattr(multi_agent, "generate",
                        lambda m, p, **k: (calls.append(p), "out")[1])
    multi_agent.run_scenario(model, _scenario(), run_dir=str(tmp_path))

    # Turn 2 interpolates {{outputs.alice1}} and sees Alice in its context.
    assert "ALICE-SAID" in calls[0]
    assert "Alice opens" in calls[0]


def test_a_torn_final_line_is_discarded_not_repaired(tmp_path):
    """Killed mid-write. Half a turn is not a turn; regenerating it is cheap
    and correct, whereas trusting it would poison every downstream turn."""
    path = os.path.join(tmp_path, "turns.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"turnID": "t1", "output": "fine"}) + "\n")
        handle.write('{"turnID": "t2", "outp')

    done = multi_agent._completed_turns(path)
    assert list(done) == ["t1"]


# --- review fixes: CRN, torn tail, prose endpoints --------------------------

def test_arms_share_a_random_stream_at_the_same_replicate(tmp_path, monkeypatch):
    """CRN (2026-07-27). Replicate r of the configured arm and replicate r of
    the baseline arm must draw the SAME seed — 'same dice, different
    intervention' — or the paired transcript test buys no variance reduction
    while implying a dependence that does not exist."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    s = _scenario()
    s.temperature = 0.7

    def seeds(condition, replicate):
        d = tmp_path / f"{condition}-{replicate}"
        d.mkdir(exist_ok=True)
        multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"), s,
                                 run_dir=str(d), condition_name=condition,
                                 replicate_index=replicate, experiment_hash="eh")
        return [json.loads(l)["seed"] for l in open(os.path.join(d, "turns.jsonl"))]

    assert seeds("configured", 0) == seeds("baseline", 0), "arms must share the stream"
    # Replicates still differ from each other — otherwise every transcript
    # would be identical and there would be nothing to average over.
    assert seeds("configured", 0) != seeds("configured", 1)


def test_a_torn_tail_is_truncated_not_just_skipped(tmp_path, monkeypatch):
    """Skipping a fragment leaves it on disk; the next append lands on the same
    line, producing a permanently malformed record the study wrapper's strict
    parse dies on later."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    path = os.path.join(tmp_path, "turns.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "turnID": "t1", "turnIndex": 1, "title": "Alice opens",
            "speakerAgentID": "a", "speakerName": "Alice", "outputLabel": "alice1",
            "prompt": "p", "output": "kept", "routedAgentIDs": ["a", "b"]}) + "\n")
        handle.write('{"turnID": "t2", "outp')

    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _scenario(), run_dir=str(tmp_path))

    # Every line parses, and the surviving turn was not regenerated.
    turns = [json.loads(line) for line in open(path)]
    assert [t["turnID"] for t in turns] == ["t1", "t2"]
    assert turns[0]["output"] == "kept"


def test_prose_endpoints_reach_the_analyzer():
    """A panel produces prose: no choice instrument, often no pinned taxonomy.
    Without wordCount/distinct2 the endpoint set was EMPTY, so the transcript
    clustering ran on nothing and the study silently reported no effect
    sizes."""
    from steerlab_server.experiment.tasks import _endpoint_values

    records = [
        {"condition": "baseline", "promptID": "t1", "output": "a",
         "wordCount": 10, "distinct2": 0.5},
        {"condition": "configured", "promptID": "t1", "output": "b",
         "wordCount": 20, "distinct2": 0.9},
    ]
    endpoints = _endpoint_values(records)

    assert endpoints["wordCount"]["configured"]["t1"] == 20.0
    assert endpoints["distinct2"]["baseline"]["t1"] == 0.5


def test_a_complete_record_missing_only_its_newline_survives(tmp_path, monkeypatch):
    """Loading before truncating LOST data: a record whose JSON landed but
    whose newline did not parses fine, so it entered `done` (never
    regenerated) and was then truncated away (no longer on disk). Truncating
    first — and recognising complete JSON — keeps it."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "regenerated")
    path = os.path.join(tmp_path, "turns.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "turnID": "t1", "turnIndex": 1, "title": "Alice opens",
            "speakerAgentID": "a", "speakerName": "Alice", "outputLabel": "alice1",
            "prompt": "p", "output": "PRECIOUS", "routedAgentIDs": []}))
        # no trailing newline — the crash landed between the two writes

    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _scenario(), run_dir=str(tmp_path))

    turns = [json.loads(line) for line in open(path)]
    assert [t["turnID"] for t in turns] == ["t1", "t2"]
    assert turns[0]["output"] == "PRECIOUS", "a finished turn was destroyed"


def test_the_accelerator_cache_is_trimmed_between_turns(tmp_path, monkeypatch):
    """A panel is 48 sequential generations in one process. The caching
    allocator was trimmed only on model EVICTION — which never happens inside
    a run — so on MPS the accounting climbed until a 33 MiB allocation was
    refused while real usage was 14 GiB on a 64 GiB machine."""
    trims = []
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    monkeypatch.setattr(multi_agent, "_trim_accelerator_cache",
                        lambda model: trims.append(getattr(model, "device", None)))
    # The probe AND the trim are MOCKED, deliberately: this fake model claims
    # device "mps" while nothing was ever allocated there, and both real
    # helpers used to walk straight into the MPS allocator and segfault the
    # test process. The claim under test is the trim cadence, not either
    # native call.
    monkeypatch.setattr(multi_agent, "accelerator_memory", lambda model: None)

    multi_agent.run_scenario(
        SimpleNamespace(model_id="m", revision="r", device="mps"),
        _scenario(), run_dir=str(tmp_path))

    # Once per generated turn, not once per run.
    assert trims == ["mps", "mps"]


def test_trimming_never_sinks_a_run(tmp_path, monkeypatch):
    """Best-effort: failing to release cache is not a reason to lose a
    transcript that is otherwise complete."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")

    def boom(model):
        raise RuntimeError("allocator unavailable")

    monkeypatch.setattr(multi_agent, "_trim_accelerator_cache", boom)

    with pytest.raises(RuntimeError):
        multi_agent.run_scenario(
            SimpleNamespace(model_id="m", revision="r", device="mps"),
            _scenario(), run_dir=str(tmp_path))
    # The turn that completed before the raise is still durably on disk.
    assert multi_agent._completed_turns(os.path.join(tmp_path, "turns.jsonl"))


def test_the_real_trim_helper_tolerates_any_device(tmp_path):
    for device in ("mps", "cuda:0", "cpu", "", None):
        multi_agent._trim_accelerator_cache(SimpleNamespace(device=device))


def _fake_torch_for_trim(calls, *, built=True, available=True):
    """A torch stand-in whose ``device`` marker class is ``SimpleNamespace``,
    so a test can hand the helper an object that passes ``isinstance`` the way
    a real ``torch.device`` would."""
    return SimpleNamespace(
        device=SimpleNamespace,
        backends=SimpleNamespace(mps=SimpleNamespace(
            is_built=lambda: built, is_available=lambda: available)),
        mps=SimpleNamespace(empty_cache=lambda: calls.append("mps")),
        cuda=SimpleNamespace(is_available=lambda: True,
                             empty_cache=lambda: calls.append("cuda")))


def test_a_device_string_never_reaches_the_allocator_trim(monkeypatch):
    """The 2026-08-06 review round-2 finding, and the exact twin of the probe
    finding above. ``torch.mps.empty_cache()`` is a native call into an
    allocator that must already exist: on a process where the backend was
    never initialized it SEGFAULTS rather than raising, so the helper's
    ``except Exception`` catches nothing and the panel dies with its
    transcript. A ``device`` attribute that is a plain STRING is a claim about
    placement, not evidence of it, so it gates the trim off entirely."""
    import sys

    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_for_trim(calls))
    for claimed in ("mps", "mps:0", "cuda", "cuda:0"):
        multi_agent._trim_accelerator_cache(SimpleNamespace(device=claimed))
    assert calls == [], "the allocator was touched on a device STRING"


def test_the_cache_trim_needs_a_built_and_available_backend(monkeypatch):
    """Same three conditions the memory probe requires: a genuine mps
    ``torch.device``, a BUILT backend, and an AVAILABLE one. Missing any of
    them and the trim is simply not taken — an unreleased cache block costs
    memory, a crashed process costs the transcript."""
    import sys

    device = SimpleNamespace(type="mps")
    for built, available in ((False, True), (True, False)):
        calls = []
        monkeypatch.setitem(sys.modules, "torch", _fake_torch_for_trim(
            calls, built=built, available=available))
        multi_agent._trim_accelerator_cache(SimpleNamespace(device=device))
        assert calls == [], "the allocator was touched on an unusable backend"


def test_the_cache_trim_is_mps_only(monkeypatch):
    """A CUDA cluster job owns its GPU and has neither the shared-pool
    contention nor the accounting drift this solves — and there
    empty_cache() is a pessimization, forcing cudaMalloc on the hot path of
    every turn."""
    import sys

    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_for_trim(calls))

    multi_agent._trim_accelerator_cache(
        SimpleNamespace(device=SimpleNamespace(type="cuda")))
    assert calls == [], "CUDA must not be trimmed"

    multi_agent._trim_accelerator_cache(
        SimpleNamespace(device=SimpleNamespace(type="mps")))
    assert calls == ["mps"]


def test_each_turn_reports_accelerator_memory(tmp_path, monkeypatch):
    """The probe has to reach the run log, or the next investigation is
    another round of guessing."""
    lines = []
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    monkeypatch.setattr(multi_agent, "_trim_accelerator_cache", lambda m: None)
    monkeypatch.setattr(multi_agent, "accelerator_memory",
                        lambda m: "mps allocated 1.00 GiB, driver 2.00 GiB")

    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r", device="mps"),
                             _scenario(), run_dir=str(tmp_path), log=lines.append)

    turn_lines = [l for l in lines if l.startswith("turn ")]
    assert len(turn_lines) == 2
    assert all("driver 2.00 GiB" in l for l in turn_lines)


def test_the_probe_never_fails_a_run(tmp_path):
    """A diagnostic that can sink a transcript is worse than no diagnostic."""
    from types import SimpleNamespace as NS
    assert multi_agent.accelerator_memory(NS(device="cpu")) is None
    assert multi_agent.accelerator_memory(NS()) is None
    assert multi_agent.accelerator_memory(NS(device=None)) is None


def test_a_device_string_never_reaches_the_allocator_probe():
    """The 2026-08-06 review finding. `torch.mps.current_allocated_memory()`
    is a native call into an allocator that must already exist: on a process
    where the backend was never initialized it SEGFAULTS rather than raising,
    so the probe's `except Exception` catches nothing and the run dies. A
    `device` attribute that is a plain string is a claim about placement, not
    evidence of it — only a real `torch.device` off the model's parameters
    is — so a string gates the probe off entirely."""
    from types import SimpleNamespace as NS
    for claimed in ("mps", "mps:0", "cuda", "cuda:0"):
        assert multi_agent.accelerator_memory(NS(device=claimed)) is None


def test_the_mps_probe_needs_a_built_and_available_backend(monkeypatch):
    """Three conditions, all required: a genuine mps `torch.device`, a BUILT
    backend, and an AVAILABLE one. Any of them missing and the reading is
    simply not taken — an absent diagnostic costs a log line, a crashed
    process costs the transcript."""
    import sys
    from types import SimpleNamespace as NS

    probed = []

    def fake_torch(built: bool, available: bool):
        return NS(
            device=NS,  # isinstance(device, torch.device) → the marker class
            backends=NS(mps=NS(is_built=lambda: built,
                               is_available=lambda: available)),
            mps=NS(driver_allocated_memory=lambda: probed.append("driver") or 0,
                   current_allocated_memory=lambda: probed.append("alloc") or 0),
            cuda=NS(is_available=lambda: False))

    device = NS(type="mps")
    for built, available in ((False, True), (True, False)):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(built, available))
        assert multi_agent.accelerator_memory(NS(device=device)) is None
    assert probed == [], "the allocator was touched on an unusable backend"

    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, True))
    reading = multi_agent.accelerator_memory(NS(device=device))
    assert reading is not None and reading.startswith("mps allocated")
    assert set(probed) == {"alloc", "driver"}


# --- scenario protocol templates -------------------------------------------
#
# A protocol template is the reusable half of a panel — seats, turn script,
# routing, caps, endpoint declarations — with the case materials deliberately
# left out. It is authored and instantiated on the Swift side; this engine's
# whole job is to never mistake one for a runnable panel. It is
# byte-compatible with a panel by design, so "it will not parse anyway" is not
# a defence: parsed as a scenario it would run, and three seats would
# deliberate about an empty record while the transcript looked entirely
# normal.


def _protocol_template(name="deliberative-appellate-panel-v1"):
    """A template exactly as the Swift store writes one: flat, marked."""
    return {
        "kind": "scenarioProtocolTemplate",
        "templateSchemaVersion": 1,
        "materialsChecklist": ["the case record", "procedural posture"],
        "schemaVersion": 1,
        "name": name,
        "description": "Three-judge appellate deliberation protocol.",
        "baseModelID": "",
        "sharedMaterials": "",
        "temperature": 0,
        "maxTokens": 2048,
        "agents": [{"id": "judge-1", "name": "Judge Whitfield",
                    "baseModelID": "", "systemPrompt": "You are Judge Whitfield."}],
        "turns": [{"id": "t1", "title": "Round 1", "speakerAgentID": "judge-1",
                   "promptTemplate": "Read the record. {{scenario.materials}}",
                   "outputLabel": "r1", "routing": "speakerOnly",
                   "routedAgentIDs": [], "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": 600}],
    }


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def test_protocol_template_beside_the_panels_is_not_listed_as_a_scenario(tmp_path):
    """It is skipped, quietly and without error, and the real panels beside it
    are unaffected."""
    panels = os.path.join(tmp_path, "prompts", "panels")
    multi_agent.save_scenario(_scenario(), root=str(tmp_path))
    _write_json(os.path.join(panels, "deliberative-appellate-panel-v1.json"),
                _protocol_template())

    listed = multi_agent.list_scenarios(root=str(tmp_path))

    assert {p["name"] for p in listed} == {"pd"}
    assert all(p["name"] != "deliberative-appellate-panel-v1" for p in listed)
    # And the file is still there — skipping is not deleting.
    assert os.path.exists(
        os.path.join(panels, "deliberative-appellate-panel-v1.json"))


def test_the_template_library_subdirectory_is_invisible_to_the_scan(tmp_path):
    """Templates live in prompts/panels/templates/. The panel scan is
    non-recursive, so the directory contributes nothing and raises nothing."""
    _write_json(
        os.path.join(tmp_path, "prompts", "panels", "templates",
                     "deliberative-appellate-panel-v1.json"),
        _protocol_template())

    assert multi_agent.list_scenarios(root=str(tmp_path)) == []


def test_from_dict_refuses_a_protocol_template(tmp_path):
    """The loud half: a template handed to the scenario parser refuses by
    name rather than yielding a panel with no case in it."""
    with pytest.raises(multi_agent.ScenarioError) as error:
        multi_agent.Scenario.from_dict(_protocol_template())

    assert "PROTOCOL TEMPLATE" in str(error.value)
    assert multi_agent.PROTOCOL_TEMPLATE_KIND in str(error.value)


def test_read_scenario_refuses_a_template_path(tmp_path):
    """Pinning a template as a study's panel fails at parse time, not at
    round three of a deliberation about nothing."""
    path = os.path.join(tmp_path, "prompts", "panels", "templates",
                        "deliberative-appellate-panel-v1.json")
    _write_json(path, _protocol_template())

    with pytest.raises(multi_agent.ScenarioError):
        multi_agent.read_scenario(path)


def test_an_ordinary_panel_still_parses(tmp_path):
    """The guard keys on the marker alone: a panel that happens to be
    materials-free is still a panel, and every pre-existing file is
    unaffected."""
    payload = _protocol_template()
    del payload["kind"]
    del payload["templateSchemaVersion"]
    del payload["materialsChecklist"]

    scenario = multi_agent.Scenario.from_dict(payload)

    assert scenario.name == "deliberative-appellate-panel-v1"
    assert scenario.shared_materials == ""
    assert len(scenario.turns) == 1


def test_the_shipped_seed_protocol_is_refused_by_this_engine():
    """The one committed template, checked against the real bytes.

    Research-tree-only: `prompts/panels/**` deliberately does not cross the
    release allowlist (workspace material, WP8), so in the release tree this
    file is absent and the check skips — the refusal RULE itself is pinned
    engine-pure by the synthetic-template tests above."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(repo, "prompts", "panels", "templates",
                        "deliberative-appellate-panel-v1.json")
    if not os.path.isfile(path):
        pytest.skip("panel templates do not ship — research-tree-only check")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["kind"] == multi_agent.PROTOCOL_TEMPLATE_KIND
    assert payload["sharedMaterials"] == ""
    assert len(payload["turns"]) == 15
    with pytest.raises(multi_agent.ScenarioError):
        multi_agent.Scenario.from_dict(payload)
