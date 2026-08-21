"""Panel sharding, end to end through the merge contract.

The previous tests proved `plan_shard` partitions a list and stopped there —
which is exactly why a `TypeError` in the real shard path and a key-shape
mismatch with the merge both shipped. These exercise the contract the merge
actually enforces.
"""

import pytest

from steerlab_server.experiment import resume as resume_mod
from steerlab_server.experiment import sharding


CONDITIONS = ["configured", "baseline"]
TURNS = ["t1", "t2", "t3"]


def _plans(count, replicates=4):
    return [sharding.plan_panel_shard(
                sharding.ShardSpec(index=i, count=count),
                condition_names=CONDITIONS, replicates=replicates,
                turn_ids=TURNS)
            for i in range(count)]


def test_every_record_lands_on_exactly_one_shard():
    seen = [key for plan in _plans(4) for key in plan.keys]
    assert len(seen) == len(set(seen)) == len(CONDITIONS) * 4 * len(TURNS)


def test_a_transcript_is_never_split_across_shards():
    """Turns are ordered — turn k is conditioned on 1..k-1 — so a transcript
    that straddles two workers cannot be produced at all."""
    for plan in _plans(4):
        transcripts = {(k[0], k[3]) for k in plan.keys}
        for transcript in transcripts:
            turns = [k for k in plan.keys if (k[0], k[3]) == transcript]
            assert len(turns) == len(TURNS), f"{transcript} was split"


def test_keys_are_the_shared_five_part_record_identity():
    """The merge proves completeness with resume.record_key. A two-part
    (condition, replicate) key would never reconcile against it."""
    plan = _plans(2)[0]
    for key in plan.keys:
        assert len(key) == 5
        condition, prompt_index, prompt_id, sample_index, kind = key
        assert condition in CONDITIONS
        assert prompt_id in TURNS
        assert kind == resume_mod.KIND_SAMPLED
        assert isinstance(sample_index, int)


def test_a_panel_record_reconciles_with_its_planned_key():
    """The identity the runner STAMPS must equal the identity the merge
    EXPECTS. Panel records carry replicateIndex; sampleIndex is what
    record_key reads, so the runner writes both."""
    plan = _plans(2)[0]
    condition, prompt_index, prompt_id, sample_index, _ = plan.keys[0]
    record = {"condition": condition, "promptIndex": prompt_index,
              "promptID": prompt_id, "replicateIndex": sample_index,
              "sampleIndex": sample_index, "output": "x"}

    assert resume_mod.record_key(record) in plan.allowed_keys


def test_replicates_of_one_turn_do_not_collide():
    """Without the replicate in the identity, every play-through of a turn
    collapses to one key and the merge refuses the partials as duplicates."""
    keys = {resume_mod.record_key(
                {"condition": "configured", "promptIndex": 0, "promptID": "t1",
                 "sampleIndex": r})
            for r in range(4)}
    assert len(keys) == 4


def test_each_condition_is_owned_by_exactly_one_shard():
    """Condition-level work must run once across the fleet, not K times."""
    owners = [c for plan in _plans(4) for c in plan.owned_conditions]
    assert sorted(owners) == sorted(CONDITIONS)


def test_the_stamp_requires_the_experiment_hash(tmp_path):
    """The shipped call omitted it and raised TypeError on every sharded panel
    run — the failure this file exists to prevent recurring."""
    plan = _plans(2)[0]
    sharding.write_shard_stamp(str(tmp_path), plan, "deadbeef")
    stamp = sharding.read_shard_stamp(str(tmp_path))

    assert stamp["shardIndex"] == 0 and stamp["shardCount"] == 2
    assert len(stamp["expectedKeys"]) == len(plan.keys)

    with pytest.raises(TypeError):
        sharding.write_shard_stamp(str(tmp_path), plan)


# --- the REAL integrated path ------------------------------------------------
#
# The tests above check planning, stamps and key reconciliation. That is what
# let a TypeError in the actual runner ship: components proved, feature not.
# These drive _run_multi_agent_study and merge_shard_runs for real.

import json
import os
from types import SimpleNamespace

from steerlab_server.experiment import multi_agent, tasks
from steerlab_server.experiment.manifest import Manifest


def _panel_workspace(tmp_path, monkeypatch, *, replicates=2):
    root = tmp_path / "ws"
    for sub in ("prompts/panels", "experiments", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    panel = {
        "schemaVersion": 1, "name": "panel", "baseModelID": "m",
        "description": "", "sharedMaterials": "rules",
        "temperature": 0.0, "maxTokens": 32,
        "agents": [{"id": "a", "name": "A", "baseModelID": "m",
                    "systemPrompt": "", "variantArtifactPath": None,
                    "variantArtifactHash": None}],
        "turns": [{"id": f"t{i}", "title": f"Turn {i}", "speakerAgentID": "a",
                   "promptTemplate": "go", "outputLabel": f"o{i}",
                   "routing": "all", "routedAgentIDs": [],
                   "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": None}
                  for i in range(2)],
    }
    (root / "prompts/panels/panel.json").write_text(json.dumps(panel))
    spec = {
        "name": "panel", "modelID": "m", "studyKind": "multiAgent",
        "multiAgentScenarioPath": "prompts/panels/panel.json",
        "multiAgentIncludeBaseline": True, "samplesPerItem": replicates,
        "temperature": 0.0, "seeds": [0]}
    # merge_shard_runs rebuilds report.json through the shared report builder,
    # which loads the manifest from disk — so the fixture has to be a real
    # workspace, not just a scenario file.
    (root / "experiments").mkdir(exist_ok=True)
    (root / "experiments/panel.json").write_text(json.dumps(spec))
    manifest = Manifest.from_dict(spec)
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    monkeypatch.setattr(tasks, "_advise_cross_substrate", lambda *a, **k: None)
    monkeypatch.setattr(tasks, "_write_config_snapshot", lambda *a, **k: None)
    return root, manifest


def test_per_turn_lines_reach_the_TASKS_logger(tmp_path, monkeypatch):
    """Feature-level twin of test_each_turn_reports_accelerator_memory.

    That test proved run_scenario emits per-turn lines — by passing its OWN
    log callback. The production call site in tasks passed none, so
    run_scenario fell back to its no-op default and a 144-turn field run
    logged ZERO turn lines while the component test stayed green. The probe
    those lines carry exists precisely so a memory investigation doesn't
    have to guess; this asserts the wiring at the level the field exercises:
    the logger handed to the TASK sees every turn."""
    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=1)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    lines = []
    tasks._run_multi_agent_study("panel", manifest, model, str(root),
                                 log=lines.append)

    turn_lines = [l for l in lines if l.startswith("turn ")]
    # 2 turns x 2 conditions (configured + baseline), every one visible.
    assert len(turn_lines) == 4, (turn_lines, lines)


def test_two_shards_run_for_real_and_merge_reconciles(tmp_path, monkeypatch):
    """The regression test the shipped TypeError deserved: drive the actual
    runner twice, then prove the merge's own completeness check passes."""
    from steerlab_server.experiment import sharding

    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=2)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    dirs = []
    for index in range(2):
        dirs.append(tasks._run_multi_agent_study(
            "panel", manifest, model, str(root),
            shard=sharding.ShardSpec(index=index, count=2)))

    # Every planned record produced exactly once across the fleet.
    keys, expected = [], 0
    for directory in dirs:
        stamp = sharding.read_shard_stamp(directory)
        assert stamp is not None, "no shard stamp written"
        expected += len(stamp["expectedKeys"])
        with open(os.path.join(directory, "generations.jsonl"),
                  encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    keys.append(resume_mod.record_key(json.loads(line)))

    assert len(keys) == expected == 2 * 2 * 2, (len(keys), expected)
    assert len(set(keys)) == len(keys), "a record was produced twice"

    # THE point of this test: run the merge's OWN completeness proof, not a
    # hand-rolled imitation of it. The previous version asserted against the
    # stamps itself and still claimed "the merge's check passes" — the same
    # component-not-feature gap that let the original TypeError ship.
    merged = sharding.merge_shard_runs("panel", dirs, root=str(root))

    with open(os.path.join(merged, "generations.jsonl"), encoding="utf-8") as handle:
        merged_keys = [resume_mod.record_key(json.loads(l))
                       for l in handle if l.strip()]
    assert sorted(merged_keys) == sorted(keys)
    assert os.path.exists(os.path.join(merged, "report.json"))

    # The voice-lint roll-up is per-shard by construction (each shard sees
    # only its own transcripts), so the merge must REBUILD it over the whole
    # matrix rather than compare partials.
    with open(os.path.join(merged, "panel-voice-lint.csv"),
              encoding="utf-8") as handle:
        rows = [line.split(",") for line in handle.read().strip().split("\n")]
    assert rows[0][0] == "condition"
    # Every merged turn is accounted for, and no shard's count survives alone.
    assert sum(int(row[2]) for row in rows[1:]) == len(merged_keys)


def test_a_checkpoint_signal_parks_the_run_and_resumes_in_place(tmp_path, monkeypatch):
    """Walltime checkpointing, end to end: the flag must be OBSERVED, the
    directory parked with resume-state.json, and a re-run continue in the same
    directory rather than starting fresh."""
    from steerlab_server.experiment import resume as resume_module

    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=2)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    flag = resume_module.CheckpointFlag()
    flag.request()  # as a trapped SIGUSR1/SIGTERM would

    with pytest.raises(resume_module.CheckpointRequested) as caught:
        tasks._run_multi_agent_study("panel", manifest, model, str(root),
                                     checkpoint=flag)
    parked = caught.value.run_directory

    # Everything the requeue needs is already on disk when it propagates.
    assert os.path.exists(os.path.join(parked, "resume-state.json"))
    assert os.path.exists(os.path.join(parked, "generations.jsonl"))

    # The scheduler re-runs with the SAME directory; it must be accepted.
    finished = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), run_directory=parked)
    assert finished == parked
    assert os.path.exists(os.path.join(parked, "report.json"))


def test_a_signal_during_the_only_transcript_is_observed(tmp_path, monkeypatch):
    """The gap: the flag was polled only BEFORE starting a transcript, so a
    signal arriving mid-transcript was ignored — and during the final or only
    transcript, ignored forever. The run then wrote report.json, no
    resume-state.json, and returned success; a hard kill after that abandons
    every durably written turn, because resolve_pointer starts fresh."""
    from steerlab_server.experiment import resume as resume_module

    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=1)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")
    flag = resume_module.CheckpointFlag()

    # Signal DURING generation, once the first turn is already written.
    calls = {"n": 0}

    def signal_on_first_turn(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            flag.request()
        return "out"

    monkeypatch.setattr(multi_agent, "generate", signal_on_first_turn)

    with pytest.raises(resume_module.CheckpointRequested) as caught:
        tasks._run_multi_agent_study("panel", manifest, model, str(root),
                                     checkpoint=flag)
    parked = caught.value.run_directory

    assert os.path.exists(os.path.join(parked, "resume-state.json")), \
        "a mid-transcript signal must still park the directory"
    assert not os.path.exists(os.path.join(parked, "report.json")), \
        "an interrupted run must not claim completion"
    # The turn that was already fsynced is folded into the parked root view.
    with open(os.path.join(parked, "generations.jsonl"), encoding="utf-8") as h:
        assert sum(1 for line in h if line.strip()) >= 1


def test_an_unsharded_partial_cannot_be_resumed_under_shard(tmp_path, monkeypatch):
    """The guard covered stamped->unsharded and wrong-shard->shard, but not
    no-stamp->shard: an unsharded partial was accepted, restamped as a shard,
    and 'completed' holding only that shard's subset."""
    from steerlab_server.experiment import resume as resume_module
    from steerlab_server.experiment import sharding

    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=2)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")
    flag = resume_module.CheckpointFlag()
    flag.request()

    with pytest.raises(resume_module.CheckpointRequested) as caught:
        tasks._run_multi_agent_study("panel", manifest, model, str(root),
                                     checkpoint=flag)
    unsharded = caught.value.run_directory
    assert sharding.read_shard_stamp(unsharded) is None

    with pytest.raises(resume_module.ResumeError, match="not a shard partial"):
        tasks._run_multi_agent_study(
            "panel", manifest, model, str(root), run_directory=unsharded,
            shard=sharding.ShardSpec(index=0, count=2))


def test_a_completed_resume_clears_its_resume_pointer(tmp_path, monkeypatch):
    """A finished run is not resumable. Leaving the pointer behind violates the
    artifact contract and invites a later caller to 'resume' a complete run."""
    from steerlab_server.experiment import resume as resume_module

    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=2)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")
    flag = resume_module.CheckpointFlag()
    flag.request()

    with pytest.raises(resume_module.CheckpointRequested) as caught:
        tasks._run_multi_agent_study("panel", manifest, model, str(root),
                                     checkpoint=flag)
    parked = caught.value.run_directory
    assert os.path.exists(os.path.join(parked, "resume-state.json"))

    finished = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), run_directory=parked)

    assert os.path.exists(os.path.join(finished, "report.json"))
    assert not os.path.exists(os.path.join(finished, "resume-state.json")), \
        "a completed run must not leave a resume pointer"


# --- submit-time fan-out resolution -----------------------------------------

def _panel_manifest(replicates, baseline=True):
    return Manifest.from_dict({
        "name": "panel", "modelID": "m", "studyKind": "multiAgent",
        "multiAgentScenarioPath": "prompts/panels/panel.json",
        "multiAgentIncludeBaseline": baseline, "samplesPerItem": replicates,
        "temperature": 0.7})


def test_a_single_transcript_panel_does_not_fan_out():
    """Reported from the cluster: submitting a 1-play-through panel with 2 GPUs
    fanned out two shard jobs that each refused, and the parent reported only
    'shard job ended without success'. Resolve it at submit, where the
    researcher is watching — and degrade rather than refuse, because they asked
    for a run."""
    from steerlab_server.api import submissions

    parallel, note, _reason = submissions._resolve_parallel_jobs(
        2, verb="run", executor="slurm", pipeline_stages=None,
        manifest=_panel_manifest(replicates=1, baseline=False))

    assert parallel == 1
    assert "one transcript" in note
    assert "Samples per item" in note, "the note must name the control to change"


def test_fan_out_is_capped_at_the_transcript_count():
    """A panel shards over whole transcripts, so more GPUs than transcripts
    would leave jobs with no work."""
    from steerlab_server.api import submissions

    # 2 conditions x 2 replicates = 4 transcripts, 8 GPUs requested.
    parallel, note, _reason = submissions._resolve_parallel_jobs(
        8, verb="run", executor="slurm", pipeline_stages=None,
        manifest=_panel_manifest(replicates=2))

    assert parallel == 4
    assert "reduced to 4" in note


def test_a_replicated_panel_still_shards():
    from steerlab_server.api import submissions

    parallel, note, _reason = submissions._resolve_parallel_jobs(
        2, verb="run", executor="slurm", pipeline_stages=None,
        manifest=_panel_manifest(replicates=4))

    assert parallel == 2 and note is None


def test_ordinary_studies_are_unaffected():
    """The panel rule must not touch model-output sharding."""
    from steerlab_server.api import submissions

    ordinary = Manifest.from_dict({"name": "s", "modelID": "m"})
    parallel, note, _reason = submissions._resolve_parallel_jobs(
        4, verb="run", executor="slurm", pipeline_stages=None, manifest=ordinary)

    assert parallel == 4 and note is None
