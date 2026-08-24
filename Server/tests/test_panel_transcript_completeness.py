"""The panel transcript layer must be COMPLETE, or the run must say so.

The 2026-08-20 ledger entry this file exists for: a multi-agent run produced a
complete ``generations.jsonl`` (every record of both conditions, parseable) and
a complete ``baseline/replicate-{0..4}/`` tree — and an EMPTY ``configured/``.
No error in the job log. Two sibling runs submitted the same minute wrote both
conditions' trees fine.

Two separate things are pinned here.

1. The mechanism found in the shard merge: ``_copy_invariant_artifacts`` walks
   only FILES, so every sharded panel merge assembled a run directory with a
   complete ``generations.jsonl`` and no transcript layer at all — silently.
2. The backstop that holds whatever else can produce that shape: at run end
   the trees on disk must match the transcripts the run was responsible for,
   and a mismatch is a LOUD advisory naming the exact condition/replicates —
   never an exit code, never a failed run, because ``generations.jsonl`` is
   the authoritative record and ``transcript.md`` is the human layer.

Fixtures are deliberately neutral (``panel-a``, ``agent-a``, ``turn-a``): the
only study vocabulary here is the engine's own condition literals.
"""

import json
import os
import shutil
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import multi_agent, sharding, tasks
from steerlab_server.experiment.manifest import Manifest


ADVISORY_MARKER = "panel transcripts incomplete"


def _workspace(tmp_path, monkeypatch, *, replicates=2):
    root = tmp_path / "ws"
    for sub in ("prompts/panels", "experiments", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    panel = {
        "schemaVersion": 1, "name": "panel-a", "baseModelID": "model-a",
        "description": "", "sharedMaterials": "materials",
        "temperature": 0.0, "maxTokens": 16,
        "agents": [{"id": "agent-a", "name": "Agent A",
                    "baseModelID": "model-a", "systemPrompt": "",
                    "variantArtifactPath": None,
                    "variantArtifactHash": None}],
        "turns": [{"id": f"turn-{letter}", "title": f"Turn {letter.upper()}",
                   "speakerAgentID": "agent-a", "promptTemplate": "go",
                   "outputLabel": f"out-{letter}", "routing": "all",
                   "routedAgentIDs": [], "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": None}
                  for letter in ("a", "b")],
    }
    (root / "prompts/panels/panel-a.json").write_text(json.dumps(panel))
    spec = {
        "name": "panel-a", "modelID": "model-a", "studyKind": "multiAgent",
        "multiAgentScenarioPath": "prompts/panels/panel-a.json",
        "multiAgentIncludeBaseline": True, "samplesPerItem": replicates,
        "temperature": 0.0, "seeds": [0]}
    (root / "experiments/panel-a.json").write_text(json.dumps(spec))
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "text")
    monkeypatch.setattr(tasks, "_advise_cross_substrate", lambda *a, **k: None)
    monkeypatch.setattr(tasks, "_advise_dependency_lock_drift",
                        lambda *a, **k: None)
    monkeypatch.setattr(tasks, "_write_config_snapshot", lambda *a, **k: None)
    return root, Manifest.from_dict(spec)


def _model():
    return SimpleNamespace(model_id="model-a", revision="rev-a", device="cpu")


def _advisories(run_directory):
    path = os.path.join(run_directory, "advisories.txt")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _conditions_in(run_directory):
    with open(os.path.join(run_directory, "generations.jsonl"),
              encoding="utf-8") as handle:
        return [json.loads(line)["condition"]
                for line in handle if line.strip()]


# --- the backstop: run-end completeness --------------------------------------

def test_a_condition_whose_tree_vanishes_is_named_in_a_loud_advisory(
        tmp_path, monkeypatch):
    """The ledger's exact shape, reproduced: complete generations.jsonl, one
    condition's replicate directories gone, the condition directory itself
    left behind and empty. Before the check, that run reported success with
    nothing anywhere saying the transcript layer was half missing."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=2)
    real_flatten = tasks._panel_records_from

    def flatten_then_erase(sub, name, manifest_, model, condition, replicate):
        records = real_flatten(sub, name, manifest_, model, condition,
                               replicate)
        if condition == "configured":
            shutil.rmtree(sub)  # the silent skip, whatever produced it
        return records

    monkeypatch.setattr(tasks, "_panel_records_from", flatten_then_erase)

    lines = []
    run_directory = tasks._run_multi_agent_study(
        "panel-a", manifest, _model(), str(root), log=lines.append)

    # The run FINISHED — the advisory changes nothing about the run's fate.
    assert os.path.isfile(os.path.join(run_directory, "report.json"))
    # generations.jsonl is complete and authoritative: both arms, all turns.
    conditions = _conditions_in(run_directory)
    assert conditions.count("configured") == conditions.count("baseline") == 4

    # ...and the transcript layer's absence is said out loud, per replicate.
    text = _advisories(run_directory)
    assert ADVISORY_MARKER in text
    assert "configured/replicate-0" in text and "configured/replicate-1" in text
    assert "baseline/replicate-0" not in text
    assert any(ADVISORY_MARKER in line and line.startswith("ADVISORY:")
               for line in lines), lines


def test_a_complete_panel_run_draws_no_transcript_advisory(tmp_path,
                                                           monkeypatch):
    """The other half of the contract: an advisory nobody can trust to be
    absent is an advisory nobody reads."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=2)
    lines = []
    run_directory = tasks._run_multi_agent_study(
        "panel-a", manifest, _model(), str(root), log=lines.append)

    for condition in ("configured", "baseline"):
        for replicate in range(2):
            sub = os.path.join(run_directory, condition,
                               f"replicate-{replicate}")
            assert os.path.isfile(os.path.join(sub, "turns.jsonl"))
            assert os.path.isfile(os.path.join(sub, "transcript.md"))
    assert ADVISORY_MARKER not in _advisories(run_directory)
    assert not [line for line in lines if ADVISORY_MARKER in line]


def test_a_single_replicate_run_is_checked_in_its_own_layout(tmp_path,
                                                             monkeypatch):
    """samplesPerItem 1 keeps the historical <run>/<condition>/ layout. A
    check that assumed replicate-N would report every such run as missing
    everything — the classic way a completeness gate gets switched off."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=1)
    run_directory = tasks._run_multi_agent_study(
        "panel-a", manifest, _model(), str(root))

    assert os.path.isfile(
        os.path.join(run_directory, "configured", "transcript.md"))
    assert ADVISORY_MARKER not in _advisories(run_directory)


def test_a_shard_is_not_faulted_for_transcripts_another_shard_owns(
        tmp_path, monkeypatch):
    """A shard partial legitimately holds one slice of the matrix. The check
    is over the transcripts THIS run admitted, not over the study's whole
    condition list, or every sharded panel run would cry wolf."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=2)
    run_directory = tasks._run_multi_agent_study(
        "panel-a", manifest, _model(), str(root),
        shard=sharding.ShardSpec(index=0, count=2))

    assert os.path.isdir(os.path.join(run_directory, "configured"))
    assert not os.path.isdir(os.path.join(run_directory, "baseline"))
    assert ADVISORY_MARKER not in _advisories(run_directory)


def test_a_cancelled_run_is_not_faulted_for_the_arm_it_never_reached(
        tmp_path, monkeypatch):
    """A parked run has trees it has not written YET. Advising over a
    deliberately partial matrix would make the advisory meaningless."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=2)
    seen = {"n": 0}

    def cancel_after_one_transcript():
        seen["n"] += 1
        return seen["n"] > 1

    run_directory = tasks._run_multi_agent_study(
        "panel-a", manifest, _model(), str(root),
        should_cancel=cancel_after_one_transcript)

    assert os.path.isfile(os.path.join(run_directory, "resume-state.json"))
    assert ADVISORY_MARKER not in _advisories(run_directory)


# --- exception hygiene -------------------------------------------------------

def test_a_transcript_write_failure_is_recorded_with_its_exception_text(
        tmp_path, monkeypatch):
    """turns.jsonl is the authoritative per-transcript record and is already
    fsynced when the summary layer is written. A failure THERE must neither
    sink a finished run nor disappear: it is recorded per condition, with the
    exception text, in the run's advisories."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=1)

    def refuse_to_render(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(multi_agent, "_transcript", refuse_to_render)

    lines = []
    run_directory = tasks._run_multi_agent_study(
        "panel-a", manifest, _model(), str(root), log=lines.append)

    # The run still completed and still has its authoritative record.
    assert os.path.isfile(os.path.join(run_directory, "report.json"))
    assert len(_conditions_in(run_directory)) == 4
    assert os.path.isfile(
        os.path.join(run_directory, "configured", "turns.jsonl"))

    text = _advisories(run_directory)
    assert ADVISORY_MARKER in text
    assert "no space left on device" in text
    assert "configured" in text and "baseline" in text


def test_an_unreadable_turns_file_says_so_before_regenerating(tmp_path,
                                                              monkeypatch):
    """``_completed_turns`` recovers from an unreadable resume file by
    starting the transcript over — right, but not something to do quietly:
    without a line, a regenerated transcript is indistinguishable from a
    first attempt."""
    turns = tmp_path / "turns.jsonl"
    turns.write_text('{"turnID": "turn-a"}\n')

    def refuse_to_open(*args, **kwargs):
        raise OSError("input/output error")

    monkeypatch.setattr("builtins.open", refuse_to_open)
    lines = []
    assert multi_agent._completed_turns(str(turns), log=lines.append) == {}
    monkeypatch.undo()

    assert any("input/output error" in line for line in lines), lines


# --- the mechanism: the shard merge dropped the whole transcript layer -------

def test_a_sharded_panel_merge_carries_every_shard_s_transcript_tree(
        tmp_path, monkeypatch):
    """The proven mechanism. ``_copy_invariant_artifacts`` copies FILES from
    shard 0; a panel's transcripts are DIRECTORIES, and shards own disjoint
    ones. So the merged run used to be assembled with a complete, correct
    generations.jsonl and no transcript layer whatsoever — no refusal, no log
    line, nothing to notice."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=2)
    model = _model()

    partials = [tasks._run_multi_agent_study(
        "panel-a", manifest, model, str(root),
        shard=sharding.ShardSpec(index=index, count=2)) for index in range(2)]
    # The premise: the two shards between them hold the whole matrix, one
    # condition each.
    assert sorted(entry for partial in partials
                  for entry in os.listdir(partial)
                  if os.path.isdir(os.path.join(partial, entry))) == \
        ["baseline", "configured"]

    merged = sharding.merge_shard_runs("panel-a", partials, root=str(root))

    for condition in ("configured", "baseline"):
        for replicate in range(2):
            sub = os.path.join(merged, condition, f"replicate-{replicate}")
            assert os.path.isfile(os.path.join(sub, "turns.jsonl")), sub
            assert os.path.isfile(os.path.join(sub, "transcript.md")), sub
            assert os.path.isfile(os.path.join(sub, "report.json")), sub
    # And the merged transcripts are the shards' bytes, not a re-render.
    with open(os.path.join(merged, "configured", "replicate-0",
                           "turns.jsonl"), "rb") as handle:
        merged_bytes = handle.read()
    with open(os.path.join(partials[0], "configured", "replicate-0",
                           "turns.jsonl"), "rb") as handle:
        assert merged_bytes == handle.read()


def test_two_shards_claiming_one_transcript_refuse_rather_than_pick(
        tmp_path, monkeypatch):
    """Exactly one shard owns a transcript. Two partials disagreeing about
    one is a broken fan-out, and a merge that silently kept one of them would
    be inventing a matrix nobody ran."""
    root, manifest = _workspace(tmp_path, monkeypatch, replicates=2)
    model = _model()
    partials = [tasks._run_multi_agent_study(
        "panel-a", manifest, model, str(root),
        shard=sharding.ShardSpec(index=index, count=2)) for index in range(2)]

    # Shard 1 also claims a transcript shard 0 owns, with different bytes.
    intruder = os.path.join(partials[1], "configured", "replicate-0")
    os.makedirs(intruder, exist_ok=True)
    with open(os.path.join(intruder, "turns.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"turnID": "turn-a", "output": "different"}\n')

    with pytest.raises(sharding.ShardMergeError) as caught:
        sharding.merge_shard_runs("panel-a", partials, root=str(root))
    assert "configured/replicate-0/turns.jsonl" in str(caught.value)
