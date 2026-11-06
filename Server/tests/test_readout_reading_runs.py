"""A study run's answer-token readouts follow its prompt reading (issue #8):
what the run passes them, what it records, how a resume continues, what a
shard merge refuses, what the memory check advises, and what the engine
advertises.

The readout itself is replaced by a stand-in that records the reading it is
handed (the readout's own arithmetic is ``test_readout_prompt_reading.py``).
Expected values are written by hand from the contract, not computed by the
code under test.
"""
import json
import os

import pytest

from steerlab_server.experiment import logprob
from steerlab_server.experiment import prompt_reading as PR
from steerlab_server.experiment import resume as resume_mod
from steerlab_server.experiment import sharding, tasks
from test_prompt_reading import (CUDA_DEFAULT, _config, _cuda_model, _patch,
                                 _study, _write)
from test_prompt_reading import _no_ambient_overrides  # noqa: F401 - autouse

KEY = "answerTokenReadoutsFollowRule"


def _choice_study(root, name, ids=("long-packet", "short-note"), **fields):
    """A study whose only outcome, unless ``fields`` say otherwise, is the
    answer-token readout (the long item renders to 217,230 tokens, the short
    one to 900)."""
    fields.setdefault("outcomeInstruments", ["answerTokenLogprob"])
    prompts = _study(root, name, ids=ids, **fields)
    _write(prompts, "".join(
        json.dumps({"id": prompt_id, "prompt": f"Read the {prompt_id}.",
                    "options": ["A", "B"], "target": "A",
                    "responseFormat": "label"}) + "\n" for prompt_id in ids))
    return prompts


def _readouts(monkeypatch, calls, *, flag=None, after=None):
    """Replace the readout with a stand-in that records what it is handed:
    the reading when one is passed, else the string "none"."""
    _patch(monkeypatch)

    def score_options(model, prompt, options, **kwargs):
        reading = kwargs.get("reading", "none")
        calls.append(reading if reading == "none" else reading.rule())
        if flag is not None and len(calls) == after:
            flag.request()
        return logprob.ChoiceResult(
            options=[logprob.OptionScore(option=o, token_ids=[i + 1],
                                         token_logprobs=[-1.0 - i])
                     for i, o in enumerate(options)],
            prompt_token_count=3)
    monkeypatch.setattr(logprob, "score_options", score_options)


# --- a new run ---------------------------------------------------------------------

def test_a_new_run_reads_out_by_its_reading_and_records_so(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _choice_study(root, "readout-new")
    calls = []
    _readouts(monkeypatch, calls)
    run_dir = tasks.run("readout-new", prompts, root, model_provider=_cuda_model,
                        log=lambda *_: None)
    # Two arms (baseline and the agent), two items: every readout is handed
    # the run's reading, the CUDA default.
    assert calls == [CUDA_DEFAULT] * 4
    entry = _config(run_dir)["notes"]["promptReading"]
    assert entry[KEY] is True
    assert (entry["mode"], entry["chunkTokens"], entry["aboveTokens"]) == (
        "chunked", 4096, 8192)
    assert PR.readouts_follow_rule(entry)


def test_a_study_that_also_generates_hands_every_readout_the_reading(
        tmp_path, monkeypatch):
    """Readouts and generations interleave item by item; every readout,
    not only the first, is handed the run's reading."""
    root = str(tmp_path)
    prompts = _choice_study(root, "readout-both",
                            outcomeInstruments=["answerTokenLogprob",
                                                "sampledText"])
    calls = []
    _readouts(monkeypatch, calls)
    tasks.run("readout-both", prompts, root, model_provider=_cuda_model,
              log=lambda *_: None)
    assert calls == [CUDA_DEFAULT] * 4


def test_a_single_pass_reading_hands_the_readout_nothing(tmp_path, monkeypatch):
    """A rule that never chunks reads exactly as no rule does, so the call is
    the one it always was; the record still says readouts follow it."""
    root = str(tmp_path)
    prompts = _choice_study(root, "readout-single",
                            promptReading={"mode": "singlePass"})
    calls = []
    _readouts(monkeypatch, calls)
    run_dir = tasks.run("readout-single", prompts, root,
                        model_provider=_cuda_model, log=lambda *_: None)
    assert calls == ["none"] * 4
    assert _config(run_dir)["notes"]["promptReading"][KEY] is True


def test_record_and_readout_reading_are_what_the_contract_says():
    cuda = PR.PromptReading.from_rule(CUDA_DEFAULT, source="engineDefault")
    marked = PR.record(cuda, device="cuda", counts=[[10]], environ={},
                       readouts_follow_rule=True)
    unmarked = PR.record(cuda, device="cuda", counts=[[10]], environ={})
    assert marked[KEY] is True and KEY not in unmarked
    assert PR.readouts_follow_rule(marked)
    for entry in (unmarked, None, {**marked, KEY: "true"}, {**marked, KEY: 1}):
        assert not PR.readouts_follow_rule(entry)
    # A new run: its own reading.
    assert PR.readout_reading(cuda) is cuda
    assert PR.chunks_any_prompt(cuda)
    assert not PR.chunks_any_prompt(PR.PromptReading.single_pass())
    assert not PR.chunks_any_prompt(None)


# --- a resumed run -----------------------------------------------------------------

def _park(root, name, monkeypatch, *, after=1):
    prompts = _choice_study(root, name)
    flag = resume_mod.CheckpointFlag()
    calls, seen = [], {}
    _readouts(monkeypatch, calls, flag=flag, after=after)
    with pytest.raises(resume_mod.CheckpointRequested):
        tasks.run(name, prompts, root, model_provider=_cuda_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    return prompts, seen["dir"], calls


def _unmark(run_dir):
    """Rewrite a parked run's record as 0.9.20 wrote it: without the key."""
    path = os.path.join(run_dir, "config.json")
    config = _config(run_dir)
    del config["notes"]["promptReading"][KEY]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def test_a_resumed_run_without_the_key_keeps_single_pass_readouts(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts, run_dir, first = _park(root, "readout-resume-old", monkeypatch)
    assert first == [CUDA_DEFAULT]
    _unmark(run_dir)
    config_path = os.path.join(run_dir, "config.json")
    before = open(config_path, "rb").read()
    calls, lines = [], []
    _readouts(monkeypatch, calls)
    assert tasks.run("readout-resume-old", prompts, root,
                     model_provider=_cuda_model, log=lines.append,
                     run_directory=run_dir) == run_dir
    # The three readouts left read their prompts in a single pass, as the
    # engine that started the run read them; the log says so once.
    assert calls == ["none"] * 3
    assert [line for line in lines
            if "its readouts continue in a single pass at every length" in line]
    # config.json is never rewritten on resume.
    assert open(config_path, "rb").read() == before


def test_a_resumed_run_with_the_key_keeps_following_its_reading(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts, run_dir, _ = _park(root, "readout-resume-new", monkeypatch)
    config_path = os.path.join(run_dir, "config.json")
    before = open(config_path, "rb").read()
    calls, lines = [], []
    _readouts(monkeypatch, calls)
    tasks.run("readout-resume-new", prompts, root, model_provider=_cuda_model,
              log=lines.append, run_directory=run_dir)
    assert calls == [CUDA_DEFAULT] * 3
    assert not [line for line in lines if "continue in a single pass" in line]
    assert open(config_path, "rb").read() == before


# --- the capability ------------------------------------------------------------------

def test_the_engine_advertises_that_readouts_follow_the_reading():
    from steerlab_server.api.profile import capability_snapshot
    described = capability_snapshot()["longPromptReading"]
    assert described[KEY] is True
    assert ("A study run's answer-token readouts read their prompts the same "
            "way") in described["summary"]


# --- the shard merge -----------------------------------------------------------------

def _readout_shards(root, name, monkeypatch, *, ids):
    prompts = _choice_study(root, name, ids=ids)
    dirs = []
    for index in range(2):
        _readouts(monkeypatch, [])
        dirs.append(tasks.run(name, prompts, root, model_provider=_cuda_model,
                              log=lambda *_: None,
                              shard=sharding.ShardSpec(index, 2)))
    return dirs


def test_a_merge_of_shards_whose_readouts_read_a_long_prompt_differently_refuses(
        tmp_path, monkeypatch):
    """Both shards' generation reads alike (the CUDA default), but shard 1's
    engine predates readouts following the reading, so it read the
    217,230-token prompt's readout in one pass."""
    root = str(tmp_path)
    dirs = _readout_shards(root, "readout-merge", monkeypatch,
                           ids=("long-packet", "short-note"))
    _unmark(dirs[1])
    with pytest.raises(sharding.ShardMergeRefusal) as refusal:
        sharding.merge_shard_runs("readout-merge", dirs, root=root)
    message = str(refusal.value)
    assert ("shard 0 read its answer-token readouts as prompts over 8,192 tokens "
            "in 4,096-token chunks (shorter prompts in a single pass), but "
            "shard 1 read its answer-token readouts in a single pass at every "
            "length") in message
    assert "so its longest prompt was read out differently" in message
    assert "the shard partials are untouched" in message
    assert refusal.value.gate == "manifestEpoch"
    assert refusal.value.repair_action.startswith(
        "steerlab-server study submit readout-merge --verb run --executor "
        "slurm --parallel 2")
    assert not os.path.exists(os.path.join(root, "runs", "exp-readout-merge-run"))


def test_a_merge_of_shards_whose_readouts_read_alike_succeeds(tmp_path, monkeypatch):
    """Every prompt at or below 8,192 tokens: a marked shard and an unmarked
    one read every readout in one pass, so they merge."""
    root = str(tmp_path)
    dirs = _readout_shards(root, "readout-merge-short", monkeypatch,
                           ids=("short-note",))
    _unmark(dirs[1])
    merged = sharding.merge_shard_runs("readout-merge-short", dirs, root=root)
    assert _config(merged)["notes"]["promptReading"][KEY] is True


def test_a_merge_of_matching_shards_succeeds_with_the_key(tmp_path, monkeypatch):
    root = str(tmp_path)
    dirs = _readout_shards(root, "readout-merge-same", monkeypatch,
                           ids=("long-packet", "short-note"))
    merged = sharding.merge_shard_runs("readout-merge-same", dirs, root=root)
    assert _config(merged)["notes"]["promptReading"][KEY] is True


def test_a_study_without_readouts_is_not_compared_on_them(tmp_path, monkeypatch):
    """A sampled-text study has no readouts: the key's absence on one shard
    changes nothing about how its prompts were read."""
    root = str(tmp_path)
    prompts = _study(root, "readout-merge-sampled", ids=("long-packet",))
    dirs = []
    for index in range(2):
        _patch(monkeypatch)
        dirs.append(tasks.run("readout-merge-sampled", prompts, root,
                              model_provider=_cuda_model, log=lambda *_: None,
                              shard=sharding.ShardSpec(index, 2)))
    _unmark(dirs[1])
    sharding.merge_shard_runs("readout-merge-sampled", dirs, root=root)
