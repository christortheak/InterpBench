"""`--shard` on the CLI's own run arm, and strict argv on the experiment
family (open-issues §16, 2026-08-19).

The field incident: a checkpointed shard partial of a `--parallel` fan-out had
NO CLI path back to life. `bundle execute` accepted `--shard k/K`; `experiment
run` did not, and the resume gate's own repair sentence — "resume it with the
same --shard k/K" — named a flag the verb refused. Worse, that gate fired from
inside `_run_impl`, i.e. AFTER the job had staged and loaded 51 GiB of weights
onto the device: jobs 47547790/47547791 and 47553041/47553042 (2026-08-18) burned
four GPU allocations to reach a refusal that reads three files.

What is pinned here:

1. `experiment run --shard k/K` parses and threads a `ShardSpec` to `tasks.run`,
   including in combination with `--resume` (the executable repair command).
2. A malformed `--shard` value is a malformed invocation (64), not a traceback.
3. `experiment pipeline --shard` REFUSES with a reason, exactly as
   `bundle execute --verb pipeline --shard` does — it is not silently dropped.
4. The resume/shard gate runs BEFORE the model provider is touched.
5. The family's pass-through verbs refuse an undeclared flag in human mode, and
   their declared surfaces cover exactly the verbs the envelope does not.
"""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server import cli, cli_envelope
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import resume, sharding, tasks


# --- fixtures -------------------------------------------------------------------

def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study(root: str, name: str = "shardcli") -> str:
    """One concept, one injection condition, three prompts."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    es.add_condition(name, {"name": "fear-a1", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 2,
                                       "alpha": 1.0}]}, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path,
           '{"id": "p0", "prompt": "One."}\n'
           '{"id": "p1", "prompt": "Two."}\n'
           '{"id": "p2", "prompt": "Three."}\n')
    return prompts_path


def _shard_partial(root: str, *, name: str, index: int, count: int,
                   experiment_hash: str | None = None) -> str:
    """A checkpointed shard partial, fabricated from its two stamps."""
    directory = os.path.join(root, "runs", f"exp-{name}-run-shard{index}of{count}")
    os.makedirs(directory, exist_ok=True)
    resume.write_state(directory, run_id=os.path.basename(directory),
                       verb="run", completed_records=1, reason="signal")
    payload = {"schemaVersion": sharding.SHARD_SCHEMA, "shardIndex": index,
               "shardCount": count, "recordRange": [0, 1],
               "expectedRecords": 1, "totalRecords": count,
               "expectedKeys": [], "ownedConditions": [],
               "experimentHash": experiment_hash or "unused"}
    with open(sharding.shard_path(directory), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return directory


@contextmanager
def _fake_model(model_id, revision=None, **kwargs):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


# --- 1. the flag reaches the engine ---------------------------------------------

def test_run_arm_threads_the_shard_spec(tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    prompts = _study(root)
    seen = {}

    def fake_run(name, prompts_file=None, root=None, dtype="auto", device=None,
                 **kwargs):
        seen.update(kwargs, name=name, prompts=prompts_file)
        return os.path.join(root or ".", "runs", "exp-shardcli-run-shard1of3")

    monkeypatch.setattr(tasks, "run", fake_run)
    assert cli.main(["experiment", "run", "shardcli", "--prompts", prompts,
                     "--shard", "1/3"]) == 0
    spec = seen["shard"]
    assert isinstance(spec, sharding.ShardSpec)
    assert (spec.index, spec.count) == (1, 3)
    capsys.readouterr()


def test_the_resume_gates_repair_sentence_is_executable(tmp_path, monkeypatch,
                                                        capsys):
    """`--resume <dir> --shard k/K` — the exact combination the gate's own
    refusal instructs, which the CLI used to reject at 64."""
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)
    partial = _shard_partial(root, name="shardcli", index=1, count=3)
    seen = {}

    def fake_run(name, prompts_file=None, root=None, dtype="auto", device=None,
                 **kwargs):
        seen.update(kwargs)
        return kwargs.get("run_directory") or ""

    monkeypatch.setattr(tasks, "run", fake_run)
    assert cli.main(["experiment", "run", "shardcli", "--resume", partial,
                     "--shard", "1/3"]) == 0
    assert seen["run_directory"] == partial
    assert (seen["shard"].index, seen["shard"].count) == (1, 3)
    capsys.readouterr()


def test_a_malformed_shard_value_is_a_usage_refusal(tmp_path, monkeypatch,
                                                    capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)

    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("tasks.run ran on a malformed --shard")

    monkeypatch.setattr(tasks, "run", explode)
    assert cli.main(["experiment", "run", "shardcli", "--shard", "3/3"]) == 64
    err = capsys.readouterr().err
    assert "usage: experiment run <name> [--shard k/K]" in err
    assert "shard index must be in [0, 3)" in err
    assert cli.main(["experiment", "run", "shardcli", "--shard", "half"]) == 64
    assert "--shard must be k/K" in capsys.readouterr().err


def test_run_is_unsharded_without_the_flag(tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    prompts = _study(root)
    seen = {}
    monkeypatch.setattr(
        tasks, "run",
        lambda name, prompts_file=None, root=None, dtype="auto", device=None,
        **kwargs: (seen.update(kwargs), "runs/x")[1])
    assert cli.main(["experiment", "run", "shardcli", "--prompts", prompts]) == 0
    assert seen["shard"] is None
    capsys.readouterr()


def test_the_shard_rides_the_json_envelope(tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)
    directory = os.path.join(root, "runs", "exp-shardcli-run-shard0of2")
    monkeypatch.setattr(
        tasks, "run",
        lambda name, prompts_file=None, root_=None, dtype="auto", device=None,
        **kwargs: directory)
    assert cli.main(["experiment", "run", "shardcli", "--shard", "0/2",
                     "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["result"]["shard"] == {"index": 0, "count": 2}
    # The next action for a PARTIAL is never "analyze it".
    assert "analyze" not in envelope["nextAction"]["verb"]


def test_run_help_declares_the_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "run", "--help"]) == 0
    assert "--shard <k/K>" in capsys.readouterr().out


# --- 2. pipeline refuses rather than dropping ------------------------------------

def test_pipeline_refuses_shard_with_the_two_paths_that_work(
        tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)

    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("a sharded pipeline ran the whole matrix")

    monkeypatch.setattr(tasks, "pipeline", explode)
    assert cli.main(["experiment", "pipeline", "shardcli", "--shard", "0/2"]) == 64
    err = capsys.readouterr().err
    assert "applies to the 'run' verb only" in err
    assert "experiment run shardcli --shard k/K" in err
    assert "--parallel K" in err


# --- 3. the gate runs before the model ------------------------------------------

def test_the_shard_gate_refuses_before_the_model_provider_is_touched(
        tmp_path, monkeypatch):
    """The §16 repair, stated as a test: the refusal costs three file reads,
    so it must not sit behind a 51 GiB load."""
    root = str(tmp_path)
    prompts = _study(root)
    partial = _shard_partial(root, name="shardcli", index=1, count=3)
    touched = []

    @contextmanager
    def exploding_provider(model_id, revision=None, **kwargs):
        touched.append(model_id)  # pragma: no cover - the assert below tells
        raise AssertionError("the model was acquired before the resume gate")
        yield  # pragma: no cover

    # Resuming shard 1/3's partial as shard 2/3 is the exact refusal the field
    # jobs hit — after staging and loading the weights.
    with pytest.raises(resume.ResumeError, match="refusing to mix shard"):
        tasks.run("shardcli", prompts, root, model_provider=exploding_provider,
                  log=lambda *_: None, run_directory=partial,
                  shard=sharding.ShardSpec(2, 3))
    assert touched == []

    # …and the same for the no-shard-flag resume of a shard partial.
    with pytest.raises(resume.ResumeError, match="same --shard"):
        tasks.run("shardcli", prompts, root, model_provider=exploding_provider,
                  log=lambda *_: None, run_directory=partial)
    assert touched == []


def test_a_complete_directory_still_refuses_before_the_model(tmp_path):
    root = str(tmp_path)
    prompts = _study(root)
    directory = os.path.join(root, "runs", "exp-shardcli-run")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "report.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"experiment": "shardcli"}, handle)
    touched = []

    @contextmanager
    def exploding_provider(model_id, revision=None, **kwargs):
        touched.append(model_id)
        raise AssertionError("the model was acquired before the resume gate")
        yield  # pragma: no cover

    with pytest.raises(resume.ResumeError, match="immutable"):
        tasks.run("shardcli", prompts, root, model_provider=exploding_provider,
                  log=lambda *_: None, run_directory=directory)
    assert touched == []


def test_the_epoch_check_still_fires_for_a_pinned_manifest(tmp_path):
    """Hoisting must not weaken the gate: a revision-pinned manifest (the
    fixture's) has a stable content hash, so the epoch refusal is hoisted with
    the rest and still fires with the model untouched."""
    root = str(tmp_path)
    prompts = _study(root)
    partial = _shard_partial(root, name="shardcli", index=0, count=2)
    _write(os.path.join(partial, "config.json"),
           json.dumps({"experimentHash": "0" * 64}))
    touched = []

    @contextmanager
    def exploding_provider(model_id, revision=None, **kwargs):
        touched.append(model_id)
        raise AssertionError("the model was acquired before the resume gate")
        yield  # pragma: no cover

    with pytest.raises(resume.ResumeError, match="refusing to mix"):
        tasks.run("shardcli", prompts, root, model_provider=exploding_provider,
                  log=lambda *_: None, run_directory=partial,
                  shard=sharding.ShardSpec(0, 2))
    assert touched == []


def test_an_ordinary_resume_still_completes_through_the_hoisted_gate(
        tmp_path, monkeypatch):
    """The gate is hoisted, not duplicated into a refusal: a legitimate
    unsharded resume still runs."""
    root = str(tmp_path)
    prompts = _study(root)
    flag = resume.CheckpointFlag()
    calls = [0]

    def generate(model, prompt, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            flag.request()
        return "answer"

    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {
                            "fear": tasks.ConceptVectorBundle(
                                vectors=_vectors(), residual_norm_per_layer=[1.0] * 4,
                                residual_norm_source="test", stimulus_hash="h")})
    monkeypatch.setattr(tasks, "generate", generate)
    seen = {}
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("shardcli", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    finished = tasks.run("shardcli", prompts, root, model_provider=_fake_model,
                         log=lambda *_: None, run_directory=seen["dir"])
    assert finished == seen["dir"]
    assert resume.is_complete(finished)


def _vectors():
    from steerlab_server.steering.vector_store import ConceptVectors
    return ConceptVectors(per_layer=[[1.0, 0.0]] * 4)


# --- 4. strict argv on the pass-through verbs ------------------------------------

def test_an_undeclared_flag_refuses_in_human_mode(tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)

    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the verb ran with a flag it does not accept")

    monkeypatch.setattr(tasks, "pipeline", explode)
    assert cli.main(["experiment", "pipeline", "shardcli", "--sahrd", "0/2"]) == 64
    err = capsys.readouterr().err
    assert "experiment pipeline does not accept --sahrd" in err
    assert "experiment pipeline accepts:" in err


def test_the_refusal_shape_matches_the_envelopes(tmp_path, monkeypatch, capsys):
    """Declared and pass-through verbs must refuse the same way, or the family
    teaches two different lessons for one mistake."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "analyze", "demo", "--nope"]) == 64
    declared = capsys.readouterr().err
    assert cli.main(["experiment", "rescore-style", "demo", "--nope"]) == 64
    passthrough = capsys.readouterr().err
    for text in (declared, passthrough):
        assert "does not accept --nope" in text
        assert "accepts:" in text


def test_a_value_that_looks_like_a_flag_is_not_read_as_one(
        tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)
    seen = {}
    monkeypatch.setattr(
        tasks, "rescore_style",
        lambda name, root_, source, **kwargs: seen.update(source=source))
    assert cli.main(["experiment", "rescore-style", "shardcli",
                     "--source", "--weird"]) == 0
    assert seen["source"] == "--weird"
    capsys.readouterr()


def test_every_pass_through_verb_declares_its_surface():
    """A new `experiment` verb cannot join the family without declaring what
    it accepts: the two tables must partition `EXPERIMENT_VERBS`."""
    declared = {spec.verb for spec in cli_envelope.VERB_SPECS
                if spec.family == "experiment"}
    passthrough = set(cli._EXPERIMENT_PASSTHROUGH_FLAGS)
    assert declared.isdisjoint(passthrough)
    assert declared | passthrough == set(cli.EXPERIMENT_VERBS)


def test_the_declared_flags_are_the_flags_the_arms_read():
    """The table is only worth having if it matches the `_flag(rest, …)` calls
    each arm makes — a flag read but undeclared would refuse the invocation
    that uses it."""
    import re
    source_path = os.path.join(os.path.dirname(os.path.abspath(cli.__file__)),
                               "cli.py")
    with open(source_path, encoding="utf-8") as handle:
        text = handle.read()
    body = text[text.index("def _experiment("):text.index("def _vectors(")]
    body += text[text.index("def _preflight_endpoints("):]
    arms = re.split(r'\n    if verb == "', body)
    for chunk in arms[1:]:
        verb = chunk[:chunk.index('"')]
        known = cli._EXPERIMENT_PASSTHROUGH_FLAGS.get(verb)
        if known is None:
            continue
        read = set(re.findall(r'_flag\(rest, "(--[a-z-]+)"\)', chunk))
        read |= set(re.findall(r'"(--[a-z-]+)" in rest', chunk))
        assert read <= set(known), (
            f"experiment {verb} reads {sorted(read - set(known))} but does "
            "not declare it")


def test_a_pass_through_verb_still_runs_with_its_own_flags(
        tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _study(root)
    seen = {}
    monkeypatch.setattr(
        tasks, "pipeline",
        lambda name, root_, dtype, device, **kwargs: seen.update(kwargs))
    assert cli.main(["experiment", "pipeline", "shardcli",
                     "--resume", "runs/whatever"]) == 0
    assert seen["pipeline_run_directory"] == "runs/whatever"
    capsys.readouterr()
