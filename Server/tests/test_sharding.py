"""Multi-GPU sharded study runs (2026-07-22).

The enabling fact is test-pinned elsewhere (derive_seed has no cross-record
state; greedy records never touch the RNG), so ANY partition of a run's
record set must produce byte-identical outputs. These tests hold the feature
to that claim:

1. Pure shard math — balanced contiguous ranges that partition the record
   set; parse errors; condition/battery ownership by first record.
2. THE ACCEPTANCE TEST — a fake-model study (2 conditions × 3 prompts × 2
   samples, derivedSHA256, choice instrument, pinned battery) run once as a
   single job and once as 3 shards + merge: generations.jsonl byte-identical,
   report equivalent (modulo the merged report's `sharded` provenance
   block), battery cells identical.
3. Completeness refusals — a missing/incomplete shard and a duplicated cell
   both refuse loudly, leaving the partials intact.
4. Shard checkpoint → resume (same --shard) → merge, byte-identical.
5. Submission fan-out — parallelJobs=K creates one parent + K shard sbatch
   jobs; non-sharding verbs ignore the knob with a logged note.
6. Parent state derivation + the reconciler's merge/failed/checkpointed
   transitions, including parent cancel fan-out.

The fake generate embeds a torch RNG draw when sampling, so the byte-identity
tests genuinely exercise the per-record seeding (fork_rng + derived seed) —
not just record ordering.
"""

import json
import os
import shutil
import stat
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment import resume
from steerlab_server.experiment import run_status
from steerlab_server.experiment import sharding
from steerlab_server.experiment import tasks
from steerlab_server.steering.vector_store import ConceptVectors

FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")


# --- 1. pure shard math ---------------------------------------------------------

def test_shard_bounds_partition_and_balance():
    for total in (0, 1, 5, 12, 13, 100):
        for count in (1, 2, 3, 5, 7):
            ranges = [sharding.shard_bounds(total, k, count)
                      for k in range(count)]
            # Contiguous partition of [0, total).
            assert ranges[0][0] == 0
            assert ranges[-1][1] == total
            for (a, b), (c, _d) in zip(ranges, ranges[1:]):
                assert b == c
                assert b >= a
            sizes = [b - a for a, b in ranges]
            assert max(sizes) - min(sizes) <= 1  # balanced


def test_parse_shard_accepts_k_of_K_and_refuses_garbage():
    spec = sharding.parse_shard("1/3")
    assert (spec.index, spec.count) == (1, 3)
    for bad in ("3/3", "-1/3", "0", "a/b", "1/0", "", "1/999"):
        with pytest.raises(sharding.ShardError):
            sharding.parse_shard(bad)


def test_plan_shard_condition_ownership_is_first_record():
    prompts = [{"id": f"p{i}", "prompt": "x"} for i in range(3)]
    keys = sharding.expected_record_keys(
        condition_names=["baseline", "steered"], prompts=prompts,
        wants_choice=False, wants_sampled=True, sample_count=2)
    assert len(keys) == 12
    owners = {}
    for k in range(3):
        plan = sharding.plan_shard(
            sharding.ShardSpec(k, 3), all_keys=keys,
            condition_names=["baseline", "steered"])
        for name in plan.owned_conditions:
            owners[name] = k
    # baseline's first record is index 0 (shard 0); steered's is index 6,
    # which falls in shard 1 (records 4..7 of 12).
    assert owners == {"baseline": 0, "steered": 1}
    union = set()
    for k in range(3):
        plan = sharding.plan_shard(
            sharding.ShardSpec(k, 3), all_keys=keys,
            condition_names=["baseline", "steered"])
        assert union.isdisjoint(plan.allowed_keys)
        union |= plan.allowed_keys
    assert union == set(keys)


def test_expected_record_keys_honor_instrument_scope():
    """The planner applies the SAME outcomeInstrumentScope rule as the
    executor (2026-08-04, `test-compare-2-2`): an option-bearing item whose
    responseFormat is outside the declared scope produces NO instrument
    readout, so it must contribute no expected instrument key — otherwise
    every shard stamp expects phantom records and the merge refuses a run
    whose shards all succeeded."""
    prompts = [
        {"id": "in-scope", "prompt": "x", "options": ["A", "B"],
         "responseFormat": "label"},
        {"id": "out-of-scope", "prompt": "y", "options": ["A", "B"],
         "responseFormat": "json"},
        {"id": "undeclared", "prompt": "z", "options": ["A", "B"]},
    ]
    scope = {"responseFormats": ["label"], "itemCount": 1}
    keys = sharding.expected_record_keys(
        condition_names=["baseline"], prompts=prompts,
        wants_choice=True, wants_sampled=True, sample_count=1,
        instrument_scope=scope)
    instrument_ids = [k[2] for k in keys if k[4] == "instrument"]
    # Only the declared-in-scope item is expected; the undeclared row is
    # excluded too (a scope exists precisely because the file is mixed —
    # `response_format.scope_includes` mirrors this conservative reading).
    assert instrument_ids == ["in-scope"]
    sampled_ids = [k[2] for k in keys if k[4] == "sampled"]
    assert sampled_ids == ["in-scope", "out-of-scope", "undeclared"]
    # No scope declared: every option-bearing item is expected — the
    # historical behavior, unchanged.
    unscoped = sharding.expected_record_keys(
        condition_names=["baseline"], prompts=prompts,
        wants_choice=True, wants_sampled=False, sample_count=1)
    assert [k[2] for k in unscoped] == [
        "in-scope", "out-of-scope", "undeclared"]


# --- shared fixture: a fake-model stochastic study ------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study_fixture(root, name):
    """2 conditions (baseline + steered) × 3 prompts × 2 samples,
    derivedSHA256, one option-bearing prompt (choice instrument), and a
    pinned 2-item capability battery."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    es.add_condition(name, {"name": "fear-a1", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 2, "alpha": 1.0}]},
                     root)
    battery_rel = "prompts/batteries/tiny.jsonl"
    battery_text = ('{"prompt": "2+2?", "answer": "4"}\n'
                    '{"prompt": "Capital of France?", "answer": "Paris"}\n')
    _write(os.path.join(root, battery_rel), battery_text)
    import hashlib
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.7
    raw["samplesPerItem"] = 2
    raw["seedPolicy"] = "derivedSHA256"
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    raw["capabilityBatteryFile"] = battery_rel
    raw["capabilityBatteryHash"] = hashlib.sha256(
        battery_text.encode("utf-8")).hexdigest()
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path,
           '{"id": "p0", "prompt": "Decide the case."}\n'
           '{"id": "p1", "prompt": "Choose.", "options": ["A", "B"], "target": "B"}\n'
           '{"id": "p2", "prompt": "Explain the ruling."}\n')
    return prompts_path


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


class _FakeOptionScore:
    def __init__(self, option, logprob):
        self.option = option
        self.token_ids = [7]
        self.logprob = logprob


class _FakeChoice:
    def __init__(self, options):
        self.options = [_FakeOptionScore(o, -0.5 - i)
                        for i, o in enumerate(options)]

    def as_record_fields(self):
        return {"instrument": "answerTokenLogprob",
                "options": [s.option for s in self.options],
                "selected": self.options[0].option,
                "choiceProbability": {s.option: 0.5 for s in self.options},
                "logOdds": {s.option: 0.0 for s in self.options}}


def _seed_sensitive_generate(counter=None, arm_flag_at=None, flag=None):
    """Deterministic per (prompt, injections, RNG state): when sampling, the
    output embeds a torch draw, so it reproduces ONLY if the per-record
    derived seeding is identical — the real byte-identity claim."""
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, **kwargs):
        if counter is not None:
            counter[0] += 1
            if arm_flag_at is not None and counter[0] == arm_flag_at:
                flag.request()
        steered = "steered" if injections else "plain"
        if temperature and temperature > 0:
            draw = int(torch.randint(0, 10 ** 9, (1,)).item())
            return f"{steered} answer to {prompt} [{draw}]"
        return f"{steered} answer to {prompt}"
    return generate


def _patch_study_fakes(monkeypatch, generate_fn):
    from steerlab_server.experiment import logprob as logprob_mod
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", generate_fn)
    monkeypatch.setattr(logprob_mod, "score_options",
                        lambda model, prompt, options, **kw: _FakeChoice(options))


def _run_single(root, name, prompts, monkeypatch):
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    return tasks.run(name, prompts, root, model_provider=_fake_model,
                     log=lambda *_: None)


def _run_shards(root, name, prompts, monkeypatch, count=3):
    dirs = []
    for k in range(count):
        _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
        dirs.append(tasks.run(name, prompts, root, model_provider=_fake_model,
                              log=lambda *_: None,
                              shard=sharding.ShardSpec(k, count)))
    return dirs


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


# --- 2. THE ACCEPTANCE TEST -----------------------------------------------------

def test_three_shards_plus_merge_is_byte_identical_to_single_job(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardacc")

    single_dir = _run_single(root, "shardacc", prompts, monkeypatch)
    single_generations = _read(os.path.join(single_dir, "generations.jsonl"))
    single_battery = _read(os.path.join(single_dir, "battery.jsonl"))
    with open(os.path.join(single_dir, "report.json"), encoding="utf-8") as fh:
        single_report = json.load(fh)
    # 2 conditions × (3 prompts × 2 samples + 1 instrument readout) = 14.
    assert len(single_generations.splitlines()) == 14

    shard_dirs = _run_shards(root, "shardacc", prompts, monkeypatch, count=3)
    for directory in shard_dirs:
        assert os.path.isfile(os.path.join(directory, "shard.json"))
        assert resume.is_complete(directory)
    # Shards partition the records: no shard holds everything.
    shard_counts = [len(_read(os.path.join(d, "generations.jsonl")).splitlines())
                    for d in shard_dirs]
    assert sum(shard_counts) == 14
    assert all(count < 14 for count in shard_counts)

    merged = sharding.merge_shard_runs(
        "shardacc", shard_dirs, root=root, shard_job_ids=["j0", "j1", "j2"])

    assert _read(os.path.join(merged, "generations.jsonl")) == single_generations
    assert _read(os.path.join(merged, "battery.jsonl")) == single_battery
    with open(os.path.join(merged, "report.json"), encoding="utf-8") as fh:
        merged_report = json.load(fh)
    sharded = merged_report.pop("sharded")
    assert sharded["shardCount"] == 3
    assert sharded["shardJobIDs"] == ["j0", "j1", "j2"]
    assert len(sharded["shardRuns"]) == 3
    assert merged_report == single_report  # equivalent modulo provenance
    # Battery cells identical (already byte-proved above; also via report).
    for condition in ("baseline", "fear-a1"):
        assert (merged_report["conditions"][condition]["capabilityBattery"]
                == single_report["conditions"][condition]["capabilityBattery"])
    # The merged run is an ordinary complete run (metrics/summaries/config).
    for artifact in ("metrics.csv", "summaries.csv", "config.json",
                     "experiment-hash.txt", "substrate.json"):
        assert os.path.isfile(os.path.join(merged, artifact))
        assert (_read(os.path.join(merged, artifact if artifact != "config.json"
                                   else artifact))
                is not None)
    assert not os.path.exists(os.path.join(merged, "shard.json"))
    # config.json is the closed schema-2 contract: byte-compare key sets
    # against the single run's (shard provenance must NOT leak in).
    with open(os.path.join(merged, "config.json"), encoding="utf-8") as fh:
        merged_config = json.load(fh)
    with open(os.path.join(single_dir, "config.json"), encoding="utf-8") as fh:
        single_config = json.load(fh)
    assert set(merged_config) == set(single_config)
    # metrics/summaries rebuilt over merged records match the single job's.
    assert (_read(os.path.join(merged, "metrics.csv"))
            == _read(os.path.join(single_dir, "metrics.csv")))
    assert (_read(os.path.join(merged, "summaries.csv"))
            == _read(os.path.join(single_dir, "summaries.csv")))
    # Shard partials are KEPT (deliberate: immutable runs area).
    for directory in shard_dirs:
        assert os.path.isdir(directory)


def test_sharded_plan_covers_every_slot_condition_times_prompt(
        tmp_path, monkeypatch):
    """Regression for the 2026-08-11 c20-* incident shape: a logprob-only
    study whose arms are manifest-level concept-SLOT conditions (several
    per concept, no variants). Every shard stamp must enumerate those
    conditions exactly like variant conditions — the union of expectedKeys
    over the fan-out is the full condition × option-bearing-prompt matrix,
    each condition owned exactly once, and the generated records cover
    every arm."""
    root = str(tmp_path)
    name = "shardslots"
    cells = [("fear", 1, 0.5), ("fear", 2, 1.0), ("calm", 3, 1.5)]
    for concept in ("fear", "calm"):
        cdir = os.path.join(root, "prompts", "concepts", concept)
        _write(os.path.join(cdir, "positive.jsonl"), '{"text": "pos"}\n')
        _write(os.path.join(cdir, "negative.jsonl"), '{"text": "neg"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear", "calm"], root=root)
    for concept, layer, alpha in cells:
        es.add_condition(name, {
            "name": f"{concept}-L{layer}", "bandWidth": 1,
            "alphaInNormUnits": False,
            "slots": [{"concept": concept, "layer": layer, "alpha": alpha}]},
            root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0
    raw["maxTokens"] = 8
    raw["outcomeInstruments"] = ["answerTokenLogprob"]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "slots.jsonl")
    _write(prompts_path, "".join(
        json.dumps({"id": f"p{i}", "prompt": f"case {i}",
                    "options": ["A", "B"], "target": "A"}) + "\n"
        for i in range(5)))

    condition_names = ["baseline"] + [f"{c}-L{l}" for c, l, _ in cells]
    shard_dirs = []
    all_expected: list[tuple] = []
    owners: dict[str, int] = {}
    for k in range(3):
        _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
        monkeypatch.setattr(
            tasks, "_extract_all",
            lambda model, manifest, root: {c: _fake_bundle()
                                           for c in ("fear", "calm")})
        directory = tasks.run(name, prompts_path, root,
                              model_provider=_fake_model,
                              log=lambda *_: None,
                              shard=sharding.ShardSpec(k, 3))
        shard_dirs.append(directory)
        with open(os.path.join(directory, "shard.json"),
                  encoding="utf-8") as handle:
            stamp = json.load(handle)
        assert stamp["totalRecords"] == len(condition_names) * 5
        all_expected.extend(tuple(key) for key in stamp["expectedKeys"])
        for owned in stamp["ownedConditions"]:
            assert owners.setdefault(owned, k) == k
    # The union of shard plans IS the full matrix — no condition dropped.
    assert len(all_expected) == len(set(all_expected)) \
        == len(condition_names) * 5
    expected_matrix = {(condition, f"p{i}")
                       for condition in condition_names for i in range(5)}
    assert {(key[0], key[2]) for key in all_expected} == expected_matrix
    assert set(owners) == set(condition_names)
    # And the executed records agree with the plan.
    generated = set()
    for directory in shard_dirs:
        for line in _read(os.path.join(directory,
                                       "generations.jsonl")).splitlines():
            record = json.loads(line)
            generated.add((record["condition"], record["promptID"]))
    assert generated == expected_matrix


def test_declared_agent_comparison_without_arms_refuses_before_sharding(
        tmp_path, monkeypatch):
    """The 2026-08-11 incident itself: the same slot-condition study with a
    declared ``studyType: agentComparison`` and no variant conditions used
    to run BASELINE ONLY on every shard, silently. It now refuses at run
    start — before any shard directory or model work."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardinert")
    raw = es.load_raw("shardinert", root)
    raw["studyType"] = "agentComparison"
    es.save_raw(raw, root)
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    with pytest.raises(RuntimeError, match="BASELINE ONLY"):
        tasks.run("shardinert", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, shard=sharding.ShardSpec(0, 3))
    with pytest.raises(RuntimeError, match="BASELINE ONLY"):
        tasks.run("shardinert", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)


def test_battery_runs_exactly_once_per_condition_across_shards(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardbat")
    shard_dirs = _run_shards(root, "shardbat", prompts, monkeypatch, count=3)
    owners: dict[str, int] = {}
    for index, directory in enumerate(shard_dirs):
        path = os.path.join(directory, "battery.jsonl")
        if not os.path.isfile(path):
            continue
        for line in _read(path).splitlines():
            condition = json.loads(line)["condition"]
            assert owners.setdefault(condition, index) == index
    assert set(owners) == {"baseline", "fear-a1"}


# --- 3. completeness refusals ---------------------------------------------------

def test_merge_refuses_missing_shard_and_incomplete_shard(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardmiss")
    shard_dirs = _run_shards(root, "shardmiss", prompts, monkeypatch, count=3)

    with pytest.raises(sharding.ShardMergeError, match="missing, duplicated"):
        sharding.merge_shard_runs("shardmiss", shard_dirs[:2], root=root)

    # An incomplete shard (no report.json — checkpointed/cancelled) refuses
    # with the resume guidance, partials untouched.
    os.remove(os.path.join(shard_dirs[1], "report.json"))
    with pytest.raises(sharding.ShardMergeError,
                       match="resume the incomplete shard"):
        sharding.merge_shard_runs("shardmiss", shard_dirs, root=root)
    assert os.path.isfile(os.path.join(shard_dirs[0], "generations.jsonl"))


def test_merge_refuses_missing_and_duplicated_cells(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "sharddup")
    shard_dirs = _run_shards(root, "sharddup", prompts, monkeypatch, count=3)

    # Duplicated cell: copy shard 0's first record into shard 1's stream.
    zero = os.path.join(shard_dirs[0], "generations.jsonl")
    one = os.path.join(shard_dirs[1], "generations.jsonl")
    first_line = _read(zero).splitlines(keepends=True)[0]
    original = _read(one)
    with open(one, "ab") as handle:
        handle.write(first_line)
    with pytest.raises(sharding.ShardMergeError, match="duplicated cells"):
        sharding.merge_shard_runs("sharddup", shard_dirs, root=root)

    # Missing cell: drop shard 1's last record (report.json still present).
    lines = original.splitlines(keepends=True)
    with open(one, "wb") as handle:
        handle.writelines(lines[:-1])
    with pytest.raises(sharding.ShardMergeError,
                       match="expected records are missing"):
        sharding.merge_shard_runs("sharddup", shard_dirs, root=root)


def test_merge_refuses_manifest_drift(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "sharddrift")
    shard_dirs = _run_shards(root, "sharddrift", prompts, monkeypatch, count=2)
    raw = es.load_raw("sharddrift", root)
    raw["maxTokens"] = 32
    es.save_raw(raw, root)
    with pytest.raises(sharding.ShardMergeError, match="epoch guard"):
        sharding.merge_shard_runs("sharddrift", shard_dirs, root=root)


# --- 4. shard checkpoint → resume → merge ---------------------------------------

def test_shard_checkpoint_resume_then_merge_is_byte_identical(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardckpt")
    single_dir = _run_single(root, "shardckpt", prompts, monkeypatch)
    single_generations = _read(os.path.join(single_dir, "generations.jsonl"))

    # Shards 0 and 2 complete normally; shard 1 checkpoints mid-generation.
    dirs: list[str] = [None] * 3  # type: ignore[list-item]
    for k in (0, 2):
        _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
        dirs[k] = tasks.run("shardckpt", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None,
                            shard=sharding.ShardSpec(k, 3))
    flag = resume.CheckpointFlag()
    counter = [0]
    _patch_study_fakes(monkeypatch,
                       _seed_sensitive_generate(counter, arm_flag_at=2, flag=flag))
    seen = {}
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("shardckpt", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  shard=sharding.ShardSpec(1, 3),
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    dirs[1] = seen["dir"]
    assert resume.is_resumable(dirs[1])

    # Merging with the incomplete shard REFUSES; the partials stay intact.
    with pytest.raises(sharding.ShardMergeError,
                       match="resume the incomplete shard"):
        sharding.merge_shard_runs("shardckpt", dirs, root=root)

    # Resuming under a DIFFERENT shard spec refuses; the right one completes.
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    with pytest.raises(resume.ResumeError, match="refusing to mix shard"):
        tasks.run("shardckpt", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=dirs[1],
                  shard=sharding.ShardSpec(2, 3))
    with pytest.raises(resume.ResumeError, match="same --shard"):
        tasks.run("shardckpt", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=dirs[1])
    resumed = tasks.run("shardckpt", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None, run_directory=dirs[1],
                        shard=sharding.ShardSpec(1, 3))
    assert resumed == dirs[1]
    assert resume.is_complete(dirs[1])

    merged = sharding.merge_shard_runs("shardckpt", dirs, root=root)
    assert _read(os.path.join(merged, "generations.jsonl")) == single_generations


# --- 5. submission fan-out ------------------------------------------------------

@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                     | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    state_file = tmp_path / "slurm-state.json"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(state_file))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.delenv("FAKE_SBATCH_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)

    class Handle:
        def set_state(self, job_id, state, exit_code="0:0"):
            table = {}
            if state_file.exists():
                table = json.loads(state_file.read_text(encoding="utf-8"))
            table[str(job_id)] = {"state": state, "exit": exit_code}
            state_file.write_text(json.dumps(table), encoding="utf-8")

        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            return path.read_text(encoding="utf-8").splitlines() \
                if path.exists() else []

    return Handle()


def _manager(tmp_path, name="jobs.sqlite"):
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    return JobManager(DurableJobStore(str(tmp_path / name)),
                      capability_provider=lambda: {})


def _staged_bundle(tmp_path, name="shardsub"):
    source = str(tmp_path / "source")
    _study_fixture(source, name)
    return bundles.package_experiment(name, root=source)["bundlePath"]


def test_parallel_submission_creates_parent_and_shard_children(
        tmp_path, monkeypatch, fake_slurm):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    os.makedirs(tmp_path / "target", exist_ok=True)
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path)
    jobs = _manager(tmp_path)
    submission = submit_run_bundle(
        bundle_path, verb="run", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=3)
    assert submission.shard_job_ids and len(submission.shard_job_ids) == 3
    assert submission.to_dict()["shardJobIDs"] == submission.shard_job_ids
    parent = jobs.get(submission.job_id)
    assert parent is not None
    rr = parent.requested_resources
    assert rr["parallelJobs"] == 3
    assert rr["shardChildren"] == submission.shard_job_ids
    assert rr["shardMerge"]["experiment"] == "shardsub"
    assert parent.executor_job_id is None
    assert (parent.result or {})["shardJobs"] == submission.shard_job_ids
    assert any("sharded across 3 GPU jobs" in line
               for line in parent.all_logs())
    # 3 real sbatch submissions, each with its own script carrying --shard.
    assert len(fake_slurm.calls("sbatch")) == 3
    for index, child_id in enumerate(submission.shard_job_ids):
        child = jobs.get(child_id)
        assert child.status == "submitted"
        assert child.requested_resources["shardIndex"] == index
        assert child.requested_resources["parentJob"] == parent.id
        script = child.requested_resources["scriptPath"]
        text = open(script, encoding="utf-8").read()
        assert f"--shard {index}/3" in text
        assert "--no-evidence" in text  # merge packages evidence, not shards


def test_non_sharding_verbs_ignore_parallel_jobs_with_a_note(
        tmp_path, monkeypatch, fake_slurm):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path, name="shardver")
    jobs = _manager(tmp_path)
    submission = submit_run_bundle(
        bundle_path, verb="validate", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target2"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=4)
    assert submission.shard_job_ids is None
    job = jobs.get(submission.job_id)
    assert any("parallelJobs=4 ignored" in line for line in job.all_logs())
    assert len(fake_slurm.calls("sbatch")) == 1  # single ordinary submission


def test_parallel_jobs_cap_refuses_loudly(tmp_path, monkeypatch, fake_slurm):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path, name="shardcap")
    with pytest.raises(ValueError, match="fan-out cap"):
        submit_run_bundle(bundle_path, verb="run", jobs=_manager(tmp_path),
                          executor="slurm", parallel_jobs=200)


# --- 6. parent state derivation + reconciler merge ------------------------------

def test_derive_shard_parent_state_matrix():
    from steerlab_server.api.jobs import derive_shard_parent_state as derive
    assert derive(["succeeded", "succeeded"]) == "succeeded"
    assert derive(["succeeded", "running"]) == "running"
    assert derive(["submitted", "submitted"]) == "submitted"
    assert derive(["succeeded", "checkpointed"]) == "checkpointed"
    # A checkpointed shard is actionable (Resume) even while others run.
    assert derive(["checkpointed", "running"]) == "checkpointed"
    assert derive(["failed", "running"]) == "failed"
    assert derive(["cancelled", "succeeded"]) == "cancelled"


def test_reconciler_merges_when_all_shards_succeed(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardrec")
    single_dir = _run_single(root, "shardrec", prompts, monkeypatch)
    shard_dirs = _run_shards(root, "shardrec", prompts, monkeypatch, count=2)

    jobs = _manager(tmp_path)
    children = []
    for index, directory in enumerate(shard_dirs):
        children.append(jobs.record_external(
            "study-submit-bundle-shard", status="succeeded", executor="slurm",
            executor_job_id=str(100 + index),
            result={"runDirectory": directory,
                    "shard": {"index": index, "count": 2}}))
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "recordsDirectory": str(tmp_path / "records"),
            "shardMerge": {"experiment": "shardrec", "verb": "run",
                           "targetRoot": root, "packageEvidence": False}},
        result={"shardJobs": [c.id for c in children]})

    assert jobs._reconcile_shard_parents() == 1
    parent = jobs.get(parent.id)
    assert parent.status == "succeeded"
    merged = (parent.result or {})["runDirectory"]
    assert (_read(os.path.join(merged, "generations.jsonl"))
            == _read(os.path.join(single_dir, "generations.jsonl")))
    # The merged run's jobId is the PARENT record's id, never the merging
    # process's own environment (the reconcile runs on the controller,
    # whose SLURM_JOB_ID stamped one stale allocation id into every run the
    # same controller incarnation merged — observed 2026-08-06 as four runs
    # sharing jobId 47285267 while their shard ids differed).
    config = json.load(open(os.path.join(merged, "config.json")))
    assert config["jobId"] == parent.id


def test_merged_run_never_stamps_the_controllers_allocation(tmp_path,
                                                            monkeypatch):
    """The field case, reproduced: reconcile under a controller-style
    SLURM_JOB_ID env — the merged config.json must carry the parent record
    id, not the controller's allocation."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardjobid")
    shard_dirs = _run_shards(root, "shardjobid", prompts, monkeypatch, count=2)

    monkeypatch.setenv("SLURM_JOB_ID", "47285267")
    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="succeeded", executor="slurm",
        executor_job_id=str(200 + index),
        result={"runDirectory": directory,
                "shard": {"index": index, "count": 2}})
        for index, directory in enumerate(shard_dirs)]
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "recordsDirectory": str(tmp_path / "records"),
            "shardMerge": {"experiment": "shardjobid", "verb": "run",
                           "targetRoot": root, "packageEvidence": False}},
        result={"shardJobs": [c.id for c in children]})

    assert jobs._reconcile_shard_parents() == 1
    merged = (jobs.get(parent.id).result or {})["runDirectory"]
    config = json.load(open(os.path.join(merged, "config.json")))
    assert config["jobId"] == parent.id
    assert config["jobId"] != "47285267"


def test_reconciler_derives_failed_and_checkpointed_parent(tmp_path):
    jobs = _manager(tmp_path)
    ok = jobs.record_external("study-submit-bundle-shard", status="succeeded",
                              executor="slurm", executor_job_id="1",
                              result={"runDirectory": "/nowhere"})
    bad = jobs.record_external("study-submit-bundle-shard", status="failed",
                               executor="slurm", executor_job_id="2")
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={"shardChildren": [ok.id, bad.id],
                             "shardMerge": {"experiment": "x",
                                            "targetRoot": "/nowhere"}})
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "failed"
    assert bad.id in (parent.error or "")

    ckpt = jobs.record_external("study-submit-bundle-shard",
                                status="checkpointed", executor="slurm",
                                executor_job_id="3")
    parent2 = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={"shardChildren": [ok.id, ckpt.id],
                             "shardMerge": {"experiment": "x",
                                            "targetRoot": "/nowhere"}})
    jobs._reconcile_shard_parents()
    assert jobs.get(parent2.id).status == "checkpointed"


def test_merge_refusal_fails_parent_and_keeps_partials(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardrecfail")
    shard_dirs = _run_shards(root, "shardrecfail", prompts, monkeypatch, count=2)
    # Sabotage: drop a record from shard 1 (missing cell at merge time).
    path = os.path.join(shard_dirs[1], "generations.jsonl")
    lines = _read(path).splitlines(keepends=True)
    with open(path, "wb") as handle:
        handle.writelines(lines[:-1])

    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="succeeded", executor="slurm",
        executor_job_id=str(index), result={"runDirectory": directory})
        for index, directory in enumerate(shard_dirs)]
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "shardMerge": {"experiment": "shardrecfail", "verb": "run",
                           "targetRoot": root, "packageEvidence": False}})
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "failed"
    assert "expected records are missing" in (parent.error or "")
    for directory in shard_dirs:
        assert os.path.isfile(os.path.join(directory, "generations.jsonl"))


def test_parent_cancel_fans_out_to_shard_children(tmp_path, fake_slurm):
    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="submitted", executor="slurm",
        executor_job_id=str(500 + index)) for index in range(2)]
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={"shardChildren": [c.id for c in children]})
    assert jobs.cancel(parent.id) is True
    assert jobs.get(parent.id).status == "cancelled"
    for child in children:
        assert jobs.get(child.id).status == "cancelled"
    assert len(fake_slurm.calls("scancel")) == 2


def test_pipeline_parent_submits_continuation_after_merge(
        tmp_path, monkeypatch, fake_slurm):
    """A run-first pipeline shards its run stage; after the merge the
    reconciler seeds a pipeline ledger whose run stage is complete and
    submits ONE continuation job for the remaining stages — the existing
    pipeline resume machinery then skips run and executes the rest."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardpipe")
    raw = es.load_raw("shardpipe", root)
    raw["pipeline"] = {"stages": ["run", "analyze"]}
    es.save_raw(raw, root)
    shard_dirs = _run_shards(root, "shardpipe", prompts, monkeypatch, count=2)

    monkeypatch.setenv("STEERLAB_ROOT", root)
    records_dir = str(tmp_path / "records")
    os.makedirs(records_dir, exist_ok=True)
    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="succeeded", executor="slurm",
        executor_job_id=str(index), result={"runDirectory": directory})
        for index, directory in enumerate(shard_dirs)]
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "recordsDirectory": records_dir,
            "walltime": "00:30:00",
            "shardMerge": {"experiment": "shardpipe", "verb": "pipeline",
                           "targetRoot": root, "packageEvidence": False,
                           "bundlePath": str(tmp_path / "bundle.tar.gz"),
                           "dtype": "auto",
                           "submissionDirectory": str(tmp_path / "sub")}})
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "running"
    continuation_id = parent.requested_resources["continuationJob"]
    continuation = jobs.get(continuation_id)
    assert continuation is not None and continuation.status == "submitted"
    # The continuation is an ordinary bundle-execute pipeline job whose
    # resume pointer names the seeded pipeline directory.
    pointer = resume.read_pointer(resume.pointer_path_for_record(
        os.path.join(records_dir, f"{continuation_id}.json")))
    assert pointer is not None and pointer["verb"] == "pipeline"
    pipeline_dir = pointer["runDirectory"]
    with open(os.path.join(pipeline_dir, "pipeline.json"),
              encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert ledger["stages"] == ["run", "analyze"]
    assert ledger["stageResults"]["run"]["status"] == "completed"
    merged = ledger["stageResults"]["run"]["runDirectory"]
    assert os.path.isfile(os.path.join(merged, "report.json"))
    assert (parent.result or {})["mergedRunDirectory"] == merged
    script = open(continuation.requested_resources["scriptPath"],
                  encoding="utf-8").read()
    assert "--verb pipeline" in script

    # The continuation finishing terminally completes the parent.
    continuation.status = "succeeded"
    continuation.result = {"runDirectory": pipeline_dir}
    jobs.store.update(continuation)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "succeeded"
    assert (parent.result or {})["runDirectory"] == pipeline_dir


def test_bundle_execute_refuses_shard_on_non_run_verbs(tmp_path):
    bundle_path = _staged_bundle(tmp_path, name="shardverb")
    with pytest.raises(bundles.BundleError, match="'run' verb only"):
        bundles.execute_run_bundle(bundle_path, verb="validate",
                                   target_root=str(tmp_path / "t"),
                                   shard="0/2")


# --- finding 3 (2026-07-22): fan-out submission is failure-atomic ---------------

def test_fanout_shard_submission_failure_cancels_and_fails_parent(
        tmp_path, monkeypatch, fake_slurm):
    """A shard sbatch failing mid-loop must not strand the earlier shards as
    disconnected children with no ids for the caller: the parent (created
    FIRST) fails with a plain-language account, the submitted shards are
    scancelled, and the caller receives the parent id."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "2")  # shard 3 of 4 fails
    os.makedirs(tmp_path / "target", exist_ok=True)
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path, name="shardatomic")
    jobs = _manager(tmp_path)
    submission = submit_run_bundle(
        bundle_path, verb="run", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=4)

    # The caller gets the PARENT id (so the app can show the aborted
    # fan-out) and no shard ids — nothing is running.
    assert submission.shard_job_ids is None
    parent = jobs.get(submission.job_id)
    assert parent is not None
    assert parent.status == "failed"
    assert "shard 3 of 4 failed to submit" in parent.error
    assert "fake quota exceeded" in parent.error
    # Per-shard outcomes, honestly recorded (finding 3, 2026-07-23): both
    # scancels went through, so the message says CONFIRMED — never an
    # unconditional "were cancelled".
    assert "were cancelled (confirmed)" in parent.error
    assert "cleanupIncomplete" not in (parent.result or {})

    # Both submitted shards were scancelled and are owned by the parent —
    # no orphan records.
    assert len(fake_slurm.calls("sbatch")) == 3   # 2 ok + 1 failed attempt
    assert len(fake_slurm.calls("scancel")) == 2
    children = parent.requested_resources["shardChildren"]
    assert len(children) == 2
    for cid in children:
        child = jobs.get(cid)
        assert child.status == "cancelled"
        assert child.requested_resources["parentJob"] == parent.id
    # Exactly the parent + the two attached shards exist — nothing dangling.
    assert {j.id for j in jobs.list()} == {parent.id, *children}


def test_fanout_success_path_creates_parent_before_shards(
        tmp_path, monkeypatch, fake_slurm):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    os.makedirs(tmp_path / "target", exist_ok=True)
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path, name="shardorder")
    jobs = _manager(tmp_path)
    submission = submit_run_bundle(
        bundle_path, verb="run", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=2)
    parent = jobs.get(submission.job_id)
    assert parent.status == "submitted"
    for cid in submission.shard_job_ids:
        assert parent.created_at <= jobs.get(cid).created_at


def test_reconciler_leaves_pending_fanout_parents_alone(tmp_path):
    """A parent still ATTACHING its shards (status "pending") must not have
    its state derived from a partial child list — the submit loop owns it."""
    jobs = _manager(tmp_path)
    ok = jobs.record_external("study-submit-bundle-shard", status="succeeded",
                              executor="slurm", executor_job_id="1",
                              result={"runDirectory": "/nowhere"})
    parent = jobs.record_external(
        "study-submit-bundle", status="pending", executor="slurm",
        requested_resources={"parallelJobs": 2, "shardChildren": [ok.id],
                             "shardMerge": {"experiment": "x",
                                            "targetRoot": "/nowhere"}})
    assert jobs._reconcile_shard_parents() == 0
    assert jobs.get(parent.id).status == "pending"


def test_restart_sweeps_orphaned_pending_fanout_parent(tmp_path, fake_slurm):
    """A server crash mid fan-out leaves the parent "pending" with the
    submit loop's thread gone — the restart sweep fails it honestly, and
    (finding 3, 2026-07-23) actively CANCELS the shards that did get
    submitted instead of asking the operator to."""
    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="submitted", executor="slurm",
        executor_job_id=str(600 + index)) for index in range(2)]
    parent = jobs.record_external(
        "study-submit-bundle", status="pending", executor="slurm",
        requested_resources={"parallelJobs": 3,
                             "shardChildren": [c.id for c in children]})
    restarted = _manager(tmp_path)
    swept = restarted.get(parent.id)
    assert swept.status == "failed"
    assert "orphaned by server restart mid fan-out" in swept.error
    assert "were cancelled (confirmed)" in swept.error
    for child in children:
        assert restarted.get(child.id).status == "cancelled"
    assert len(fake_slurm.calls("scancel")) == 2


def test_fanout_cancel_failure_is_honest_and_reconciler_retries(
        tmp_path, monkeypatch, fake_slurm):
    """Finding 3 (2026-07-23): a failed scancel during fan-out cleanup must
    NOT read as a clean stop — the parent's message names the cancelled vs
    cancel-FAILED shards with their live Slurm ids, stamps
    ``cleanupIncomplete``, and the reconciler keeps retrying the
    cancellation until the scheduler confirms."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "2")  # shard 3 of 3 fails
    monkeypatch.setenv("FAKE_SCANCEL_FAIL", "fake scancel rejected")
    os.makedirs(tmp_path / "target", exist_ok=True)
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path, name="shardcleanup")
    jobs = _manager(tmp_path)
    submission = submit_run_bundle(
        bundle_path, verb="run", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=3)
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    # The honest account: no "were cancelled" lie; the FAILED shards are
    # named with their Slurm ids and the retry promise.
    assert "were cancelled (confirmed)" not in parent.error
    assert "cancel FAILED for shard job(s)" in parent.error
    assert "allocations may still be running" in parent.error
    children = parent.requested_resources["shardChildren"]
    for cid in children:
        child = jobs.get(cid)
        assert child.status != "cancelled", \
            "a refused scancel must not stamp terminal cancelled"
        assert f"(Slurm {child.executor_job_id})" in parent.error
    pending = (parent.result or {})["cleanupIncomplete"]["pendingCancel"]
    assert sorted(pending) == sorted(str(c) for c in children)

    # The scheduler recovers: the reconciler's retry confirms the stops and
    # clears the stamp — the parent's terminal state is untouched.
    monkeypatch.delenv("FAKE_SCANCEL_FAIL", raising=False)
    assert jobs.poll_slurm() >= 1
    parent = jobs.get(parent.id)
    assert "cleanupIncomplete" not in (parent.result or {})
    for cid in children:
        assert jobs.get(cid).status == "cancelled"
    assert any("cleanup complete" in line for line in parent.all_logs())
    assert parent.status == "failed"  # the abort outcome is unchanged


# --- finding 2 (2026-07-22): sharded pipelines carry the FINAL evidence ---------

def _pipeline_parent_fixture(tmp_path, monkeypatch, name, stages,
                             package_evidence=True):
    """Shard runs + a reconcilable sharded parent whose merge config declares
    a pipeline verb over ``stages``."""
    root = str(tmp_path)
    prompts = _study_fixture(root, name)
    raw = es.load_raw(name, root)
    raw["pipeline"] = {"stages": stages}
    es.save_raw(raw, root)
    shard_dirs = _run_shards(root, name, prompts, monkeypatch, count=2)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    records_dir = str(tmp_path / "records")
    os.makedirs(records_dir, exist_ok=True)
    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="succeeded", executor="slurm",
        executor_job_id=str(index), result={"runDirectory": directory})
        for index, directory in enumerate(shard_dirs)]
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "recordsDirectory": records_dir,
            "walltime": "00:30:00",
            "shardMerge": {"experiment": name, "verb": "pipeline",
                           "targetRoot": root,
                           "packageEvidence": package_evidence,
                           "bundlePath": str(tmp_path / "bundle.tar.gz"),
                           "dtype": "auto",
                           "submissionDirectory": str(tmp_path / "sub")}})
    return jobs, parent, records_dir


def test_sharded_pipeline_defers_evidence_to_the_continuation(
        tmp_path, monkeypatch, fake_slurm):
    """The parent's evidence bundle must be the FINAL pipeline evidence
    (judge outputs, analysis, ledger) — packaged by the continuation over the
    finished pipeline directory and folded onto the parent — never the
    merged RUN's partial story stored at merge time."""
    from steerlab_server.experiment import bundles as bundles_mod
    packaged = []
    monkeypatch.setattr(
        bundles_mod, "package_evidence",
        lambda run_dir, **kw: packaged.append(run_dir) or {
            "bundlePath": f"{run_dir}/fake.tar.gz", "bundleSha256": "aa"})
    jobs, parent, _records = _pipeline_parent_fixture(
        tmp_path, monkeypatch, "shardpipev", ["run", "analyze"])

    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "running"
    # NOTHING was packaged at merge time, and the parent carries no bundle
    # while the pipeline is still on the cluster.
    assert packaged == []
    assert "evidenceBundle" not in (parent.result or {})

    # The continuation completes carrying the pipeline evidence bundle its
    # bundle-execute child packaged.
    continuation = jobs.get(parent.requested_resources["continuationJob"])
    pipeline_evidence = {"bundlePath": "/runs/pipe/x.evidence-bundle.tar.gz",
                         "bundleSha256": "bb"}
    continuation.status = "succeeded"
    continuation.result = {"runDirectory": "/runs/pipe",
                           "evidenceBundle": pipeline_evidence}
    jobs.store.update(continuation)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "succeeded"
    # The continuation's FINAL bundle is folded onto the parent — the record
    # the app imports from.
    assert (parent.result or {})["evidenceBundle"] == pipeline_evidence
    assert (parent.result or {})["runDirectory"] == "/runs/pipe"


def test_pipeline_success_without_requested_evidence_fails_parent(
        tmp_path, monkeypatch, fake_slurm):
    """Evidence-as-success invariant (finding 4, 2026-07-23): when evidence
    packaging was requested and the continuation succeeded WITHOUT an
    evidence bundle, the parent must NOT be stamped succeeded — the app
    imports the parent's bundle, so a bundle-less success looks imported
    while the outputs stay on the cluster. The parent fails with the
    pipeline directory path and the manual packaging recovery."""
    jobs, parent, _records = _pipeline_parent_fixture(
        tmp_path, monkeypatch, "shardpipenoev", ["run", "analyze"],
        package_evidence=True)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "running"
    pipeline_dir = (parent.result or {}).get("pipelineDirectory")
    assert pipeline_dir, "the pipeline directory is stamped at continuation start"

    continuation = jobs.get(parent.requested_resources["continuationJob"])
    continuation.status = "succeeded"
    continuation.result = {"runDirectory": pipeline_dir}  # NO evidenceBundle
    jobs.store.update(continuation)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "failed", \
        "evidence requested + absent must never be a succeeded parent"
    assert "evidence_missing" in parent.error
    assert pipeline_dir in parent.error
    assert "bundle evidence" in parent.error   # the manual packaging verb
    assert "import" in parent.error
    assert (parent.result or {}).get("evidenceMissing") is True


def test_run_only_sharded_pipeline_packages_the_seeded_pipeline_directory(
        tmp_path, monkeypatch, fake_slurm):
    """No remaining stages: the seeded pipeline directory IS the final
    artifact, so ITS evidence (not the bare merged run's) lands on the
    parent."""
    from steerlab_server.experiment import bundles as bundles_mod
    packaged = []
    monkeypatch.setattr(
        bundles_mod, "package_evidence",
        lambda run_dir, **kw: packaged.append(run_dir) or {
            "bundlePath": f"{run_dir}/fake.tar.gz", "bundleSha256": "cc"})
    jobs, parent, _records = _pipeline_parent_fixture(
        tmp_path, monkeypatch, "shardpiper", ["run"])

    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "succeeded"
    pipeline_dir = (parent.result or {})["runDirectory"]
    assert packaged == [pipeline_dir]  # the seeded pipeline dir, exactly once
    assert ((parent.result or {})["evidenceBundle"]["bundlePath"]
            == f"{pipeline_dir}/fake.tar.gz")


def test_run_verb_sharded_parent_still_packages_after_merge(
        tmp_path, monkeypatch):
    """A plain sharded RUN (no pipeline) keeps today's behavior: evidence is
    packaged over the merged run at merge time."""
    from steerlab_server.experiment import bundles as bundles_mod
    packaged = []
    monkeypatch.setattr(
        bundles_mod, "package_evidence",
        lambda run_dir, **kw: packaged.append(run_dir) or {
            "bundlePath": f"{run_dir}/fake.tar.gz", "bundleSha256": "dd"})
    root = str(tmp_path)
    prompts = _study_fixture(root, "shardrunev")
    shard_dirs = _run_shards(root, "shardrunev", prompts, monkeypatch, count=2)
    jobs = _manager(tmp_path)
    children = [jobs.record_external(
        "study-submit-bundle-shard", status="succeeded", executor="slurm",
        executor_job_id=str(index), result={"runDirectory": directory})
        for index, directory in enumerate(shard_dirs)]
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "shardMerge": {"experiment": "shardrunev", "verb": "run",
                           "targetRoot": root, "packageEvidence": True}})
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "succeeded"
    merged = (parent.result or {})["runDirectory"]
    assert packaged == [merged]
    assert ((parent.result or {})["evidenceBundle"]["bundlePath"]
            == f"{merged}/fake.tar.gz")


# --- shared-artifact comparison (external review round 3, finding 1) ---------

def test_shards_extracting_in_different_seconds_still_merge(tmp_path):
    """The P0. Each shard extracts independently and stamps the CURRENT time
    into its vector sidecar; the merge byte-compared shared artifacts, so
    shards that crossed a second boundary refused with "cross-shard
    nondeterminism". In tests that was a ~1-in-10 flake; on Slurm, where
    shards start minutes apart by design, it would fire every time.

    Forces the clock difference rather than hoping for it — a green run
    without that forcing is not evidence.
    """
    import json as _json

    sidecar = {"modelID": "org/m", "concept": "fear", "layerCount": 2,
               "normsPerLayer": [1.0, 2.0], "stimulusSetHash": "ab" * 32,
               "extractionDate": "2026-07-24T12:00:00Z"}
    later = {**sidecar, "extractionDate": "2026-07-24T12:41:07Z"}
    assert sharding._shared_artifacts_agree(
        _json.dumps(sidecar).encode(), _json.dumps(later).encode())


def test_a_genuine_vector_difference_still_refuses(tmp_path):
    # The refusal's INTENT is right: cross-shard nondeterminism in derived
    # vectors is a real scientific problem. Only the wall-clock reading was
    # wrong, so every field that bears on the science still has to match.
    import json as _json

    base = {"modelID": "org/m", "concept": "fear", "normsPerLayer": [1.0, 2.0],
            "extractionDate": "2026-07-24T12:00:00Z"}
    for field, value in (("normsPerLayer", [1.0, 2.5]),
                         ("stimulusSetHash", "ff" * 32),
                         ("recipeHash", "changed"),
                         ("residualNormSource", "extraction-stimuli"),
                         ("modelID", "org/other")):
        divergent = {**base, field: value}
        assert not sharding._shared_artifacts_agree(
            _json.dumps(base).encode(), _json.dumps(divergent).encode()), field


def test_non_json_artifacts_keep_strict_byte_equality():
    # The weights are what the check exists to protect: no canonicalisation,
    # no field exclusions, just bytes.
    assert sharding._shared_artifacts_agree(b"\x00weights", b"\x00weights")
    assert not sharding._shared_artifacts_agree(b"\x00weights", b"\x00weightt")
    # A JSON *array* is not a sidecar either — byte equality only.
    assert not sharding._shared_artifacts_agree(b'[1,2]', b'[1,3]')


# --- the exception is scoped to sidecars (review round 4, finding 3) ---------

def _sidecar(**overrides):
    base = {"modelID": "org/m", "concept": "fear", "layerCount": 2,
            "normsPerLayer": [1.0, 2.0], "stimulusSetHash": "ab" * 32,
            "extractionDate": "2026-07-24T12:00:00Z"}
    base.update(overrides)
    return base


def test_only_vector_sidecars_get_the_volatile_field_exception():
    """Some OTHER JSON artifact that happens to carry an `extractionDate`
    must not inherit an exception nobody reasoned about — a merge that
    tolerates a difference it should have refused is the failure this whole
    check exists to prevent."""
    import json as _json

    not_a_sidecar = {"extractionDate": "2026-07-24T12:00:00Z",
                     "somethingElse": 1}
    later = {**not_a_sidecar, "extractionDate": "2026-07-24T12:41:07Z"}
    assert not sharding._shared_artifacts_agree(
        _json.dumps(not_a_sidecar).encode(), _json.dumps(later).encode())


def test_a_sidecar_is_recognized_by_its_required_fields_not_its_name():
    assert sharding._is_vector_sidecar(_sidecar())
    # Drop any one signature field and it is no longer recognized.
    for field in sharding._SIDECAR_SIGNATURE:
        partial = {k: v for k, v in _sidecar().items() if k != field}
        assert not sharding._is_vector_sidecar(partial), field
    # Optional sidecar fields do not affect recognition.
    assert sharding._is_vector_sidecar(_sidecar(residualNormSource="neutral"))


def test_the_allowlist_stays_minimal():
    """The exception rests on being a MINIMAL allowlist of fields known not
    to bear on the science. Widening it is a deliberate act, not a drift."""
    assert sharding._VOLATILE_SIDECAR_FIELDS == frozenset({"extractionDate"})


def test_shard_local_status_files_are_never_cross_compared():
    """A shard's own failure/progress record is timestamped and shard-local
    by nature. Nothing writes it on the success path a merge consumes, so
    this is not a bug being fixed — it is the class of file that must never
    be cross-shard compared, listed before it can become one."""
    assert run_status.STATUS_FILENAME in sharding._PER_SHARD_FILES
    assert run_status.FAILURE_NOTE_FILENAME in sharding._PER_SHARD_FILES


def test_end_to_end_merge_survives_shards_extracting_minutes_apart(
        tmp_path, monkeypatch):
    """The production case, through the REAL merge.

    The unit tests above pin `_shared_artifacts_agree`; this drives
    `merge_shard_runs` over shard directories whose vector sidecars differ
    ONLY in `extractionDate` — the situation Slurm produces every time,
    since shards start minutes apart. Before the fix this raised
    ShardMergeError; the intermittency in the rest of this file was only
    ever whether the shards happened to straddle a second boundary.
    """
    import json as _json

    root = str(tmp_path)
    prompts = _study_fixture(root, "shardclock")
    shard_dirs = _run_shards(root, "shardclock", prompts, monkeypatch, count=2)

    # Force the divergence rather than hoping for it: rewrite shard 1's
    # sidecars with a timestamp 41 minutes later, leaving every other field
    # — and the .safetensors weights — untouched.
    rewritten = 0
    for entry in sorted(os.listdir(shard_dirs[1])):
        if not entry.endswith(".json") or entry in sharding._PER_SHARD_FILES:
            continue
        path = os.path.join(shard_dirs[1], entry)
        with open(path, encoding="utf-8") as handle:
            payload = _json.load(handle)
        if not isinstance(payload, dict) or "extractionDate" not in payload:
            continue
        payload["extractionDate"] = "2026-07-24T13:41:07Z"
        with open(path, "w", encoding="utf-8") as handle:
            _json.dump(payload, handle, indent=2, sort_keys=True)
        rewritten += 1
    assert rewritten, "fixture produced no vector sidecar to diverge"

    merged = sharding.merge_shard_runs("shardclock", shard_dirs, root=root)
    assert os.path.isfile(os.path.join(merged, "generations.jsonl"))
    # The merged run keeps shard 0's sidecar — one recorded extraction time,
    # not a fabricated one.
    with open(os.path.join(merged, "fear.json"), encoding="utf-8") as handle:
        assert _json.load(handle)["extractionDate"] != "2026-07-24T13:41:07Z"


def test_end_to_end_merge_still_refuses_genuinely_divergent_vectors(
        tmp_path, monkeypatch):
    # The property the refusal exists to protect, through the real merge:
    # a sidecar field that BEARS on the science still stops the merge.
    import json as _json

    root = str(tmp_path)
    prompts = _study_fixture(root, "sharddiverge")
    shard_dirs = _run_shards(root, "sharddiverge", prompts, monkeypatch, count=2)

    path = os.path.join(shard_dirs[1], "fear.json")
    with open(path, encoding="utf-8") as handle:
        payload = _json.load(handle)
    payload["normsPerLayer"] = [9.99 for _ in payload.get("normsPerLayer", [1])]
    with open(path, "w", encoding="utf-8") as handle:
        _json.dump(payload, handle, indent=2, sort_keys=True)

    with pytest.raises(sharding.ShardMergeError, match="cross-shard"):
        sharding.merge_shard_runs("sharddiverge", shard_dirs, root=root)


def test_substrate_divergence_names_the_differing_fields(tmp_path):
    """Wording fix 2026-07-27: a merge refused because shard 1 ran on a
    different node should SAY so — gpu/platform named with both values —
    not leave the researcher diffing substrate.json to learn whether the
    nondeterminism is in the science or in the scheduler's node placement."""
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    merged = tmp_path / "merged"
    for directory in (shard0, shard1, merged):
        directory.mkdir()
    (shard0 / "substrate.json").write_text(json.dumps(
        {"engine": "python-hf-transformers", "gpu": "NVIDIA A100-SXM4-80GB",
         "platform": "Linux-5.14"}), encoding="utf-8")
    (shard1 / "substrate.json").write_text(json.dumps(
        {"engine": "python-hf-transformers", "gpu": "NVIDIA H100 80GB",
         "platform": "Linux-6.2"}), encoding="utf-8")

    with pytest.raises(sharding.ShardMergeError) as excinfo:
        sharding._copy_invariant_artifacts(
            [{"_dir": str(shard0), "shardIndex": 0},
             {"_dir": str(shard1), "shardIndex": 1}], str(merged))
    message = str(excinfo.value)
    assert "substrate.json" in message
    assert "gpu" in message
    assert "A100" in message and "H100" in message
    assert "platform" in message
    # The refusal itself is unchanged — only the account improved.
    assert "cross-shard nondeterminism" in message
    # The field that AGREES is not named.
    assert "engine" not in message


def test_non_json_divergence_keeps_the_generic_refusal(tmp_path):
    shard0 = tmp_path / "s0"
    shard1 = tmp_path / "s1"
    merged = tmp_path / "m"
    for directory in (shard0, shard1, merged):
        directory.mkdir()
    (shard0 / "fear.safetensors").write_bytes(b"\x00weights")
    (shard1 / "fear.safetensors").write_bytes(b"\x00weightt")
    with pytest.raises(sharding.ShardMergeError) as excinfo:
        sharding._copy_invariant_artifacts(
            [{"_dir": str(shard0), "shardIndex": 0},
             {"_dir": str(shard1), "shardIndex": 1}], str(merged))
    message = str(excinfo.value)
    assert "fear.safetensors" in message
    # No field detail for binary artifacts — the generic refusal, verbatim.
    assert "differs between shard 0 and shard 1 — cross-shard" in message
