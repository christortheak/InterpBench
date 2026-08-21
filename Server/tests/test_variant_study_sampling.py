"""Study-owned sampling for saved agents (2026-07-21).

The defect this pins closed: a study declaring ``temperature: 0.7,
samplesPerItem: N`` gave ordinary/baseline conditions the repeated-sampling
branch while saved-agent (variant) conditions required the agent artifact's
stored temperature to be zero and ran ONE greedy path — an unbalanced design.

The contract now enforced (and advertised as the ``variantStudySampling``
capability):

- the STUDY manifest owns the measured-run sampling policy — effective
  temperature, samplesPerItem, seed policy — for baseline and every saved
  agent uniformly;
- the AGENT artifact owns intervention identity only (base model, injections,
  adapters, prompt mode, system instruction, thinking mode); its stored
  Playground temperature is non-operative provenance, stamped on variant
  records as ``agentPlaygroundTemperature`` (the record's ``temperature``
  remains the single source of what governed generation);
- answer-token/choice instrument records stay temperature-free, one per
  (condition, prompt), regardless of sampling;
- a guard refuses BEFORE generation if any condition would receive a
  different effective sampling policy than the manifest declares.

Acceptance list (design note "Required design change: study-owned sampling
for saved agents"): count parity, temperature parity + provenance,
seed determinism, seed uniqueness (+ the condition-identity derivation
policy), intervention persistence under sampling, adapter parity, instrument
invariance, resume parity, and the sampling-policy guard.
"""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob as logprob_mod
from steerlab_server.experiment import model_variant, resume, tasks
from steerlab_server.experiment.generate import CellInjection
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors

OPTIONS = ["affirm", "reverse"]
TEMPERATURE = 0.7
SAMPLES = 3
PROMPTS = 2
# baseline + two saved agents.
CONDITIONS = ("baseline", "agent-a", "agent-b")


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _artifact(name, *, temperature=0.0, injections=None, adapters=None,
              **extra):
    artifact = {"name": name, "baseModelID": "org/m",
                "promptMode": "chatAssistant", "temperature": temperature,
                "alphaInNormUnits": False,
                "injections": injections or [], "adapters": adapters or []}
    artifact.update(extra)
    return artifact


def _study_fixture(root, name, *, agent_a=None, agent_b=None):
    """2 prompts x (baseline + 2 saved agents) x 3 samples at temperature
    0.7, seedPolicy derivedSHA256, with the answer-token instrument on."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = TEMPERATURE
    raw["samplesPerItem"] = SAMPLES
    raw["seedPolicy"] = "derivedSHA256"
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    raw["variantConditions"] = [
        {"name": "agent-a", "artifactPath": "runs/variants/a.json",
         "artifactHash": "aa" * 32,
         "artifact": agent_a if agent_a is not None else _artifact("agent-a")},
        {"name": "agent-b", "artifactPath": "runs/variants/b.json",
         "artifactHash": "bb" * 32,
         "artifact": agent_b if agent_b is not None else _artifact("agent-b")},
    ]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    lines = []
    for i in range(PROMPTS):
        lines.append(json.dumps({
            "id": f"p{i}", "prompt": f"Decide case {i}.",
            "options": OPTIONS, "target": "reverse"}))
    _write(prompts_path, "\n".join(lines) + "\n")
    # The artifact preflight refuses a run whose variant references do not
    # resolve on disk (the injection/adapter layers are monkeypatched in
    # these tests, but the preflight is deliberately not): back the fixture
    # references with dummies.
    _write(os.path.join(root, "runs", "x", "fear.json"), "{}")
    _write(os.path.join(root, "runs", "x", "fear.safetensors"), "")
    os.makedirs(os.path.join(root, "runs", "x", "adapter"), exist_ok=True)
    return prompts_path


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _fake_generate(log=None, counter=None, arm_flag_at=None, flag=None):
    """Deterministic on (prompt, temperature, injections) — the envelope
    (seed, sampleIndex, temperature stamps) is what these tests measure."""
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False):
        if counter is not None:
            counter[0] += 1
            if arm_flag_at is not None and counter[0] == arm_flag_at:
                flag.request()
        if log is not None:
            log.append({"kind": "generate", "prompt": prompt,
                        "temperature": temperature,
                        "injections": list(injections or [])})
        choice = "reverse" if injections else "affirm"
        return f"I choose {choice} in {prompt} (T={temperature})"
    return generate


def _fake_score_options(log=None):
    def score_options(model, prompt, options, *, model_id=None, injections=None,
                      prompt_mode=None, system_prompt=None,
                      qwen_thinking_enabled=False):
        if log is not None:
            log.append({"kind": "instrument", "prompt": prompt,
                        "injections": list(injections or [])})
        preferred = "reverse" if injections else "affirm"
        scores = [logprob_mod.OptionScore(
            option=o, token_ids=[7],
            token_logprobs=[-0.5 if o == preferred else -2.0])
            for o in options]
        return logprob_mod.ChoiceResult(options=scores, prompt_token_count=5,
                                        prompt_text=prompt)
    return score_options


def _patch(monkeypatch, generate_fn, score_fn):
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", generate_fn)
    monkeypatch.setattr(logprob_mod, "score_options", score_fn)


def _records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def _sampled(records, condition=None):
    mine = [r for r in records if "instrument" not in r and "error" not in r]
    if condition is not None:
        mine = [r for r in mine if r.get("condition") == condition]
    return mine


def _instruments(records, condition=None):
    mine = [r for r in records if "instrument" in r]
    if condition is not None:
        mine = [r for r in mine if r.get("condition") == condition]
    return mine


# --- acceptance 1: count parity ------------------------------------------------

def test_count_parity_baseline_and_agents_sample_identically(tmp_path,
                                                             monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "vss-count")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("vss-count", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    sampled = _sampled(records)
    # 2 prompts x (baseline + 2 agents) x 3 samples = 18, exactly.
    assert len(sampled) == PROMPTS * len(CONDITIONS) * SAMPLES == 18
    for condition in CONDITIONS:
        for prompt_id in ("p0", "p1"):
            cell = [r for r in _sampled(records, condition)
                    if r["promptID"] == prompt_id]
            assert len(cell) == SAMPLES, (condition, prompt_id)
            assert sorted(r["sampleIndex"] for r in cell) == list(range(SAMPLES))


# --- acceptance 2: temperature parity + playground provenance ------------------

def test_every_sampled_record_stamps_the_manifest_temperature(tmp_path,
                                                              monkeypatch):
    root = str(tmp_path)
    # agent-a stores a NON-ZERO Playground temperature — previously refused,
    # now non-operative provenance.
    prompts = _study_fixture(root, "vss-temp",
                             agent_a=_artifact("agent-a", temperature=0.9))
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("vss-temp", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    for record in _sampled(records):
        assert record["temperature"] == TEMPERATURE
        assert record["doSample"] is True
        assert record["seedPolicy"] == "derivedSHA256"
    # Provenance: variant records carry the artifact temperature under
    # agentPlaygroundTemperature; baseline records carry no such key. There
    # is never a second field named "temperature".
    for record in _sampled(records, "agent-a"):
        assert record["agentPlaygroundTemperature"] == 0.9
    for record in _sampled(records, "agent-b"):
        assert record["agentPlaygroundTemperature"] == 0.0
    for record in _sampled(records, "baseline"):
        assert "agentPlaygroundTemperature" not in record


def test_artifact_temperature_is_non_operative_in_resolution():
    """Unit form of 'changing an artifact's stored temperature changes
    nothing measured': two variants differing ONLY in stored temperature
    resolve to the same effective condition except the provenance stamp."""
    manifest = Manifest.from_dict({
        "name": "s", "modelID": "org/m", "concepts": [], "conditions": [],
        "temperature": TEMPERATURE, "samplesPerItem": SAMPLES,
        "seedPolicy": "derivedSHA256"})
    model = SimpleNamespace(model_id="org/m", revision="abc")

    def resolve(stored_temperature):
        vc = SimpleNamespace(
            name="agent-a", artifact_path="runs/variants/a.json",
            artifact_hash="aa" * 32,
            artifact=_artifact("agent-a", temperature=stored_temperature))
        return tasks._effective_variant_condition(vc, manifest, model, None,
                                                  wants_choice=False)

    cold, hot = resolve(0.0), resolve(0.9)
    assert cold.temperature == hot.temperature == TEMPERATURE
    assert cold.provenance["agentPlaygroundTemperature"] == 0.0
    assert hot.provenance["agentPlaygroundTemperature"] == 0.9
    assert {k: v for k, v in cold.provenance.items()
            if k != "agentPlaygroundTemperature"} \
        == {k: v for k, v in hot.provenance.items()
            if k != "agentPlaygroundTemperature"}
    assert cold.injections == hot.injections
    assert cold.prompt_mode == hot.prompt_mode


# --- acceptance 3: seed determinism --------------------------------------------

def test_rerun_yields_byte_identical_records_per_cell(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "vss-seed")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    first = tasks.run("vss-seed", prompts, root, model_provider=_fake_model,
                      log=lambda *_: None)
    second = tasks.run("vss-seed", prompts, root, model_provider=_fake_model,
                       log=lambda *_: None)
    assert first != second  # two immutable run directories

    def by_cell(run_dir):
        return {(r["condition"], r["promptID"], r["sampleIndex"]):
                json.dumps(r, sort_keys=True)
                for r in _sampled(_records(run_dir))}

    cells_a, cells_b = by_cell(first), by_cell(second)
    assert set(cells_a) == set(cells_b) and len(cells_a) == 18
    assert cells_a == cells_b  # byte-identical per (condition, prompt, sampleIndex)


# --- acceptance 4: seed uniqueness + the derivation policy ---------------------

def test_seeds_are_distinct_within_cells_and_derive_from_condition_identity(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "vss-uniq")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("vss-uniq", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _sampled(_records(run_dir))
    seeds = {}
    for r in records:
        seeds[(r["condition"], r["promptID"], r["sampleIndex"])] = r["seed"]
    # Distinct seeds across sampleIndex within every cell.
    for condition in CONDITIONS:
        for prompt_id in ("p0", "p1"):
            cell = [seeds[(condition, prompt_id, i)] for i in range(SAMPLES)]
            assert len(set(cell)) == SAMPLES
    # POLICY (asserted, see derive_seed docstring): the derivation includes
    # condition identity, so baseline and agents draw DISTINCT streams for
    # the same (prompt, sampleIndex) — paired at the prompt level, not a
    # common-random-numbers design.
    for prompt_id in ("p0", "p1"):
        for i in range(SAMPLES):
            across = {seeds[(c, prompt_id, i)] for c in CONDITIONS}
            assert len(across) == len(CONDITIONS)
    # And the stamped seed is exactly the documented derivation, from the
    # run's own stamped experiment hash.
    experiment_hash = records[0]["experimentHash"]
    for (condition, prompt_id, index), seed in seeds.items():
        assert seed == tasks.derive_seed(experiment_hash, condition,
                                         prompt_id, index)


# --- acceptance 5: intervention persistence under sampling ---------------------

def test_variant_injections_reach_every_sampled_generation(tmp_path,
                                                           monkeypatch):
    root = str(tmp_path)
    steered = _artifact(
        "agent-a",
        injections=[{"concept": "fear", "vectorArtifactID": "runs/x/fear",
                     "layer": 1, "alpha": 2.0}])
    prompts = _study_fixture(root, "vss-hooks", agent_a=steered)
    calls = []
    _patch(monkeypatch, _fake_generate(log=calls), _fake_score_options(log=calls))
    cells = [CellInjection(layer=1, vector=[1.0, 0.0], alpha=2.0)]
    real_injections = model_variant.variant_injections
    monkeypatch.setattr(
        model_variant, "variant_injections",
        lambda variant: (list(cells) if variant.injections
                         else real_injections(variant)))
    run_dir = tasks.run("vss-hooks", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    # The injection configuration is armed for EVERY sampled draw of the
    # steered agent — the hook fires per forward pass inside generate (the
    # unit contract in test_injection_fires_per_token.py, which covers
    # prefill + every decode step regardless of how the next token is
    # selected); here we pin that every one of the 6 sampled generations of
    # the steered condition was armed.
    steered_calls = [c for c in calls if c["kind"] == "generate"
                     and c["injections"]]
    assert len(steered_calls) == PROMPTS * SAMPLES == 6
    assert all(c["temperature"] == TEMPERATURE for c in steered_calls)
    # And the records agree the intervention moved the outcome on every draw.
    for record in _sampled(_records(run_dir), "agent-a"):
        assert record["parsedChoice"] == "reverse"
        assert record["interventionState"]["slots"] == [
            {"concept": "fear", "layer": 1, "alpha": 2.0}]


# --- acceptance 6: adapter parity ----------------------------------------------

def test_adapter_backed_variant_gets_the_same_sampling_policy(tmp_path,
                                                              monkeypatch):
    root = str(tmp_path)
    adapter_backed = _artifact(
        "agent-b",
        adapters=[{"adapterDirectory": "runs/x/adapter",
                   "adapterHash": "cc" * 32}])
    prompts = _study_fixture(root, "vss-adapter", agent_b=adapter_backed)
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    adapter_events = []
    adapter_roots = []
    monkeypatch.setattr(
        model_variant, "apply_adapter",
        lambda model, variant, root=None: (
            "adapter-h" if variant.adapters else None,
            adapter_events.append(bool(variant.adapters)),
            adapter_roots.append(root))[0])
    monkeypatch.setattr(model_variant, "remove_adapter",
                        lambda model, handle: adapter_events.append("removed"))
    run_dir = tasks.run("vss-adapter", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)
    records = _records(run_dir)
    vector_only = _sampled(records, "agent-a")
    with_adapter = _sampled(records, "agent-b")
    assert len(vector_only) == len(with_adapter) == PROMPTS * SAMPLES
    for record in vector_only + with_adapter:
        assert record["temperature"] == TEMPERATURE
        assert record["seedPolicy"] == "derivedSHA256"
    assert sorted(r["sampleIndex"] for r in with_adapter
                  if r["promptID"] == "p0") == list(range(SAMPLES))
    assert True in adapter_events and "removed" in adapter_events
    # The adapter must load from the SAME workspace it was verified in. Passing
    # no root resolves through STEERLAB_ROOT/cwd, so a study could verify
    # adapter A here and load adapter B from the process-default workspace
    # under the same relative path (external review round 11).
    assert adapter_roots and all(r == root for r in adapter_roots), adapter_roots


# --- acceptance 7: instrument invariance ---------------------------------------

def test_choice_records_stay_single_and_temperature_free_under_sampling(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "vss-choice")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("vss-choice", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    for condition in CONDITIONS:
        instruments = _instruments(records, condition)
        # Exactly one per (condition, prompt) — sampling does not multiply.
        assert len(instruments) == PROMPTS
        assert sorted(r["promptID"] for r in instruments) == ["p0", "p1"]
        for record in instruments:
            assert record["temperature"] == 0.0
            assert record["doSample"] is False
            assert "seed" not in record
            assert "sampleIndex" not in record


# --- acceptance 8: resume parity -----------------------------------------------

def test_checkpoint_mid_sampled_agent_resumes_byte_identical(tmp_path,
                                                             monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "vss-resume")

    # Control: an uninterrupted run.
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    control_dir = tasks.run("vss-resume", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None)
    control_bytes = open(os.path.join(control_dir, "generations.jsonl"),
                         "rb").read()
    control_report = open(os.path.join(control_dir, "report.json"), "rb").read()

    # Interrupted mid-agent-a: baseline makes 6 sampled generations; arm the
    # flag on generation 9 (agent-a's third sampled draw).
    flag = resume.CheckpointFlag()
    counter = [0]
    _patch(monkeypatch,
           _fake_generate(counter=counter, arm_flag_at=9, flag=flag),
           _fake_score_options())
    seen = {}
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("vss-resume", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    run_dir = seen["dir"]
    assert resume.is_resumable(run_dir)
    interrupted = _records(run_dir)
    assert any(r.get("condition") == "agent-a" for r in interrupted)

    # Resume: only the missing records generate; byte-identical union.
    resumed_counter = [0]
    _patch(monkeypatch, _fake_generate(counter=resumed_counter),
           _fake_score_options())
    resumed_dir = tasks.run("vss-resume", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None,
                            run_directory=run_dir)
    assert resumed_dir == run_dir
    assert counter[0] == 9
    assert counter[0] + resumed_counter[0] == PROMPTS * len(CONDITIONS) * SAMPLES
    assert open(os.path.join(run_dir, "generations.jsonl"),
                "rb").read() == control_bytes
    assert open(os.path.join(run_dir, "report.json"),
                "rb").read() == control_report
    assert not resume.is_resumable(run_dir)


# --- the sampling-policy guard --------------------------------------------------

def test_divergent_effective_policy_refuses_before_any_generation(
        tmp_path, monkeypatch):
    """Defense in depth: if condition resolution ever hands a condition a
    temperature other than the manifest's, the run refuses BEFORE generating
    anything, in plain language."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "vss-guard")
    counter = [0]
    _patch(monkeypatch, _fake_generate(counter=counter), _fake_score_options())
    real_resolver = tasks._effective_variant_condition

    def regressed(vc, manifest, model, r, *, wants_choice):
        eff = real_resolver(vc, manifest, model, r, wants_choice=wants_choice)
        eff.temperature = 0.0  # the historical bug, reintroduced
        return eff

    monkeypatch.setattr(tasks, "_effective_variant_condition", regressed)
    with pytest.raises(RuntimeError, match="unbalanced design"):
        tasks.run("vss-guard", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)
    assert counter[0] == 0  # refused before any model work


def test_unresolvable_variant_does_not_mask_the_guard(tmp_path, monkeypatch):
    """A variant that fails to resolve is the condition loop's error record,
    not a guard crash — baseline still measures."""
    root = str(tmp_path)
    prompts = _study_fixture(
        root, "vss-badvariant",
        agent_a=_artifact("agent-a", baseModelID="other/m"))
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("vss-badvariant", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)
    records = _records(run_dir)
    errors = [r for r in records if "error" in r]
    assert len(errors) == 1 and errors[0]["condition"] == "agent-a"
    assert len(_sampled(records, "baseline")) == PROMPTS * SAMPLES
    assert len(_sampled(records, "agent-b")) == PROMPTS * SAMPLES


# --- the capability token -------------------------------------------------------

def test_capabilities_advertise_variant_study_sampling():
    from steerlab_server.api.profile import capability_snapshot
    snapshot = capability_snapshot()
    assert snapshot["remoteStudy"]["variantStudySampling"] is True


# --- artifact preflight: fail in seconds, not after the model load --------------

def test_artifact_preflight_refuses_dangling_variant_reference(tmp_path,
                                                               monkeypatch):
    """A variant condition whose vector reference resolves nowhere (another
    machine's absolute path, artifact NOT in this workspace) refuses at run
    start, before any model is acquired — every dangling reference named."""
    root = str(tmp_path)
    broken = _artifact(
        "agent-a",
        injections=[{"concept": "fear",
                     "vectorArtifactID": "/Users/nobody/Workspace/runs/gone/fear",
                     "layer": 1, "alpha": 2.0}])
    prompts = _study_fixture(root, "vss-preflight", agent_a=broken)
    acquired = []
    monkeypatch.setattr(tasks, "_acquire_model",
                        lambda *a, **k: acquired.append(True) or (_ for _ in ()).throw(
                            AssertionError("model must not be acquired")))
    with pytest.raises(RuntimeError, match="artifact preflight"):
        tasks.run("vss-preflight", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)
    assert acquired == []
