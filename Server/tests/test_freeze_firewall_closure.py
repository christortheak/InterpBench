"""Phase E freeze-firewall closure: variant validity gate, external-input
pinning at freeze, thinking-mode gate, run-time prompt drift checks, bundle
manifest-path guard, multi-agent revision threading."""

import hashlib
import json
import os
from contextlib import contextmanager

import pytest

from steerlab_server.experiment import bundles, experiment_store as es, multi_agent, tasks
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload, indent=1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _variant_artifact(root, *, adapter_hash="ab" * 32, system_prompt=None,
                      system_prompt_hash=None):
    artifact = {
        "name": "fear-lora", "baseModelID": "org/m",
        "adapters": [{"name": "a1", "artifactPath": "x", "adapterDirectory": "y",
                      "adapterHash": adapter_hash}],
        "injections": [],
        "temperature": 0.0,
    }
    if system_prompt is not None:
        artifact["systemPrompt"] = system_prompt
    if system_prompt_hash is not None:
        artifact["systemPromptHash"] = system_prompt_hash
    path = os.path.join(root, "runs", "model-variants", "fear-lora", "model-variant.json")
    digest = _write(path, artifact)
    return "runs/model-variants/fear-lora/model-variant.json", digest, artifact


def _variant_study(root, name="vs", **artifact_kwargs):
    rel, digest, artifact = _variant_artifact(root, **artifact_kwargs)
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["variantConditions"] = [{"name": "fear-lora", "artifactPath": rel,
                               "artifactHash": digest, "artifact": artifact}]
    es.save_raw(d, root)
    return name


def _variant_validate_evidence(root, name, *, conditions=("baseline", "fear-lora"),
                               battery_hash="bb" * 32, stamp="v"):
    """Scope-matched validate evidence WITH per-condition battery results —
    what a real `experiment validate` writes for a variant study."""
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"{stamp}-exp-{name}-validate")
    os.makedirs(rundir)
    evidence = {"schemaVersion": 1, "task": "validate",
                "substrate": "python-hf-transformers",
                "validationScopeHash": scope,
                "batteryResults": [
                    {"condition": c, "batteryHash": battery_hash,
                     "total": 4, "correct": 4, "accuracy": 1.0}
                    for c in conditions]}
    json.dump(evidence,
              open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {}},
              open(os.path.join(rundir, "validation-report.json"), "w"))
    return rundir


def test_variant_freeze_requires_adapter_hash(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root, adapter_hash=None)
    # Strip the None so the adapter simply lacks the key.
    d = es.load_raw(name, root)
    del d["variantConditions"][0]["artifact"]["adapters"][0]["adapterHash"]
    es.save_raw(d, root)
    with pytest.raises(es.ExperimentStoreError, match="adapterHash"):
        es.freeze(name, force=False, root=root)
    # --force skips the evidence gate loudly, like the other gates.
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["status"] == "frozen"


def test_variant_freeze_requires_system_prompt_hash(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root, system_prompt="You are calm.")
    with pytest.raises(es.ExperimentStoreError, match="systemPromptHash"):
        es.freeze(name, force=False, root=root)


def test_variant_freeze_passes_with_full_pins_and_repins_path(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root, system_prompt="You are calm.",
                          system_prompt_hash="cd" * 32)
    _variant_validate_evidence(root, name)
    frozen = es.freeze(name, force=False, root=root)
    assert frozen["status"] == "frozen"
    # The artifact lived under gitignored runs/ — freeze copied it into the
    # experiment directory and repointed the manifest.
    new_path = frozen["variantConditions"][0]["artifactPath"]
    assert new_path.startswith(f"experiments/{name}/pinned/")
    absolute = os.path.join(root, new_path)
    assert os.path.isfile(absolute)
    with open(absolute, "rb") as handle:
        live = hashlib.sha256(handle.read()).hexdigest()
    assert live == frozen["variantConditions"][0]["artifactHash"]


def test_freeze_pins_scenario_out_of_runs(tmp_path):
    root = str(tmp_path)
    scenario = {"name": "panel", "baseModelID": "org/m",
                "agents": [{"id": "j1", "name": "Judge"}],
                "turns": [{"id": "t1", "speakerAgentID": "j1",
                           "promptTemplate": "Vote."}]}
    digest = _write(os.path.join(root, "runs", "multi-agent-scenarios",
                                 "panel", "scenario.json"), scenario)
    es.create("mas", model_id="org/m", revision="abc", root=root)
    d = es.load_raw("mas", root)
    d["studyKind"] = "multiAgent"
    d["multiAgentScenarioPath"] = "runs/multi-agent-scenarios/panel/scenario.json"
    d["multiAgentScenarioHash"] = digest
    es.save_raw(d, root)
    frozen = es.freeze("mas", force=False, root=root)
    assert frozen["multiAgentScenarioPath"] == "experiments/mas/pinned/scenario.json"
    assert os.path.isfile(os.path.join(root, frozen["multiAgentScenarioPath"]))
    # Hash still verifies against the copied file.
    assert Manifest.from_dict(frozen).verify(root) == []


def test_thinking_mode_gate():
    manifest = Manifest.from_dict({
        "name": "t", "modelID": "org/m", "qwenThinkingEnabled": True,
        "outcomeInstruments": ["answerTokenLogprob"],
        "variantConditions": [{"name": "v", "artifactPath": "",
                               "artifactHash": "", "artifact": {}}],
    })
    violations = manifest.verify("/nonexistent-root")
    assert any("thinking" in v for v in violations)


def _prompts_manifest(tmp_path, *, status="draft", pin=True):
    prompts = tmp_path / "items.jsonl"
    prompts.write_text('{"id": "a", "prompt": "p"}\n')
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    d = {"name": "pm", "modelID": "org/m", "status": status,
         "taskPromptsFile": str(prompts)}
    if pin:
        d["taskPromptsHash"] = digest
    return Manifest.from_dict(d), prompts


def test_load_prompts_detects_drift_at_run_time(tmp_path):
    manifest, prompts = _prompts_manifest(tmp_path)
    assert tasks._load_prompts(manifest, None, str(tmp_path))
    prompts.write_text('{"id": "a", "prompt": "EDITED"}\n')
    with pytest.raises(RuntimeError, match="drifted"):
        tasks._load_prompts(manifest, None, str(tmp_path))


def test_load_prompts_frozen_rules(tmp_path):
    manifest, prompts = _prompts_manifest(tmp_path, status="frozen")
    # Identical override is fine; different override is refused.
    same = tmp_path / "same.jsonl"
    same.write_bytes(prompts.read_bytes())
    assert tasks._load_prompts(manifest, str(same), str(tmp_path))
    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "b", "prompt": "different"}\n')
    with pytest.raises(RuntimeError, match="FROZEN"):
        tasks._load_prompts(manifest, str(other), str(tmp_path))
    # Frozen with no pin at all cannot run.
    unpinned, _ = _prompts_manifest(tmp_path, status="frozen", pin=False)
    with pytest.raises(RuntimeError, match="no pinned task prompts"):
        tasks._load_prompts(unpinned, None, str(tmp_path))


def test_bundle_manifest_path_guard(tmp_path):
    target = str(tmp_path)
    experiments = os.path.join(target, "experiments")
    assert bundles._is_manifest_path(
        os.path.join(experiments, "s", "experiment.json"), target)
    assert bundles._is_manifest_path(os.path.join(experiments, "s.json"), target)
    assert not bundles._is_manifest_path(
        os.path.join(target, "runs", "x", "report.json"), target)


def test_run_scenario_threads_default_revision(tmp_path, monkeypatch):
    scenario = multi_agent.Scenario.from_dict({
        "name": "panel", "baseModelID": "org/m",
        # Every seat names its own model: validate() refuses a blank one, so a
        # panel cannot silently inherit whatever model happens to be loaded.
        "agents": [{"id": "j1", "name": "Judge", "baseModelID": "org/m"}],
        "turns": [{"id": "t1", "speakerAgentID": "j1",
                   "promptTemplate": "Vote now."}]})
    seen = {}

    class _FakeModel:
        model_id = "org/m"
        revision = "rev123"

    @contextmanager
    def provider(model_id, revision=None):
        seen["revision"] = revision
        yield _FakeModel()

    monkeypatch.setattr(multi_agent, "generate",
                        lambda *args, **kwargs: "I vote to affirm.")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    multi_agent.run_scenario(None, scenario, run_dir=str(run_dir),
                             model_provider=provider, default_revision="rev123")
    assert seen["revision"] == "rev123"
    with open(run_dir / "turns.jsonl", encoding="utf-8") as handle:
        (turn,) = [json.loads(line) for line in handle if line.strip()]
    assert turn["modelRevision"] == "rev123"


def test_freeze_writes_canonical_bytes(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root, name="canon")
    _variant_validate_evidence(root, name)
    frozen = es.freeze(name, force=False, root=root)
    canonical = os.path.join(root, "experiments", name, "freeze-canonical.json")
    with open(canonical, "rb") as handle:
        blob = handle.read()
    assert hashlib.sha256(blob).hexdigest() == frozen["freezeHash"]
    payload = json.loads(blob)
    for volatile in ("status", "frozenAt", "freezeHash", "gitCommit", "frozenBy"):
        assert volatile not in payload
    assert payload["name"] == name


def _validate_evidence_run(root, name, scope, *, substrate=None,
                           report_file="validation-report.json", stamp=None):
    rundir = os.path.join(root, "runs", f"{stamp or 'x'}-exp-{name}-validate")
    os.makedirs(rundir)
    evidence = {"schemaVersion": 1, "task": "validate",
                "validationScopeHash": scope}
    if substrate is not None:
        evidence["substrate"] = substrate
    if report_file != "validation-report.json":
        evidence["reportFile"] = report_file
    json.dump(evidence, open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"c": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, report_file), "w"))


def test_validate_evidence_is_substrate_scoped(tmp_path):
    root = str(tmp_path)
    d = os.path.join(root, "prompts", "concepts", "french")
    os.makedirs(d)
    open(os.path.join(d, "positive.jsonl"), "w").write('{"text": "bonjour"}\n')
    open(os.path.join(d, "negative.jsonl"), "w").write('{"text": "hello"}\n')
    es.create("sub", model_id="org/m", revision="abc", root=root)
    es.attach("sub", ["french"], root=root)
    scope = Manifest.load("sub", root=root).validation_scope_hash()

    # Foreign-substrate evidence never satisfies this engine's freeze.
    _validate_evidence_run(root, "sub", scope, substrate="swift-mlx", stamp="a")
    with pytest.raises(es.ExperimentStoreError, match="no validate run"):
        es.freeze("sub", force=False, root=root)

    # Own-substrate evidence (with a custom reportFile) does.
    _validate_evidence_run(root, "sub", scope, substrate="python-hf-transformers",
                           report_file="report.json", stamp="b")
    assert es.freeze("sub", force=False, root=root)["status"] == "frozen"


def test_legacy_unstamped_evidence_still_counts(tmp_path):
    root = str(tmp_path)
    d = os.path.join(root, "prompts", "concepts", "french")
    os.makedirs(d)
    open(os.path.join(d, "positive.jsonl"), "w").write('{"text": "bonjour"}\n')
    open(os.path.join(d, "negative.jsonl"), "w").write('{"text": "hello"}\n')
    es.create("leg", model_id="org/m", revision="abc", root=root)
    es.attach("leg", ["french"], root=root)
    scope = Manifest.load("leg", root=root).validation_scope_hash()
    # Pre-substrate evidence was necessarily produced by THIS engine (the
    # filename divergence made cross-engine evidence impossible).
    _validate_evidence_run(root, "leg", scope, substrate=None)
    assert es.freeze("leg", force=False, root=root)["status"] == "frozen"


def test_validate_evidence_matches_by_scope_not_name(tmp_path):
    root = str(tmp_path)
    d = os.path.join(root, "prompts", "concepts", "french")
    os.makedirs(d)
    open(os.path.join(d, "positive.jsonl"), "w").write('{"text": "bonjour"}\n')
    open(os.path.join(d, "negative.jsonl"), "w").write('{"text": "hello"}\n')
    es.create("orig", model_id="org/m", revision="abc", root=root)
    es.attach("orig", ["french"], root=root)
    scope = Manifest.load("orig", root=root).validation_scope_hash()
    # Evidence produced under the ORIGINAL experiment's name…
    _validate_evidence_run(root, "orig", scope,
                           substrate="python-hf-transformers")
    # …satisfies the freeze of a DUPLICATE with identical pins (scope-based,
    # matching Swift).
    dup = es.duplicate("orig", "copy", root=root)
    assert dup["name"] == "copy"
    frozen = es.freeze("copy", force=False, root=root)
    assert frozen["status"] == "frozen"
