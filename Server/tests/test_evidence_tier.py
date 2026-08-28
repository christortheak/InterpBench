"""Pre-freeze evidence tier: judge-rubric versioning (pinned rubric file,
judge panel, agreement stats, human-validation subset) and the capability
battery as validate evidence consumed by freeze's variant gate."""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import battery as battery_mod
from steerlab_server.experiment import experiment_store as es, tasks
from steerlab_server.experiment.manifest import Manifest


RUBRIC_TEXT = "Prefer the response that follows the sentencing guideline.\n"


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _variant_study(root, name="vs"):
    artifact = {"name": "fear-lora", "baseModelID": "org/m",
                "adapters": [], "injections": [], "temperature": 0.0,
                "promptMode": "chatAssistant"}
    rel = "runs/model-variants/fear-lora/model-variant.json"
    digest = _write(os.path.join(root, rel), artifact)
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["variantConditions"] = [{"name": "fear-lora", "artifactPath": rel,
                               "artifactHash": digest, "artifact": artifact}]
    es.save_raw(d, root)
    return name


def _validate_evidence(root, name, *, battery_results=None, stamp="v"):
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"{stamp}-exp-{name}-validate")
    os.makedirs(rundir)
    evidence = {"schemaVersion": 1, "task": "validate",
                "substrate": "python-hf-transformers",
                "validationScopeHash": scope}
    if battery_results is not None:
        evidence["batteryResults"] = battery_results
    json.dump(evidence, open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {}},
              open(os.path.join(rundir, "validation-report.json"), "w"))


def _battery_rows(battery_hash, conditions=("baseline", "fear-lora")):
    return [{"condition": c, "batteryHash": battery_hash,
             "total": 4, "correct": 3, "accuracy": 0.75} for c in conditions]


# --- manifest fields + verify ------------------------------------------------

def test_manifest_parses_judge_and_battery_fields():
    manifest = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "judgeRubricFile": "prompts/rubrics/default-paired-v1.md",
        "judgeRubricHash": "ab" * 32,
        "judges": [{"name": "opus", "kind": "claude", "model": "claude-opus-4-8"},
                   {"name": "local", "kind": "local", "model": "Qwen/Qwen3-4B"}],
        "humanValidation": {"path": "prompts/rubrics/human.jsonl", "hash": "cd" * 32},
        "capabilityBatteryFile": "prompts/batteries/basic.jsonl",
        "capabilityBatteryHash": "ef" * 32,
    })
    assert manifest.judge_rubric_file == "prompts/rubrics/default-paired-v1.md"
    assert [j.name for j in manifest.judges] == ["opus", "local"]
    assert manifest.judges[1].kind == "local"
    assert manifest.human_validation.hash == "cd" * 32
    assert manifest.capability_battery_file == "prompts/batteries/basic.jsonl"


def test_verify_flags_rubric_drift_and_half_pins(tmp_path):
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"), RUBRIC_TEXT)
    base = {"name": "s", "modelID": "org/m",
            "variantConditions": [{"name": "v", "artifactPath": "",
                                   "artifactHash": "x", "artifact": {}}]}

    ok = Manifest.from_dict({**base, "judgeRubricFile": "prompts/rubrics/r.md",
                             "judgeRubricHash": rubric_hash})
    assert not [v for v in ok.verify(root) if "rubric" in v]

    drifted = Manifest.from_dict({**base, "judgeRubricFile": "prompts/rubrics/r.md",
                                  "judgeRubricHash": "00" * 32})
    assert any("judge rubric" in v and "changed since pinning" in v
               for v in drifted.verify(root))

    half = Manifest.from_dict({**base, "judgeRubricFile": "prompts/rubrics/r.md"})
    assert any("incomplete" in v for v in half.verify(root))


def test_verify_flags_battery_and_human_validation_drift(tmp_path):
    root = str(tmp_path)
    battery_hash = _write(os.path.join(root, "prompts", "batteries", "basic.jsonl"),
                          '{"prompt": "2+2?", "answer": "4"}\n')
    human_hash = _write(os.path.join(root, "prompts", "validation", "human.jsonl"),
                        '{"promptID": "p0", "seed": 0, "condition": "c", '
                        '"outcome": "variant"}\n')
    base = {"name": "s", "modelID": "org/m",
            "variantConditions": [{"name": "v", "artifactPath": "",
                                   "artifactHash": "x", "artifact": {}}]}
    ok = Manifest.from_dict({
        **base, "capabilityBatteryFile": "prompts/batteries/basic.jsonl",
        "capabilityBatteryHash": battery_hash,
        "humanValidation": {"path": "prompts/validation/human.jsonl",
                            "hash": human_hash}})
    assert ok.verify(root) == []
    bad = Manifest.from_dict({
        **base, "capabilityBatteryFile": "prompts/batteries/basic.jsonl",
        "capabilityBatteryHash": "00" * 32,
        "humanValidation": {"path": "prompts/validation/human.jsonl",
                            "hash": "11" * 32}})
    violations = bad.verify(root)
    assert any("capability battery" in v for v in violations)
    assert any("human validation" in v for v in violations)


def test_verify_flags_malformed_but_hash_matching_battery_and_baseline(tmp_path):
    """Finding 3 (2026-07-19, Swift PinShapeValidation twin): a
    present-but-malformed pinned battery or human baseline whose hash
    MATCHES used to pass verify and die later — battery at task load,
    baseline at analyze. Shape fires only on a hash match; drift and
    missing files keep their own messages."""
    root = str(tmp_path)
    bad_battery_hash = _write(os.path.join(root, "prompts", "batteries", "mal.jsonl"),
                              "not a battery item\n")
    bad_baseline_hash = _write(os.path.join(root, "prompts", "baselines", "mal.csv"),
                               "caseID,delta\nA,0.12\n")
    m = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "capabilityBatteryFile": "prompts/batteries/mal.jsonl",
        "capabilityBatteryHash": bad_battery_hash,
        "humanBaseline": {"path": "prompts/baselines/mal.csv",
                          "hash": bad_baseline_hash}})
    violations = m.verify(root)
    # The battery violation names the line the tasks loader would refuse.
    assert any("prompts/batteries/mal.jsonl" in v and "line 1" in v
               for v in violations)
    # The baseline violation names both column lists.
    assert any("endpoint, deltaHuman, ciLower, ciUpper" in v
               and "caseID, delta" in v for v in violations)
    # Shape only — no spurious drift report rides along.
    assert not any("changed since pinning" in v for v in violations)

    # A malformed ROW behind a valid header is named row-and-field.
    bad_row_hash = _write(os.path.join(root, "prompts", "baselines", "row.csv"),
                          "endpoint,deltaHuman,ciLower,ciUpper\nrate,oops,0.1,0.2\n")
    rows = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "humanBaseline": {"path": "prompts/baselines/row.csv",
                          "hash": bad_row_hash}}).verify(root)
    assert any("data row 1" in v and "deltaHuman" in v for v in rows)

    # An EMPTY battery refuses too (0-of-0 scoring is an accident).
    empty_hash = _write(os.path.join(root, "prompts", "batteries", "empty.jsonl"),
                        "\n")
    empty = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "capabilityBatteryFile": "prompts/batteries/empty.jsonl",
        "capabilityBatteryHash": empty_hash}).verify(root)
    assert any("no rows" in v for v in empty)


def test_verify_shape_checks_stay_clean_and_defer_to_drift_and_missing(tmp_path):
    root = str(tmp_path)
    battery_hash = _write(os.path.join(root, "prompts", "batteries", "ok.jsonl"),
                          '{"prompt": "2+2?", "answer": "4"}\n')
    baseline_hash = _write(os.path.join(root, "prompts", "baselines", "ok.csv"),
                           "endpoint,deltaHuman,ciLower,ciUpper\n"
                           "rate,0.12,0.05,0.19\n")
    clean = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "capabilityBatteryFile": "prompts/batteries/ok.jsonl",
        "capabilityBatteryHash": battery_hash,
        "humanBaseline": {"path": "prompts/baselines/ok.csv",
                          "hash": baseline_hash}})
    assert not any("battery" in v or "baseline" in v or "analyze step" in v
                   for v in clean.verify(root))

    # DRIFT keeps its message and never double-reports shape, even though
    # the drifted bytes are ALSO malformed.
    _write(os.path.join(root, "prompts", "batteries", "ok.jsonl"), "garbage\n")
    _write(os.path.join(root, "prompts", "baselines", "ok.csv"),
           "bad,header\nx,1\n")
    drifted = clean.verify(root)
    assert any("capability battery" in v and "changed since pinning" in v
               for v in drifted)
    assert any("human baseline" in v and "changed since pinning" in v
               for v in drifted)
    assert not any("analyze step" in v for v in drifted)
    assert not any("not shaped the way" in v for v in drifted)

    # MISSING keeps its message.
    missing = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "capabilityBatteryFile": "prompts/batteries/absent.jsonl",
        "capabilityBatteryHash": battery_hash,
        "humanBaseline": {"path": "prompts/baselines/absent.csv",
                          "hash": baseline_hash}}).verify(root)
    assert any("capability battery" in v and "file missing" in v
               for v in missing)
    assert any("human baseline" in v and "file missing" in v
               for v in missing)


def test_battery_hash_enters_variant_scope_only():
    concept_study = Manifest.from_dict({"name": "a", "modelID": "org/m"})
    variant_study = Manifest.from_dict({
        "name": "a", "modelID": "org/m",
        "variantConditions": [{"name": "v", "artifactPath": "",
                               "artifactHash": "x", "artifact": {}}]})
    pinned = Manifest.from_dict({
        "name": "a", "modelID": "org/m",
        "capabilityBatteryHash": "ab" * 32,
        "variantConditions": [{"name": "v", "artifactPath": "",
                               "artifactHash": "x", "artifact": {}}]})
    assert concept_study.validation_scope_hash() != variant_study.validation_scope_hash()
    assert variant_study.validation_scope_hash() != pinned.validation_scope_hash()


# --- freeze gates --------------------------------------------------------------

def test_freeze_requires_pinned_rubric_and_at_least_one_judge(tmp_path):
    """The panel-size rule after the 2026-08-28 ruling: ONE judge is a legal
    design and freezes cleanly, ZERO is the invalid state the gate refuses,
    and two or more is unchanged. Swift twin:
    ``EvidenceTierTests.freezeRequiresPinnedRubricAndAtLeastOneJudge``."""
    root = str(tmp_path)
    name = _variant_study(root)
    d = es.load_raw(name, root)
    d["evaluation"] = {"kind": "pairedJudge", "judgeModel": "claude-opus-4-8",
                       "judgePrompt": "inline"}
    es.save_raw(d, root)
    with pytest.raises(es.ExperimentStoreError, match="no judge rubric is pinned"):
        es.freeze(name, force=False, root=root)

    # Pinned rubric, judged evaluation, and NO judge — a judged instrument
    # with no judge codes nothing.
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"), RUBRIC_TEXT)
    d = es.load_raw(name, root)
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = rubric_hash
    d["judges"] = []
    es.save_raw(d, root)
    with pytest.raises(es.ExperimentStoreError, match="pins no judge"):
        es.freeze(name, force=False, root=root)

    # A SINGLE judge freezes — and carries the single-coder advisory.
    d = es.load_raw(name, root)
    d["judges"] = [{"name": "opus", "kind": "claude", "model": "claude-opus-4-8"}]
    es.save_raw(d, root)
    _validate_evidence(root, name, battery_results=_battery_rows("bb" * 32))
    solo = es.freeze(name, force=False, root=root)
    assert solo["status"] == "frozen"
    assert not solo.get("freezeForced"), "a legal design must not be forced"
    assert es.SINGLE_JUDGE_PANEL_ADVISORY in es.freeze_advisories(solo)

    # Two judges: unchanged, and the advisory falls silent.
    pair_name = _variant_study(root, name="jg-pair")
    d = es.load_raw(pair_name, root)
    d["evaluation"] = {"kind": "pairedJudge", "judgeModel": "claude-opus-4-8",
                       "judgePrompt": "inline"}
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = rubric_hash
    d["judges"] = [
        {"name": "opus", "kind": "claude", "model": "claude-opus-4-8"},
        {"name": "sonnet", "kind": "claude", "model": "claude-sonnet-4-6"},
    ]
    es.save_raw(d, root)
    _validate_evidence(root, pair_name, battery_results=_battery_rows("bb" * 32))
    pair = es.freeze(pair_name, force=False, root=root)
    assert pair["status"] == "frozen"
    assert es.SINGLE_JUDGE_PANEL_ADVISORY not in es.freeze_advisories(pair)


def _judged_study(root, name, judges):
    """A variant study with pinned rubric + the given judge panel and
    matching validate/battery evidence — freeze-ready except for whatever
    the judges themselves gate on."""
    _variant_study(root, name=name)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         RUBRIC_TEXT)
    d = es.load_raw(name, root)
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = rubric_hash
    d["judges"] = judges
    es.save_raw(d, root)
    _validate_evidence(root, name, battery_results=_battery_rows("bb" * 32))
    return name


def test_freeze_refuses_two_blank_local_judges_as_one_judge(tmp_path):
    """Finding 4 (external review 2026-07-22): two blank-model local judges
    both resolve to the study model at temperature 0 — identical
    deterministic judges with guaranteed perfect agreement. Count-only
    panels must not satisfy the gate."""
    root = str(tmp_path)
    name = _judged_study(root, "dupjudges", [
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "local", "model": ""},
    ])
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.freeze(name, force=False, root=root)
    assert str(excinfo.value) == (
        "cannot freeze 'dupjudges': judges 'judge-1' and 'judge-2' both "
        "resolve to the same deterministic judge (the study model at "
        "temperature 0) — they would agree perfectly by construction; use "
        "judges with different models, kinds, or providers")
    # Pre-freeze, the same finding is a draft advisory.
    assert any("agree perfectly by construction" in a
               for a in es.freeze_advisories(es.load_raw(name, root), root))


def test_force_freeze_stamps_judge_validity_for_indistinct_panel(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, "dupforced", [
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "local"},
    ])
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["status"] == "frozen"
    assert frozen["freezeForced"] is True
    assert "judgeValidity" in frozen["forcedGatesSkipped"]


def test_distinct_judge_panels_freeze(tmp_path):
    root = str(tmp_path)
    # Blank local + claude: two distinct resolved identities.
    name = _judged_study(root, "mixpanel", [
        {"name": "judge-1", "kind": "local"},
        {"name": "claude", "kind": "claude"},
    ])
    assert es.freeze(name, force=False, root=root)["status"] == "frozen"
    # Two DIFFERENT local models also pass (the gate is identity, not kind).
    name2 = _judged_study(root, "twolocals", [
        {"name": "judge-1", "kind": "local"},
        # A foreign local judge must pin the bytes that will judge
        # (review round 2, finding 3) — without it, two sessions can
        # load different defaults and both record "none".
        {"name": "judge-2", "kind": "local", "model": "other/model",
         "revision": "cafe01", "dtype": "bfloat16"},
    ])
    assert es.freeze(name2, force=False, root=root)["status"] == "frozen"


def test_judge_identity_resolution_rules():
    """The identity resolver mirrors the cross-engine judge-model rules:
    blank local → study model; blank claude → default judge model; provider
    distinguishes openrouter judges."""
    study = {"name": "x", "modelID": "org/m"}
    # Duplicate claude judges (blank + explicit default) collapse too.
    from steerlab_server.experiment import paired_judge
    dup_claude = {**study, "judges": [
        {"name": "c1", "kind": "claude"},
        {"name": "c2", "kind": "claude", "model": paired_judge.DEFAULT_JUDGE_MODEL},
    ]}
    problem = es.judge_panel_indistinct_problem(dup_claude)
    assert problem is not None and "the claude judge" in problem
    # Three collapsing judges read "all", not "both".
    triple = {**study, "judges": [
        {"name": "a", "kind": "local"}, {"name": "b", "kind": "local"},
        {"name": "c", "kind": "local"},
    ]}
    assert "'a', 'b' and 'c' all resolve" in es.judge_panel_indistinct_problem(triple)
    # Same openrouter slug via different pinned providers is two judges.
    providers = {**study, "judges": [
        {"name": "o1", "kind": "openrouter", "model": "org/slug", "provider": "p1"},
        {"name": "o2", "kind": "openrouter", "model": "org/slug", "provider": "p2"},
    ]}
    assert es.judge_panel_indistinct_problem(providers) is None
    # The routing slug and OpenRouter's response display name identify one
    # serving endpoint, not two independent judges.
    aliases = {**study, "judges": [
        {"name": "o1", "kind": "openrouter",
         "model": "google/gemini-3.6-flash",
         "provider": "google-ai-studio"},
        {"name": "o2", "kind": "openrouter",
         "model": "google/gemini-3.6-flash",
         "provider": "Google AI Studio"},
    ]}
    assert "openrouter judge" in es.judge_panel_indistinct_problem(aliases)
    # Panels of < 2 named judges are the count gate's business, not this one.
    assert es.judge_panel_indistinct_problem(
        {**study, "judges": [{"name": "solo", "kind": "local"}]}) is None


def test_legacy_frozen_manifest_with_indistinct_panel_still_verifies(tmp_path):
    """The distinctness rule is a freeze-time GATE, never retroactive: a
    manifest already frozen with a collapsed panel keeps verifying clean."""
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         RUBRIC_TEXT)
    legacy = Manifest.from_dict({
        "name": "legacy", "modelID": "org/m", "status": "frozen",
        "judgeRubricFile": "prompts/rubrics/r.md",
        "judgeRubricHash": rubric_hash,
        "judges": [{"name": "judge-1", "kind": "local"},
                   {"name": "judge-2", "kind": "local"}],
    })
    assert not [v for v in legacy.verify(root) if "judge" in v.lower()]


def test_freeze_force_skips_judge_gate_loudly(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root, name="forced")
    d = es.load_raw(name, root)
    d["evaluation"] = {"kind": "pairedJudge", "judgeModel": "claude-opus-4-8",
                       "judgePrompt": "inline"}
    es.save_raw(d, root)
    assert es.freeze(name, force=True, root=root)["status"] == "frozen"


def test_variant_freeze_requires_battery_results_naming_conditions(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root)
    # Evidence exists and scope-matches, but has NO battery results.
    _validate_evidence(root, name, battery_results=None)
    with pytest.raises(es.ExperimentStoreError,
                       match="capability-battery results .*baseline, fear-lora"):
        es.freeze(name, force=False, root=root)
    # Partial coverage names only the missing condition.
    _validate_evidence(root, name, stamp="w",
                       battery_results=_battery_rows("bb" * 32, ("baseline",)))
    with pytest.raises(es.ExperimentStoreError,
                       match="capability-battery results .*fear-lora"):
        es.freeze(name, force=False, root=root)


def test_variant_freeze_rejects_battery_drift_against_pin(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root)
    battery_hash = _write(os.path.join(root, "prompts", "batteries", "basic.jsonl"),
                          '{"prompt": "2+2?", "answer": "4"}\n')
    d = es.load_raw(name, root)
    d["capabilityBatteryFile"] = "prompts/batteries/basic.jsonl"
    d["capabilityBatteryHash"] = battery_hash
    es.save_raw(d, root)
    # Evidence produced from a DIFFERENT battery than the pin → gate failure.
    _validate_evidence(root, name, battery_results=_battery_rows("00" * 32))
    with pytest.raises(es.ExperimentStoreError, match="battery drifted"):
        es.freeze(name, force=False, root=root)
    # Matching battery hash freezes.
    _validate_evidence(root, name, stamp="w",
                       battery_results=_battery_rows(battery_hash))
    assert es.freeze(name, force=False, root=root)["status"] == "frozen"


# --- battery execution in validate ------------------------------------------

def _battery_manifest(root, *, pin_hash=None, variant_system_prompt="VARIANT"):
    battery_hash = _write(
        os.path.join(root, "prompts", "batteries", "basic.jsonl"),
        '{"prompt": "2+2?", "answer": "4"}\n'
        '{"prompt": "Capital of France?", "answer": "Paris"}\n')
    d = {"name": "bat", "modelID": "org/m",
         "variantConditions": [{"name": "v1", "artifactPath": "",
                                "artifactHash": "x",
                                "artifact": {"name": "v1", "baseModelID": "org/m",
                                             "adapters": [], "injections": [],
                                             "temperature": 0.0,
                                             "promptMode": "chatAssistant",
                                             "systemPrompt": variant_system_prompt}}]}
    if pin_hash is not None:
        d["capabilityBatteryFile"] = "prompts/batteries/basic.jsonl"
        d["capabilityBatteryHash"] = pin_hash
    return Manifest.from_dict(d), battery_hash


def test_battery_results_scores_baseline_and_variants(tmp_path, monkeypatch):
    root = str(tmp_path)
    manifest, battery_hash = _battery_manifest(root)

    def fake_generate(model, prompt, **kwargs):
        # The variant condition generates under ITS OWN prompt settings; make
        # it fail one probe so accuracies separate.
        if kwargs.get("system_prompt") == "VARIANT":
            return "5" if "2+2" in prompt else "The capital is Paris."
        return "4" if "2+2" in prompt else "Paris"

    monkeypatch.setattr(tasks, "generate", fake_generate)
    results = tasks._battery_results(manifest, object(), root, lambda *a: None)
    by_condition = {r["condition"]: r for r in results}
    assert by_condition["baseline"]["accuracy"] == 1.0
    assert by_condition["v1"]["accuracy"] == 0.5
    assert all(r["batteryHash"] == battery_hash for r in results)
    assert all(r["total"] == 2 for r in results)

    # Battery generations are greedy and capped at the Swift-parity 24 tokens.
    seen = {}

    def spy_generate(model, prompt, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        seen["temperature"] = kwargs.get("temperature")
        return "4"

    monkeypatch.setattr(tasks, "generate", spy_generate)
    tasks._battery_results(manifest, object(), root, lambda *a: None)
    assert seen["max_tokens"] == 24 and seen["temperature"] == 0.0


def test_battery_results_refuse_pinned_hash_drift(tmp_path, monkeypatch):
    root = str(tmp_path)
    manifest, _ = _battery_manifest(root, pin_hash="00" * 32)
    monkeypatch.setattr(tasks, "generate", lambda *a, **k: "4")
    with pytest.raises(RuntimeError, match="drifted from the pinned hash"):
        tasks._battery_results(manifest, object(), root, lambda *a: None)


def test_validate_impl_stamps_battery_results(tmp_path, monkeypatch):
    root = str(tmp_path)
    exp_dir = os.path.join(root, "experiments", "vs")
    manifest, battery_hash = _battery_manifest(root)
    _write(os.path.join(exp_dir, "experiment.json"), manifest.raw or {
        "name": "vs", "modelID": "org/m"})
    monkeypatch.setattr(tasks, "_extract_all", lambda model, manifest, root: {})
    monkeypatch.setattr(
        tasks, "_battery_results",
        lambda manifest, model, root, log: _battery_rows(battery_hash, ("baseline", "v1")))
    run_dir = tasks._validate_impl("vs", manifest, None, root, lambda *a: None)
    evidence = json.load(open(os.path.join(run_dir, "validation-evidence.json")))
    assert {r["condition"] for r in evidence["batteryResults"]} == {"baseline", "v1"}
    assert evidence["validationScopeHash"] == manifest.validation_scope_hash()
    # config.json rides along (canonical per-run stamp; the closed-key
    # schema firewall lives in test_run_config.py).
    from steerlab_server.experiment.run_config import RUN_CONFIG_SCHEMA_VERSION
    config = json.load(open(os.path.join(run_dir, "config.json")))
    assert config["runType"] == "validate"
    assert config["schemaVersion"] == RUN_CONFIG_SCHEMA_VERSION


# --- evaluate: pinned rubric, judge panel, agreement --------------------------

def _evaluate_fixture(tmp_path, *, status="draft", pin_rubric=True,
                      with_human=False):
    root = str(tmp_path)
    d = {"name": "ev", "modelID": "org/m", "status": status,
         "evaluation": {"kind": "pairedJudge", "judgeModel": "claude-opus-4-8",
                        "judgePrompt": "inline rubric"},
         "judges": [{"name": "j-variant", "kind": "claude", "model": "claude-a"},
                    {"name": "j-baseline", "kind": "claude", "model": "claude-b"}]}
    if pin_rubric:
        rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                             RUBRIC_TEXT)
        d["judgeRubricFile"] = "prompts/rubrics/r.md"
        d["judgeRubricHash"] = rubric_hash
    if with_human:
        rows = "\n".join(json.dumps({"promptID": p, "seed": 0,
                                     "condition": "fear", "outcome": "variant"})
                         for p in ("p0", "p1")) + "\n"
        human_hash = _write(os.path.join(root, "prompts", "validation",
                                         "human.jsonl"), rows)
        d["humanValidation"] = {"path": "prompts/validation/human.jsonl",
                                "hash": human_hash}
    _write(os.path.join(root, "experiments", "ev", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-ev-run")
    os.makedirs(run_dir)
    # Epoch stamp: evaluate only accepts source runs produced under the
    # CURRENT manifest content hash.
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline", "output": "calm"},
        {"promptID": "p0", "seed": 0, "condition": "fear", "output": "scared"},
        {"promptID": "p1", "seed": 0, "condition": "baseline", "output": "fine"},
        {"promptID": "p1", "seed": 0, "condition": "fear", "output": "afraid"},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root


VARIANT_TEXTS = {"scared", "afraid"}


def _fake_judge_pair(model, rubric, a, b, structured=None, task_prompt=None):
    assert rubric == RUBRIC_TEXT  # judged from the PINNED FILE, not inline
    variant_slot_a = a in VARIANT_TEXTS
    if model == "claude-a":     # always prefers the steered response
        winner = "A" if variant_slot_a else "B"
    else:                       # always prefers the baseline
        winner = "B" if variant_slot_a else "A"
    return {"winner": winner, "confidence": 0.9}


def test_evaluate_stamps_judges_and_reports_agreement(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    root = _evaluate_fixture(tmp_path, with_human=True)
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge_pair)
    out = tasks.evaluate("ev", root=root)

    judgments = [json.loads(line) for line in
                 open(os.path.join(out, "judgments.jsonl")) if line.strip()]
    assert {j["judge"] for j in judgments} == {"j-variant", "j-baseline"}
    assert all(j["outcome"] == "variant" for j in judgments
               if j["judge"] == "j-variant")
    assert all(j["outcome"] == "baseline" for j in judgments
               if j["judge"] == "j-baseline")

    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["rubricFile"] == "prompts/rubrics/r.md"
    assert [b["name"] for b in report["judges"]] == ["j-variant", "j-baseline"]
    (agreement,) = report["agreement"]
    assert agreement["judges"] == ["j-variant", "j-baseline"]
    assert agreement["n"] == 2
    assert agreement["percentAgreement"] == 0.0
    assert agreement["kappa"] == 0.0  # constant, opposite raters

    human = {h["judge"]: h for h in report["humanAgreement"]}
    assert human["j-variant"]["percentAgreement"] == 1.0
    assert human["j-variant"]["kappa"] == 1.0
    assert human["j-baseline"]["percentAgreement"] == 0.0
    # Canonical per-run stamp for the evaluate run type.
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["runType"] == "evaluate" and config["experiment"] == "ev"


def test_evaluate_refuses_unpinned_rubric_on_frozen_study(tmp_path):
    root = _evaluate_fixture(tmp_path, status="frozen", pin_rubric=False)
    with pytest.raises(RuntimeError, match="hashed rubric FILE"):
        tasks.evaluate("ev", root=root)


def test_evaluate_refuses_rubric_drift(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    root = _evaluate_fixture(tmp_path)
    with open(os.path.join(root, "prompts", "rubrics", "r.md"), "a") as handle:
        handle.write("EDITED\n")
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge_pair)
    with pytest.raises(RuntimeError, match="drifted from the pinned hash"):
        tasks.evaluate("ev", root=root)


def test_evaluate_draft_falls_back_to_inline_rubric_with_warning(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    root = _evaluate_fixture(tmp_path, pin_rubric=False)
    seen_rubrics = []

    def inline_judge(model, rubric, a, b, structured=None, task_prompt=None):
        seen_rubrics.append(rubric)
        return {"winner": "tie", "confidence": 0.1}

    monkeypatch.setattr(paired_judge, "judge_pair", inline_judge)
    warnings = []
    tasks.evaluate("ev", root=root, log=warnings.append)
    assert set(seen_rubrics) == {"inline rubric"}
    assert any("UNPINNED inline rubric" in w for w in warnings)


# --- audit round: battery auto-pin + judge-gate trigger parity ---------------

def _write_default_battery(root):
    import hashlib
    path = os.path.join(root, battery_mod.DEFAULT_BATTERY_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"prompt": "2+2?", "answer": "4"}\n')
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_freeze_autopins_default_battery_into_variant_manifest(tmp_path):
    # A frozen variant study must NAME its battery: freeze pins the default
    # file/hash into the manifest before computing scope/freeze hashes.
    root = str(tmp_path)
    name = _variant_study(root)
    digest = _write_default_battery(root)

    # Evidence fabricated against the POST-pin scope (what a post-fix
    # validate run produces): pin, snapshot the scope, unpin.
    es.set_protocol(name, {"capabilityBatteryFile": battery_mod.DEFAULT_BATTERY_FILE,
                           "capabilityBatteryHash": digest}, root)
    _validate_evidence(root, name, battery_results=_battery_rows(digest))
    d = es.load_raw(name, root)
    d.pop("capabilityBatteryFile", None)
    d.pop("capabilityBatteryHash", None)
    es.save_raw(d, root)

    frozen = es.freeze(name, root=root)
    assert frozen["capabilityBatteryFile"] == battery_mod.DEFAULT_BATTERY_FILE
    assert frozen["capabilityBatteryHash"] == digest
    assert frozen["status"] == "frozen"


def test_validate_autopin_stamps_draft_and_skips_frozen(tmp_path):
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    name = _variant_study(root)
    digest = _write_default_battery(root)

    manifest = Manifest.load(name, root=root)
    pinned = tasks._autopin_capability_battery(name, manifest, root, lambda *_: None)
    assert pinned.capability_battery_hash == digest
    on_disk = es.load_raw(name, root)
    assert on_disk["capabilityBatteryFile"] == battery_mod.DEFAULT_BATTERY_FILE
    assert on_disk["capabilityBatteryHash"] == digest

    # Already-pinned and non-draft manifests are never touched.
    already = tasks._autopin_capability_battery(name, pinned, root, lambda *_: None)
    assert already.capability_battery_hash == digest
    frozen_like = Manifest.from_dict(
        dict(es.load_raw(name, root), status="frozen"))
    untouched = tasks._autopin_capability_battery(
        name, frozen_like, root, lambda *_: None)
    assert untouched is frozen_like


def test_judges_panel_alone_triggers_the_rubric_gate(tmp_path):
    # Swift parity: an explicit judges panel means judged evaluation even
    # without evaluation.kind == pairedJudge — the gates must agree across
    # engines or a manifest freezes on one and not the other.
    root = str(tmp_path)
    name = _variant_study(root, "panel-only")
    es.set_protocol(
        name,
        {"judges": [{"name": "j1", "kind": "claude", "model": None}]},
        root)
    with pytest.raises(es.ExperimentStoreError, match="rubric"):
        es.freeze(name, root=root)


def _git(root, *args):
    import subprocess
    subprocess.run(["git", "-C", root, *args], check=True, capture_output=True,
                   env={"HOME": root, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": os.environ["PATH"]})


def _git_out(root, *args):
    import subprocess
    out = subprocess.run(["git", "-C", root, *args], check=True,
                         capture_output=True, text=True,
                         env={"HOME": root, "PATH": os.environ["PATH"]})
    return out.stdout.strip()


def _init_workspace_repo(root):
    """A standalone git workspace: init + LOCAL identity, so the freeze
    auto-commit (which runs with the ambient process env) can commit."""
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@t")


def _dirty_pinned_study(root, name):
    """A variant study plus a pinned-but-uncommitted task-prompt file."""
    _variant_study(root, name)
    prompts_rel = f"prompts/tasks/{name}.jsonl"
    digest = _write(os.path.join(root, prompts_rel),
                    '{"id": "a", "prompt": "p"}\n')
    es.set_protocol(name,
                    {"taskPromptsFile": prompts_rel, "taskPromptsHash": digest},
                    root)
    _validate_evidence(root, name, battery_results=_battery_rows("any"))
    return prompts_rel


def test_freeze_auto_commits_a_standalone_git_workspace(tmp_path):
    # (a) Freeze = commit + stamp in one gesture: in a workspace that is its
    # own git work-tree root, dirty pinned inputs are AUTO-COMMITTED (the
    # uncommitted-pins gate does not fire), the stamped gitCommit is the
    # post-freeze HEAD, and that commit contains the pinned inputs plus the
    # experiments/<name>/pinned/ snapshot.
    root = str(tmp_path)
    _init_workspace_repo(root)
    name = "gitgate"
    prompts_rel = _dirty_pinned_study(root, name)
    frozen = es.freeze(name, root=root)
    assert frozen["status"] == "frozen"
    # Two-commit lifecycle (Swift parity): the stamped gitCommit is the
    # pre-stamp auto-commit (the one containing the pinned bytes); a
    # follow-up "(stamp)" commit lands the frozen manifest/canonical/
    # preregistration so the workspace ends CLEAN.
    assert _git_out(root, "log", "-1", "--format=%s") == f"freeze {name} (stamp)"
    assert _git_out(root, "log", "-1", "--format=%s", "HEAD~1") == f"freeze {name}"
    assert frozen["gitCommit"] == _git_out(root, "rev-parse", "HEAD~1")
    assert _git_out(root, "status", "--porcelain") == ""
    tracked = set(_git_out(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    assert prompts_rel in tracked
    assert f"experiments/{name}/pinned/{prompts_rel}" in tracked
    for artifact in ("experiment.json", "freeze-canonical.json", "preregistration.md"):
        assert f"experiments/{name}/{artifact}" in tracked
    # The engine stamp landed alongside the freeze stamps — the full build
    # identity, not the bare declared version (which never changes).
    from steerlab_server.build_identity import engine_version
    assert frozen["appVersion"] == engine_version()
    assert frozen["appVersion"].startswith("steerlab-server ")


def test_freeze_gates_when_workspace_is_inside_a_larger_repo(tmp_path):
    # (b) Auto-commit safety rule: the workspace is a SUBDIRECTORY of a larger
    # repository (e.g. a source checkout) — committing that whole repo as a
    # freeze side effect would be hostile, so auto-commit is skipped and the
    # uncommitted-pins gate FIRES instead.
    outer = str(tmp_path)
    _git(outer, "init", "-q")
    root = os.path.join(outer, "ws")
    os.makedirs(root)
    name = "nested"
    _dirty_pinned_study(root, name)
    with pytest.raises(es.ExperimentStoreError, match="not committed to git"):
        es.freeze(name, root=root)
    # Committing in the OUTER repo (the operator's gesture) satisfies the
    # gate; freeze then stamps the outer HEAD without creating a new commit.
    _git(outer, "add", "-A")
    _git(outer, "commit", "-q", "-m", "pin inputs")
    head_before = _git_out(outer, "rev-parse", "HEAD")
    frozen = es.freeze(name, root=root)
    assert frozen["status"] == "frozen"
    assert frozen["gitCommit"] == head_before  # no auto-commit happened
    assert _git_out(outer, "rev-parse", "HEAD") == head_before


def test_force_freeze_still_skips_the_git_pin_gate(tmp_path):
    # (c) --force skips the gate (loudly, like every evidence gate) even where
    # auto-commit is impossible.
    outer = str(tmp_path)
    _git(outer, "init", "-q")
    root = os.path.join(outer, "ws")
    os.makedirs(root)
    _dirty_pinned_study(root, "forced")
    forced = es.freeze("forced", force=True, root=root)
    assert forced["status"] == "frozen"


def test_freeze_snapshots_all_pinned_inputs(tmp_path):
    # The no-git reproducibility floor: freeze copies EVERY pinned input into
    # experiments/<name>/pinned/ — files byte-identically, concept stimulus
    # DIRECTORIES recursively — without rewriting the manifest's canonical
    # prompts/-resident paths (the hashes prove identity).
    root = str(tmp_path)
    _init_workspace_repo(root)
    es.create("snap", model_id="org/m", revision="abc", root=root)
    pos = '{"text": "afraid"}\n'
    neg = '{"text": "calm"}\n'
    _write(os.path.join(root, "prompts", "concepts", "fear", "positive.jsonl"), pos)
    _write(os.path.join(root, "prompts", "concepts", "fear", "negative.jsonl"), neg)
    es.attach("snap", ["fear"], root=root)
    prompts_rel = "prompts/tasks/items.jsonl"
    prompt_line = '{"id": "a", "prompt": "p"}\n'
    digest = _write(os.path.join(root, prompts_rel), prompt_line)
    es.set_protocol(
        "snap", {"taskPromptsFile": prompts_rel, "taskPromptsHash": digest}, root)
    _validate_evidence(root, "snap")
    frozen = es.freeze("snap", root=root)
    assert frozen["status"] == "frozen"
    pinned = os.path.join(root, "experiments", "snap", "pinned")
    # Concept directory mirrored recursively, bytes identical.
    for filename, expected in (("positive.jsonl", pos), ("negative.jsonl", neg)):
        copied = os.path.join(pinned, "prompts", "concepts", "fear", filename)
        assert open(copied, encoding="utf-8").read() == expected
    # File pin mirrored, bytes identical.
    assert open(os.path.join(pinned, prompts_rel),
                encoding="utf-8").read() == prompt_line
    # No path rewriting: the manifest still points at the canonical tree.
    assert frozen["taskPromptsFile"] == prompts_rel
    # The snapshot is inside the stamped freeze commit.
    tracked = set(_git_out(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    assert f"experiments/snap/pinned/{prompts_rel}" in tracked
    assert "experiments/snap/pinned/prompts/concepts/fear/positive.jsonl" in tracked
