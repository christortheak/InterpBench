"""One measurement pipeline for every condition (2026-07-13 unification).

The finding: baseline/concept conditions and variant conditions used different
measurement paths — the variant path generated sampled text and basic text
metrics only, with no prompt-metadata copy, no parsedChoice, no answer-token
instrument, and no intervention/sampling metadata, so summaries.csv carried
blank choice-rate fields for a no-intervention variant whose raw outputs were
identical to baseline. Now every condition resolves to an
``EffectiveCondition`` and runs through the shared ``_execute_condition``
executor; these tests pin the acceptance list:

1. field-set equality between baseline and variant records (minus the variant
   provenance fields);
2. parsedChoice + non-blank summaries/report choice rates for BOTH conditions;
3. instrument records for every condition — including a variant with
   injections and a (mock) adapter — with identical injection semantics
   between the instrument and sampled generation;
5. a no-intervention variant at temperature 0 reproduces baseline outputs
   item-for-item, and report.json stamps the computable parity summary
   ``conditions[].agreementWithBaseline`` = {"n", "agreement"};
plus checkpoint-mid-variant resume to a byte-identical union, analyze
uniformity over unified records, and the historical variant error-record
shape.
"""

import csv
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob as logprob_mod
from steerlab_server.experiment import model_variant, resume, tasks
from steerlab_server.experiment.generate import CellInjection
from steerlab_server.steering.vector_store import ConceptVectors

OPTIONS = ["affirm", "reverse"]


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _plain_artifact(**extra):
    artifact = {"name": "v-plain", "baseModelID": "org/m",
                "promptMode": "chatAssistant", "temperature": 0.0,
                "alphaInNormUnits": False, "injections": [], "adapters": []}
    artifact.update(extra)
    return artifact


def _study_fixture(root, name, *, artifact=None, variant_name="v-plain"):
    """A six-item categorical variant study: implicit baseline + one variant
    condition, options+target+arm(+caseID) on every item, temperature 0."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    raw["variantConditions"] = [{
        "name": variant_name, "artifactPath": "runs/variants/v.json",
        "artifactHash": "aa" * 32,
        "artifact": artifact if artifact is not None else _plain_artifact(),
    }]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    lines = []
    for i in range(6):
        lines.append(json.dumps({
            "id": f"p{i}", "prompt": f"Decide case {i}.",
            "options": OPTIONS, "target": "reverse",
            "arm": "treatment" if i % 2 == 0 else "control",
            "caseID": f"case-{i}"}))
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
    """Deterministic on (prompt, injections): steering flips the choice."""
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False):
        if counter is not None:
            counter[0] += 1
            if arm_flag_at is not None and counter[0] == arm_flag_at:
                flag.request()
        if log is not None:
            log.append({"kind": "generate", "prompt": prompt,
                        "injections": list(injections or []),
                        "promptMode": prompt_mode, "modelID": model_id})
        choice = "reverse" if injections else "affirm"
        return f"I choose {choice} in {prompt}"
    return generate


def _fake_score_options(log=None):
    """A REAL ChoiceResult (full as_record_fields contract, equal option token
    counts so the length guard passes); steering flips the selected option."""
    def score_options(model, prompt, options, *, model_id=None, injections=None,
                      prompt_mode=None, system_prompt=None,
                      qwen_thinking_enabled=False):
        if log is not None:
            log.append({"kind": "instrument", "prompt": prompt,
                        "injections": list(injections or []),
                        "promptMode": prompt_mode, "modelID": model_id})
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


def _split(records, condition):
    mine = [r for r in records if r.get("condition") == condition]
    sampled = [r for r in mine if "instrument" not in r and "error" not in r]
    instruments = [r for r in mine if "instrument" in r]
    return sampled, instruments


# Variant-only provenance keys stamped on every record of a variant
# condition. agentPlaygroundTemperature (2026-07-21, study-owned sampling)
# is the artifact's stored Playground temperature — provenance only; the
# record's "temperature" field is the single source of what governed
# generation.
PROVENANCE_FIELDS = {"variantArtifactPath", "variantArtifactHash",
                     "agentPlaygroundTemperature"}


# --- acceptance 1: field-set equality baseline vs variant ---------------------

def test_variant_records_carry_the_full_ordinary_field_set(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "uni1")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("uni1", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    base_sampled, base_instruments = _split(records, "baseline")
    var_sampled, var_instruments = _split(records, "v-plain")
    assert len(base_sampled) == len(var_sampled) == 6
    assert len(base_instruments) == len(var_instruments) == 6

    for base, variant in zip(sorted(base_sampled, key=lambda r: r["promptID"]),
                             sorted(var_sampled, key=lambda r: r["promptID"])):
        assert set(variant) - PROVENANCE_FIELDS == set(base)
        assert PROVENANCE_FIELDS <= set(variant)
    for base, variant in zip(
            sorted(base_instruments, key=lambda r: r["promptID"]),
            sorted(var_instruments, key=lambda r: r["promptID"])):
        assert set(variant) - PROVENANCE_FIELDS == set(base)
        assert PROVENANCE_FIELDS <= set(variant)

    # The previously missing metadata, spot-checked on a variant record.
    record = var_sampled[0]
    assert record["arm"] in ("treatment", "control")
    assert record["target"] == "reverse"
    assert record["caseID"].startswith("case-")
    assert record["promptMode"] == "chatAssistant"
    assert record["modelID"] == "org/m"
    assert record["seedPolicy"] == "manifestSeeds"
    assert record["interventionState"]["variant"] == "v-plain"
    assert record["interventionState"]["slots"] == []
    # Ordinary records are NOT polluted with variant keys.
    assert not (PROVENANCE_FIELDS & set(base_sampled[0]))
    assert "variant" not in base_sampled[0]["interventionState"]


# --- acceptance 2: parsedChoice + non-blank choice rates everywhere -----------

def test_choice_rates_populate_for_baseline_and_variant(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "uni2")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("uni2", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    for condition in ("baseline", "v-plain"):
        sampled, instruments = _split(records, condition)
        assert all(r["parsedChoice"] == "affirm" for r in sampled)
        assert all(r["selected"] == "affirm" for r in instruments)

    with open(os.path.join(run_dir, "summaries.csv"), encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    assert set(by_condition) == {"baseline", "v-plain"}
    for condition, condition_rows in by_condition.items():
        assert len(condition_rows) == 6
        for row in condition_rows:
            assert json.loads(row["choiceRates"]) == {"affirm": 1.0}, condition
            assert row["selectedOption"] == "affirm"
            assert row["targetProbability"] != ""
            assert row["targetLogOdds"] != ""

    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    for condition in ("baseline", "v-plain"):
        block = report["conditions"][condition]
        assert block["choiceReadouts"] == 6
        assert block["choiceRate"] == 0.0  # every item's target is "reverse"


# --- acceptance 3: instruments under variant injections + (mock) adapter ------

def test_instrument_runs_under_variant_adapter_with_identical_injections(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    artifact = _plain_artifact(
        name="v-steered",
        injections=[{"concept": "fear", "vectorArtifactID": "runs/x/fear",
                     "layer": 1, "alpha": 2.0}],
        adapters=[{"adapterDirectory": "runs/x/adapter", "adapterHash": "b" * 64}])
    prompts = _study_fixture(root, "uni3", artifact=artifact,
                             variant_name="v-steered")
    calls = []
    _patch(monkeypatch, _fake_generate(log=calls), _fake_score_options(log=calls))
    variant_cells = [CellInjection(layer=1, vector=[1.0, 0.0], alpha=2.0)]
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant: list(variant_cells))
    adapter_events = []
    # `root` is accepted AND recorded: a stub that omitted it is what let the
    # run loop load an adapter from a different workspace than the one it
    # verified in (external review round 11).
    adapter_roots = []
    monkeypatch.setattr(model_variant, "apply_adapter",
                        lambda model, variant, root=None: (
                            calls.append({"kind": "apply"}),
                            adapter_events.append("apply"),
                            adapter_roots.append(root),
                            "adapter-h")[-1])
    monkeypatch.setattr(model_variant, "remove_adapter",
                        lambda model, handle: (calls.append({"kind": "remove"}),
                                               adapter_events.append(handle)))

    run_dir = tasks.run("uni3", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    base_sampled, base_instruments = _split(records, "baseline")
    var_sampled, var_instruments = _split(records, "v-steered")
    # Every condition emits instrument records — the variant included.
    assert len(base_instruments) == len(var_instruments) == 6
    assert all(r["instrument"] == "answerTokenLogprob" for r in var_instruments)
    # Steering moved the deterministic readout AND the sampled parse.
    assert all(r["selected"] == "reverse" for r in var_instruments)
    assert all(r["parsedChoice"] == "reverse" for r in var_sampled)
    assert all(r["selected"] == "affirm" for r in base_instruments)

    # Identical injection semantics: the instrument and the sampled generation
    # of the SAME condition were armed with the SAME injections.
    def injections_by_kind(after, before):
        window = calls[after:before] if before is not None else calls[after:]
        return ({tuple((c.layer, tuple(c.vector), c.alpha)
                       for c in e["injections"])
                 for e in window if e["kind"] == "instrument"},
                {tuple((c.layer, tuple(c.vector), c.alpha)
                       for c in e["injections"])
                 for e in window if e["kind"] == "generate"})

    apply_index = next(i for i, e in enumerate(calls) if e["kind"] == "apply")
    remove_index = next(i for i, e in enumerate(calls) if e["kind"] == "remove")
    base_instrument_inj, base_generate_inj = injections_by_kind(0, apply_index)
    var_instrument_inj, var_generate_inj = injections_by_kind(apply_index,
                                                              remove_index)
    assert base_instrument_inj == base_generate_inj == {()}
    expected = {tuple((c.layer, tuple(c.vector), c.alpha) for c in variant_cells)}
    assert var_instrument_inj == var_generate_inj == expected
    # The mock adapter was applied around the WHOLE variant condition
    # (instrument scoring included) and removed exactly once.
    assert adapter_events == ["apply", "adapter-h"]
    assert remove_index == len(calls) - 1
    # The variant's prompt configuration drove both measurement kinds.
    assert all(e["promptMode"] == "chatAssistant"
               for e in calls if e["kind"] in ("instrument", "generate"))


# --- acceptance 5: baseline parity of a no-intervention variant ----------------

def test_no_intervention_variant_matches_baseline_and_stamps_agreement(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "uni5")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("uni5", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    base_sampled, _ = _split(records, "baseline")
    var_sampled, _ = _split(records, "v-plain")
    base_outputs = {r["promptID"]: r["output"] for r in base_sampled}
    var_outputs = {r["promptID"]: r["output"] for r in var_sampled}
    assert base_outputs == var_outputs  # item-for-item identical

    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    parity = report["conditions"]["v-plain"]["agreementWithBaseline"]
    # 6 sampled parses + 6 instrument selections, all agreeing.
    assert parity == {"n": 12, "agreement": 1.0}
    assert "agreementWithBaseline" not in report["conditions"]["baseline"]


def test_agreement_with_baseline_counts_disagreements_and_skips_null_parses():
    from steerlab_server.experiment.manifest import Manifest
    manifest = Manifest.from_dict({"name": "s", "modelID": "m",
                                   "concepts": [], "conditions": []})

    def sampled(condition, prompt_id, choice):
        record = {"condition": condition, "promptID": prompt_id,
                  "sampleIndex": 0, "output": "t", "wordCount": 1,
                  "distinct2": 1.0, "target": "A"}
        record["parsedChoice"] = choice
        return record

    def instrument(condition, prompt_id, selected):
        return {"condition": condition, "promptID": prompt_id,
                "instrument": "answerTokenLogprob", "selected": selected,
                "target": "A", "choiceProbability": {"A": 1.0},
                "logOdds": {"A": 3.0}}

    records = [
        sampled("baseline", "p0", "A"), sampled("baseline", "p1", None),
        instrument("baseline", "p0", "A"),
        sampled("steered", "p0", "B"),   # disagrees
        sampled("steered", "p1", "A"),   # baseline parse was None: excluded
        instrument("steered", "p0", "A"),  # agrees
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        tasks._write_report("s", manifest, records, out)
        report = json.loads(open(os.path.join(out, "report.json")).read())
    parity = report["conditions"]["steered"]["agreementWithBaseline"]
    assert parity == {"n": 2, "agreement": 0.5}
    assert "agreementWithBaseline" not in report["conditions"]["baseline"]


# --- resume under unification --------------------------------------------------

def test_checkpoint_mid_variant_condition_resumes_byte_identical(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "uniresume")

    # Control: an uninterrupted run.
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    control_dir = tasks.run("uniresume", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None)
    control_bytes = open(os.path.join(control_dir, "generations.jsonl"),
                         "rb").read()
    control_report = open(os.path.join(control_dir, "report.json"), "rb").read()

    # Interrupted: the flag arms during the variant condition's 3rd sampled
    # generation (baseline makes 6; generation 9 is mid-variant).
    flag = resume.CheckpointFlag()
    counter = [0]
    _patch(monkeypatch,
           _fake_generate(counter=counter, arm_flag_at=9, flag=flag),
           _fake_score_options())
    seen = {}
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("uniresume", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    run_dir = seen["dir"]
    assert resume.is_resumable(run_dir)
    interrupted = _records(run_dir)
    assert any(r.get("condition") == "v-plain" for r in interrupted)
    assert 0 < len(interrupted) < len(control_bytes.splitlines())

    # Resume: only the missing records generate, and the union is byte-equal.
    resumed_counter = [0]
    _patch(monkeypatch, _fake_generate(counter=resumed_counter),
           _fake_score_options())
    resumed_dir = tasks.run("uniresume", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None,
                            run_directory=run_dir)
    assert resumed_dir == run_dir
    assert counter[0] == 9 and resumed_counter[0] == 3
    assert open(os.path.join(run_dir, "generations.jsonl"),
                "rb").read() == control_bytes
    assert open(os.path.join(run_dir, "report.json"), "rb").read() == control_report
    assert not resume.is_resumable(run_dir)


# --- analyze consumes unified records uniformly ---------------------------------

def test_analyze_extracts_variant_endpoints_like_ordinary_ones(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    artifact = _plain_artifact(
        name="v-steered",
        injections=[{"concept": "fear", "vectorArtifactID": "runs/x/fear",
                     "layer": 1, "alpha": 2.0}])
    prompts = _study_fixture(root, "unianalyze", artifact=artifact,
                             variant_name="v-steered")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant: [CellInjection(layer=1, vector=[1.0, 0.0],
                                                       alpha=2.0)])
    run_dir = tasks.run("unianalyze", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    out = tasks.analyze("unianalyze", root=root, source_run=run_dir)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as h:
        # The POOLED rows (stratified companion rows ride in the same file).
        rows = [r for r in csv.DictReader(h) if r["stratifyBy"] == "pooled"]
    endpoints = {r["endpoint"] for r in rows if r["condition"] == "v-steered"}
    # Both the instrument endpoint and the sampled-parse endpoint pair the
    # variant against baseline, exactly as a concept condition would.
    assert {"choiceLogOdds", "choiceRate"} <= endpoints
    by_endpoint = {r["endpoint"]: r for r in rows
                   if r["condition"] == "v-steered"}
    assert int(by_endpoint["choiceRate"]["n"]) == 6
    assert float(by_endpoint["choiceRate"]["deltaMean"]) == 1.0  # 0% → 100% target


# --- invalid variants keep the historical error-record contract -----------------

def test_invalid_variant_emits_error_record_and_baseline_still_measures(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    artifact = _plain_artifact(name="v-bad", baseModelID="other/m")
    prompts = _study_fixture(root, "unibad", artifact=artifact,
                             variant_name="v-bad")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("unibad", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    errors = [r for r in records if "error" in r]
    assert len(errors) == 1
    assert set(errors[0]) == {"experiment", "condition", "error",
                              "variantArtifactPath"}
    assert errors[0]["condition"] == "v-bad"
    assert "base model other/m" in errors[0]["error"]
    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    assert report["conditions"]["v-bad"]["error"]
    assert report["conditions"]["baseline"]["choiceReadouts"] == 6


def test_thinking_variant_refuses_answer_token_instrument(tmp_path, monkeypatch):
    root = str(tmp_path)
    artifact = _plain_artifact(name="v-think", qwenThinkingEnabled=True)
    prompts = _study_fixture(root, "unithink", artifact=artifact,
                             variant_name="v-think")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    run_dir = tasks.run("unithink", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    errors = [r for r in _records(run_dir) if "error" in r]
    assert len(errors) == 1
    assert "thinking" in errors[0]["error"]


def test_inert_carried_conditions_warn_loudly_and_stamp_config(
        tmp_path, monkeypatch):
    """2026-08-11 follow-up to the c20-* incident: a DECLARED agent
    comparison that legally proceeds (agent arms exist) while carrying
    concepts and injection conditions runs agents+baseline only — correct
    by the 2026-07-19 rule, but it must be LOUD at run start and stamped in
    config.json's notes, because a baseline-only result otherwise looks
    completed and ordinary (it cost two GPU rounds to notice). The carried
    condition itself must still NOT execute — the rule is unchanged."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "uni-inert")
    es.add_condition("uni-inert", {
        "name": "fear-4", "bandWidth": 1, "alphaInNormUnits": False,
        "slots": [{"concept": "fear", "layer": 2, "alpha": 1.0}]}, root)
    raw = es.load_raw("uni-inert", root)
    raw["studyType"] = "agentComparison"
    es.save_raw(raw, root)
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    lines: list[str] = []
    run_dir = tasks.run("uni-inert", prompts, root, model_provider=_fake_model,
                        log=lambda msg: lines.append(str(msg)))
    warnings = [line for line in lines
                if "INERT" in line and "WARNING" in line]
    assert len(warnings) == 1
    assert "agentComparison" in warnings[0]
    assert "fear-4" in warnings[0]
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as h:
        config = json.load(h)
    assert config["notes"]["inertConceptMachinery"] == {
        "declaredStudyType": "agentComparison",
        "inertConditions": ["fear-4"],
        "inertConcepts": ["fear"],
    }
    conditions = {r.get("condition") for r in _records(run_dir)}
    assert conditions == {"baseline", "v-plain"}  # the rule stands

    # An operative study (no declared type) stamps NO note and warns
    # nothing — the loudness is scoped to actual inertness.
    prompts2 = _study_fixture(root, "uni-live")
    _patch(monkeypatch, _fake_generate(), _fake_score_options())
    quiet: list[str] = []
    run_dir2 = tasks.run("uni-live", prompts2, root,
                         model_provider=_fake_model,
                         log=lambda msg: quiet.append(str(msg)))
    assert not any("INERT" in line for line in quiet)
    with open(os.path.join(run_dir2, "config.json"), encoding="utf-8") as h:
        assert "inertConceptMachinery" not in json.load(h)["notes"]
