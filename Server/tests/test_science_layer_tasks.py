"""Science-layer manifest fields, seed derivation, matched-norm random
controls, control-matrix generation, and the run-artifact writers."""

import csv
import json
import math

from steerlab_server.experiment import control_matrix, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import vector_math as vm


def _base_manifest(**extra):
    d = {
        "name": "study", "modelID": "test/model",
        "concepts": [], "conditions": [],
        "taskPromptsFile": None,
    }
    d.update(extra)
    return d


def test_manifest_parses_science_fields():
    manifest = Manifest.from_dict(_base_manifest(
        phase="screen", caseFamily="sentencing",
        outcomeInstruments=["answerTokenLogprob", "sampledText"],
        samplesPerItem=8, seedPolicy="derivedSHA256", temperature=0.7,
        humanBaseline={"path": "prompts/baselines/h.csv", "hash": "ab" * 32},
        promotionRule={"fdrThreshold": 0.1, "doseMonotone": False},
        screenTaskPromptsHash="cd" * 32,
        conditions=[{"name": "fear-random", "controlType": "randomMatchedNorm",
                     "slots": [{"concept": "fear", "layer": 10, "alpha": 2.0}]}]))
    assert manifest.phase == "screen"
    assert manifest.case_family == "sentencing"
    assert manifest.samples_per_item == 8
    assert manifest.seed_policy == "derivedSHA256"
    assert manifest.human_baseline.path == "prompts/baselines/h.csv"
    assert manifest.promotion_rule.fdr_threshold == 0.1
    assert manifest.promotion_rule.dose_monotone is False
    assert manifest.conditions[0].control_type == "randomMatchedNorm"


def test_verify_flags_confirm_pool_reuse(tmp_path):
    prompts = tmp_path / "items.jsonl"
    prompts.write_text('{"id": "a", "prompt": "p"}\n')
    import hashlib
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    manifest = Manifest.from_dict(_base_manifest(
        phase="confirm",
        concepts=[],
        variantConditions=[{"name": "v", "artifactPath": "", "artifactHash": "",
                            "artifact": {}}],
        taskPromptsFile=str(prompts), taskPromptsHash=digest,
        screenTaskPromptsHash=digest))
    violations = manifest.verify(str(tmp_path))
    assert any("IDENTICAL to the screen pool" in v for v in violations)

    missing = Manifest.from_dict(_base_manifest(
        phase="confirm",
        variantConditions=[{"name": "v", "artifactPath": "", "artifactHash": "",
                            "artifact": {}}]))
    assert any("screenTaskPromptsHash" in v for v in missing.verify(str(tmp_path)))


def test_verify_flags_bad_sampling_policy(tmp_path):
    manifest = Manifest.from_dict(_base_manifest(
        samplesPerItem=4, temperature=0.0,
        variantConditions=[{"name": "v", "artifactPath": "", "artifactHash": "",
                            "artifact": {}}]))
    violations = manifest.verify(str(tmp_path))
    assert any("temperature > 0" in v for v in violations)
    assert any("derivedSHA256" in v for v in violations)


def test_derive_seed_deterministic_and_distinct():
    a = tasks.derive_seed("hash", "fear-a2", "prompt-1", 0)
    b = tasks.derive_seed("hash", "fear-a2", "prompt-1", 0)
    c = tasks.derive_seed("hash", "fear-a2", "prompt-1", 1)
    d = tasks.derive_seed("hash", "baseline", "prompt-1", 0)
    assert a == b
    assert len({a, c, d}) == 3
    assert 0 <= a < 2 ** 63


def test_matched_norm_random_deterministic_and_norm_matched():
    v1 = tasks._matched_norm_random("cond|fear|10", dimension=64, norm=7.5)
    v2 = tasks._matched_norm_random("cond|fear|10", dimension=64, norm=7.5)
    v3 = tasks._matched_norm_random("cond|fear|11", dimension=64, norm=7.5)
    assert v1 == v2
    assert v1 != v3
    # vm.l2_norm accumulates in float32; match its precision.
    assert math.isclose(vm.l2_norm(v1), 7.5, rel_tol=1e-6)


def test_matched_norm_random_is_gaussian_not_cube_uniform():
    """gaussian-isotropic-v1 distribution sanity (seeded, deterministic).

    Discriminators at dimension 4096: coordinate kurtosis ≈ 3 for a Gaussian
    vs ≈ 1.8 for uniform[-1, 1] (rescaling is per-coordinate linear, so
    kurtosis is invariant to it); and max|coord|·sqrt(n) ≈ 3+ for a Gaussian
    while cube-uniform-then-rescale is capped near sqrt(3) ≈ 1.73 (no
    coordinate exceeds 1/||raw|| with ||raw|| ≈ sqrt(n/3) — the cube-corner
    bias the Gaussian recipe avoids). The Swift twin test pins the same
    contract on its own RNG (VectorMathTests)."""
    n = 4096
    v = tasks._matched_norm_random("dist|check|0", dimension=n, norm=1.0)
    mean = sum(v) / n
    assert abs(mean) < 1e-3
    variance = sum((x - mean) ** 2 for x in v) / n
    m4 = sum((x - mean) ** 4 for x in v) / n
    kurtosis = m4 / variance ** 2
    assert kurtosis > 2.5, f"cube-uniform kurtosis ~1.8, got {kurtosis}"
    assert max(abs(x) for x in v) * math.sqrt(n) > 2.0


def test_intervention_state_stamps_random_vector_algorithm():
    """Random-control records must carry the recipe stamp (cross-engine
    contract string); non-control conditions omit the key, and so do legacy
    runs — unstamped = legacy (Gaussian on this engine, cube-uniform on
    Swift)."""
    manifest = Manifest.from_dict(_base_manifest(conditions=[
        {"name": "fear-random", "controlType": "randomMatchedNorm",
         "slots": [{"concept": "fear", "layer": 10, "alpha": 2.0}]},
        {"name": "fear-a2",
         "slots": [{"concept": "fear", "layer": 10, "alpha": 2.0}]},
    ]))
    control_state = tasks._intervention_state(manifest.conditions[0])
    assert control_state["randomVectorAlgorithm"] == "gaussian-isotropic-v1"
    assert control_state["randomVectorAlgorithm"] == tasks.RANDOM_VECTOR_ALGORITHM
    plain_state = tasks._intervention_state(manifest.conditions[1])
    assert "randomVectorAlgorithm" not in plain_state


def test_control_matrix_conditions():
    cells = control_matrix.control_matrix_conditions("fear", 10, 2.0)
    names = [c["name"] for c in cells]
    assert names == ["fear-a1", "fear-a2", "fear-neg-a2", "fear-randomMatchedNorm-a2"]
    random_cell = cells[-1]
    assert random_cell["controlType"] == "randomMatchedNorm"
    assert random_cell["slots"][0]["alpha"] == 2.0
    negative = cells[-2]
    assert negative["slots"][0]["alpha"] == -2.0
    try:
        control_matrix.control_matrix_conditions("fear", 10, 0.0)
        raise AssertionError("expected ValueError for zero alpha")
    except ValueError:
        pass


def _sample_record(condition, prompt_id, months=None, choice=None, target=None,
                   sample_index=0):
    record = {"condition": condition, "promptID": prompt_id, "output": "text",
              "wordCount": 5, "distinct2": 0.9, "sampleIndex": sample_index}
    if months is not None or condition:  # sentencing records stamp the parse
        record["parsedMonths"] = months
    if choice is not None:
        record["parsedChoice"] = choice
    if target is not None:
        record["target"] = target
    return record


def test_write_summaries_csv(tmp_path):
    records = [
        _sample_record("baseline", "p1", months=12.0, sample_index=0),
        _sample_record("baseline", "p1", months=18.0, sample_index=1),
        _sample_record("baseline", "p1", months=None, sample_index=2),
        {"condition": "fear-a2", "promptID": "p1",
         "instrument": "answerTokenLogprob", "selected": "A",
         "target": "A", "choiceProbability": {"A": 0.8, "B": 0.2},
         "logOdds": {"A": 1.3862943611, "B": -1.3862943611}},
    ]
    tasks._write_summaries_csv(records, str(tmp_path))
    with open(tmp_path / "summaries.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline = next(r for r in rows if r["condition"] == "baseline")
    assert baseline["samples"] == "3"
    assert math.isclose(float(baseline["monthsParseFailureRate"]), 1 / 3,
                        rel_tol=1e-5)
    assert float(baseline["monthsMean"]) == 15.0
    fear = next(r for r in rows if r["condition"] == "fear-a2")
    assert fear["selectedOption"] == "A"
    assert math.isclose(float(fear["targetProbability"]), 0.8)


def test_endpoint_values_and_pairing():
    records = [
        # Instrument endpoint per condition.
        {"condition": "baseline", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "A",
         "logOdds": {"A": 0.0, "B": 0.0}},
        {"condition": "fear-a2", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "A",
         "logOdds": {"A": 1.5, "B": -1.5}},
        # Sampled sentencing endpoint (2 samples per cell).
        _sample_record("baseline", "p2", months=12.0, sample_index=0),
        _sample_record("baseline", "p2", months=16.0, sample_index=1),
        _sample_record("fear-a2", "p2", months=20.0, sample_index=0),
        _sample_record("fear-a2", "p2", months=24.0, sample_index=1),
    ]
    endpoints = tasks._endpoint_values(records)
    assert endpoints["choiceLogOdds"]["fear-a2"]["p1"] == 1.5
    assert endpoints["choiceLogOdds"]["baseline"]["p1"] == 0.0
    assert endpoints["meanMonths"]["fear-a2"]["p2"] == 22.0
    assert endpoints["meanMonths"]["baseline"]["p2"] == 14.0
    assert "monthsSpread" in endpoints
    # No declared parser kind (or the months kind): months labels are
    # honest, no neutral twin endpoints.
    assert "parsedValueMean" not in endpoints
    assert "parsedValueSpread" not in endpoints


def test_non_months_parser_adds_honest_parsed_value_endpoints():
    """Endpoint-label honesty (2026-08-06): a percentage or 1-7 scale
    parser writes the parsedMonths record key too, so its effect rows were
    labeled meanMonths/monthsSpread — verified misleading on real runs.
    A non-months parser kind now ADDITIONALLY emits the neutral
    parsedValueMean/parsedValueSpread endpoints with identical values; the
    months names stay as deprecated aliases so existing readers (residual
    joins, _PRIMARY_ENDPOINT_ORDER, the explorer) keep working."""
    records = [
        _sample_record("baseline", "p2", months=3.0, sample_index=0),
        _sample_record("baseline", "p2", months=5.0, sample_index=1),
        _sample_record("fear-a2", "p2", months=6.0, sample_index=0),
        _sample_record("fear-a2", "p2", months=2.0, sample_index=1),
    ]
    endpoints = tasks._endpoint_values(records, numeric_parser_kind="number")
    assert endpoints["parsedValueMean"]["baseline"]["p2"] == 4.0
    assert endpoints["parsedValueMean"] == endpoints["meanMonths"]
    assert endpoints["parsedValueSpread"] == endpoints["monthsSpread"]

    # The months kind keeps its honest labels alone.
    months_only = tasks._endpoint_values(
        records, numeric_parser_kind="durationMonths")
    assert "parsedValueMean" not in months_only
    assert "meanMonths" in months_only


def test_write_report_handles_instrument_records(tmp_path):
    manifest = Manifest.from_dict(_base_manifest(
        variantConditions=[{"name": "v", "artifactPath": "", "artifactHash": "",
                            "artifact": {}}]))
    records = [
        _sample_record("baseline", "p1", months=12.0),
        {"condition": "baseline", "promptID": "p1",
         "instrument": "answerTokenLogprob", "selected": "A", "target": "A",
         "choiceProbability": {"A": 1.0}, "logOdds": {"A": 27.0}},
    ]
    tasks._write_report("study", manifest, records, str(tmp_path))
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["conditions"]["baseline"]["generations"] == 1
    assert report["conditions"]["baseline"]["choiceReadouts"] == 1
    # metrics.csv skips the instrument record instead of writing zeros.
    tasks._write_metrics_csv(records, str(tmp_path))
    with open(tmp_path / "metrics.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1


def test_option_length_gate():
    from steerlab_server.experiment import logprob

    def result(counts):
        return logprob.ChoiceResult(options=[
            logprob.OptionScore(option=f"o{i}", token_ids=list(range(n)) or [0],
                                token_logprobs=[-1.0] * max(1, n))
            for i, n in enumerate(counts)])

    equal = Manifest.from_dict(_base_manifest())
    tasks._check_option_lengths(result([2, 2]), equal, "p1")  # no raise

    unequal = Manifest.from_dict(_base_manifest())
    try:
        tasks._check_option_lengths(result([1, 4]), unequal, "p1")
        raise AssertionError("expected unequal-length rejection")
    except RuntimeError as exc:
        assert "unequal token counts" in str(exc)

    acknowledged = Manifest.from_dict(
        _base_manifest(acknowledgeUnequalOptionLengths=True))
    tasks._check_option_lengths(result([1, 4]), acknowledged, "p1")  # no raise


# --- modality tag + behavioral-fingerprint table (RESULTS-ARCHITECTURE §1) ----

def _variant_condition(name, artifact):
    base = {"baseModelID": "org/m", "promptMode": "chatAssistant"}
    base.update(artifact)
    return {"name": name, "artifactPath": "", "artifactHash": "", "artifact": base}


def test_condition_modality_derivation():
    manifest = Manifest.from_dict(_base_manifest(
        conditions=[
            {"name": "fear-a2", "slots": [{"concept": "fear", "layer": 1, "alpha": 2.0}]},
            {"name": "fear-rand", "controlType": "randomMatchedNorm",
             "slots": [{"concept": "fear", "layer": 1, "alpha": 2.0}]},
        ],
        variantConditions=[
            _variant_condition("v-adapter", {"adapters": [{"adapterDirectory": "runs/x/a"}]}),
            _variant_condition("v-prompt", {"systemPrompt": "You feel fear."}),
            _variant_condition("v-inj", {"injections": [
                {"concept": "fear", "vectorArtifactID": "v", "layer": 1, "alpha": 2.0}]}),
            _variant_condition("v-stacked", {
                "adapters": [{"adapterDirectory": "runs/x/a"}],
                "systemPrompt": "You feel fear."}),
            _variant_condition("v-empty", {}),
        ]))
    modality = tasks._condition_modalities(manifest)
    assert modality["baseline"] == "none"
    # Steering slots (matched-norm controls included) are injection modality.
    assert modality["fear-a2"] == "injection"
    assert modality["fear-rand"] == "injection"
    assert modality["v-adapter"] == "adapter"
    assert modality["v-prompt"] == "systemPrompt"
    assert modality["v-inj"] == "injection"
    assert modality["v-stacked"] == "stacked"
    assert modality["v-empty"] == "none"


def test_analyze_stamps_modality_and_writes_fingerprints(tmp_path):
    exp_dir = tmp_path / "experiments" / "study"
    exp_dir.mkdir(parents=True)
    manifest_dict = _base_manifest(
        conditions=[{"name": "fear-a2",
                     "slots": [{"concept": "fear", "layer": 1, "alpha": 2.0}]}],
        variantConditions=[
            _variant_condition("fear-adapter",
                               {"adapters": [{"adapterDirectory": "runs/x/a"}]}),
            _variant_condition("fear-prompt", {"systemPrompt": "You feel fear."}),
            _variant_condition("fear-stacked", {
                "adapters": [{"adapterDirectory": "runs/x/a"}],
                "injections": [{"concept": "fear", "vectorArtifactID": "v",
                                "layer": 1, "alpha": 2.0}]}),
        ])
    (exp_dir / "experiment.json").write_text(json.dumps(manifest_dict), encoding="utf-8")

    run_dir = tmp_path / "runs" / "20260101T000000000-exp-study-run"
    run_dir.mkdir(parents=True)
    # Epoch stamp: analyze only accepts source runs produced under the
    # CURRENT manifest content hash.
    (run_dir / "experiment-hash.txt").write_text(
        Manifest.from_dict(manifest_dict).content_hash() + "\n", encoding="utf-8")
    shifts = {"baseline": 0.0, "fear-a2": 6.0, "fear-adapter": 3.0,
              "fear-prompt": 1.0, "fear-stacked": 9.0}
    records = []
    for condition, shift in shifts.items():
        for prompt_id, base_months in (("p1", 12.0), ("p2", 18.0)):
            records.append({"condition": condition, "promptID": prompt_id,
                            "output": "text", "wordCount": 5, "distinct2": 0.9,
                            "parsedMonths": base_months + shift})
    (run_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    out = tasks.analyze("study", root=str(tmp_path))

    with open(f"{out}/effect-sizes.csv", encoding="utf-8") as handle:
        effect_rows = list(csv.DictReader(handle))
    modality = {r["condition"]: r["modality"] for r in effect_rows}
    assert modality == {"fear-a2": "injection", "fear-adapter": "adapter",
                        "fear-prompt": "systemPrompt", "fear-stacked": "stacked"}

    with open(f"{out}/fingerprints.csv", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["condition", "modality", "endpoint",
                                     "effect", "ciLow", "ciHigh", "n"]
        fingerprint_rows = list(reader)
    # One row per condition × endpoint, sorted by endpoint then condition for
    # cross-condition comparison. Endpoints now include the prose pair
    # (wordCount, distinct2) alongside meanMonths: Swift's effect sizes have
    # always carried those two, and the server's collector never emitted them
    # — a parity gap that also left PANEL studies (prose turns, no choice
    # instrument) with an empty endpoint set and therefore no effect sizes at
    # all. Correction is applied per endpoint family, so adding families does
    # not perturb meanMonths' adjusted p.
    assert sorted({r["endpoint"] for r in fingerprint_rows}) == [
        "distinct2", "meanMonths", "wordCount"]
    months_rows = [r for r in fingerprint_rows if r["endpoint"] == "meanMonths"]
    assert [r["condition"] for r in months_rows] == [
        "fear-a2", "fear-adapter", "fear-prompt", "fear-stacked"]
    assert all(r["n"] == "2" for r in months_rows)
    by_condition = {r["condition"]: r for r in months_rows}
    # Pure reshaping of the effect rows — same paired effect, no new stats.
    assert math.isclose(float(by_condition["fear-a2"]["effect"]), 6.0)
    assert math.isclose(float(by_condition["fear-stacked"]["effect"]), 9.0)
    assert by_condition["fear-adapter"]["modality"] == "adapter"


# --- factorial `factors` metadata reaches run evidence (review finding 3) ----

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob as logprob_mod


def _write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _factor_run(tmp_path, monkeypatch):
    """A two-item baseline run with sampled + choice-instrument records:
    item f1 declares a factorial cell, item plain does not (the byte-level
    no-schema-churn control)."""
    root = str(tmp_path)
    # No concept is attached: this is a BASELINE run about record shape, and
    # attaching a concept without giving it an injection condition is the
    # measures-nothing shape the run loop now refuses (2026-08-17,
    # ``no_measured_conditions_problem``). The concept was never used here —
    # ``_extract_all`` is stubbed to {} below.
    es.create("fact", model_id="org/m", revision="abc", root=root)
    raw = es.load_raw("fact", root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write_file(prompts_path, "\n".join([
        # `target` is declared because the study declares answerTokenLogprob:
        # its endpoint IS the declared target's log-odds, and a file that
        # names none is refused at run start (open-issues #6) rather than
        # having `options[0]` synthesized into every record.
        json.dumps({"id": "f1", "prompt": "Cell one.",
                    "options": ["yes", "no"], "target": "yes",
                    "factors": {"anchor": "low", "frame": "gain"}}),
        json.dumps({"id": "plain", "prompt": "No factors.",
                    "options": ["yes", "no"], "target": "yes"}),
    ]) + "\n")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {})
    monkeypatch.setattr(tasks, "generate",
                        lambda model, prompt, **kw: "an answer")

    def score_options(model, prompt, options, **kw):
        return logprob_mod.ChoiceResult(options=[
            logprob_mod.OptionScore(option=o, token_ids=[i + 1],
                                    token_logprobs=[-0.5 - i])
            for i, o in enumerate(options)], prompt_token_count=3)

    monkeypatch.setattr(logprob_mod, "score_options", score_options)
    run_dir = tasks.run("fact", prompts_path, root, model_provider=_fake_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"),
              encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return run_dir, records


def test_factors_reach_both_record_kinds_and_metrics_csv(tmp_path, monkeypatch):
    """Swift twin ``recordsCarryFactorsForBothRecordKinds`` +
    ``metricsCSVAddsFactorColumnsOnlyWhenDeclared``: the item's `factors`
    object lands verbatim on sampled AND instrument records, factor-less
    items produce records WITHOUT the key, and metrics.csv appends one
    ``factor_<name>`` column per declared factor (sorted union)."""
    run_dir, records = _factor_run(tmp_path, monkeypatch)
    sampled = {r["promptID"]: r for r in records if "instrument" not in r}
    instruments = {r["promptID"]: r for r in records if "instrument" in r}
    expected = {"anchor": "low", "frame": "gain"}
    assert sampled["f1"]["factors"] == expected
    assert instruments["f1"]["factors"] == expected
    assert "factors" not in sampled["plain"]
    assert "factors" not in instruments["plain"]
    with open(os.path.join(run_dir, "metrics.csv"),
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    header = list(rows[0].keys())
    assert header[-2:] == ["factor_anchor", "factor_frame"]
    by_id = {r["promptID"]: r for r in rows}
    assert by_id["f1"]["factor_anchor"] == "low"
    assert by_id["f1"]["factor_frame"] == "gain"
    assert by_id["plain"]["factor_anchor"] == ""


def test_metrics_csv_without_factors_keeps_the_historical_header(tmp_path):
    records = [{"condition": "baseline", "seed": 0, "promptIndex": 0,
                "promptID": "p1", "wordCount": 10, "distinct2": 0.8}]
    tasks._write_metrics_csv(records, str(tmp_path))
    text = open(os.path.join(str(tmp_path), "metrics.csv"),
                encoding="utf-8").read()
    assert text.splitlines()[0] == \
        "condition,seed,promptIndex,promptID,wordCount,distinct2"
    assert "factor_" not in text


def test_load_prompts_validates_factors_shape(tmp_path):
    root = str(tmp_path)
    manifest = Manifest.from_dict({"name": "x", "modelID": "org/m"})
    _write_file(os.path.join(root, "prompts", "bad.jsonl"),
                '{"id": "bad", "prompt": "x", "factors": {"anchor": 3}}\n')
    with pytest.raises(RuntimeError) as excinfo:
        tasks._load_prompts(manifest, "prompts/bad.jsonl", root)
    # The pinned cross-engine message (Swift twin
    # ``ExperimentTasks.taskPromptFactorsMessage``).
    assert str(excinfo.value) == (
        "task prompts: item 'bad' has a 'factors' value that is not a flat "
        "string-to-string object — factor names and level names must both "
        "be strings")
    _write_file(os.path.join(root, "prompts", "bad2.jsonl"),
                '{"id": "bad2", "prompt": "x", "factors": "anchor=low"}\n')
    with pytest.raises(RuntimeError, match="flat string-to-string"):
        tasks._load_prompts(manifest, "prompts/bad2.jsonl", root)
    # An EMPTY factors object is treated as absent; a real one is carried.
    _write_file(os.path.join(root, "prompts", "ok.jsonl"),
                '{"id": "e", "prompt": "x", "factors": {}}\n'
                '{"id": "f", "prompt": "y", "factors": {"a": "b"}}\n')
    prompts = tasks._load_prompts(manifest, "prompts/ok.jsonl", root)
    assert "factors" not in prompts[0]
    assert prompts[1]["factors"] == {"a": "b"}
