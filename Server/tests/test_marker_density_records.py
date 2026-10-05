"""Python study runs record marker density, as Mac runs do.

Every sampled response's record carries ``markerDensity``: one value per
declared concept, measured from the concept's ``markers.json`` as the
response is generated (a concept without one reads 0, as on the Mac). The
Python engine's ``analyze`` then pairs ``<concept>MarkerDensity`` for its own
runs, which it could pair before only for records the Mac engine wrote.

The expected values are worked out by hand from the fake responses below.
"""

import csv
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.vector_materialization as _owner_vector_materialization
from steerlab_server.experiment import condition_execution
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.scoring import MarkerRubric
from steerlab_server.steering.vector_store import ConceptVectors


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study(root, name):
    """Two concepts, one with a markers.json ("warm") and one without
    ("plain"), an implicit baseline and one steering arm, two items."""
    for concept in ("warm", "plain"):
        concept_dir = os.path.join(root, "prompts", "concepts", concept)
        _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "toasty"}\n')
        _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "frosty"}\n')
    _write(os.path.join(root, "prompts", "concepts", "warm", "markers.json"),
           json.dumps({"words": ["warm", "été"]}))
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["warm", "plain"], root=root)
    es.add_condition(name, {"name": "steered", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "warm", "layer": 1,
                                       "alpha": 1.0}]}, root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 8
    raw["outcomeInstruments"] = ["sampledText"]
    es.save_raw(raw, root)
    prompts = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts, '{"id": "p0", "prompt": "Describe the day."}\n'
                    '{"id": "p1", "prompt": "Describe the night."}\n')
    return prompts


def _bundle():
    return _owner_vector_materialization.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


#: The fake responses. Steered: "warm été warm cold" is four words with three
#: markers (0.75) on the day item and "a warm night" three words with one
#: marker (1/3) on the night item; the baseline mentions no marker.
RESPONSES = {
    (True, "day"): "warm été warm cold",
    (True, "night"): "a warm night",
    (False, "day"): "a cold day",
    (False, "night"): "a cold night",
}


def _generate(model, prompt, *, injections=None, **kwargs):
    return RESPONSES[(bool(injections), "day" if "day" in prompt else "night")]


def _run(tmp_path, monkeypatch, name):
    root = str(tmp_path)
    prompts = _study(root, name)
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {"warm": _bundle(),
                                                       "plain": _bundle()})
    monkeypatch.setattr(_owner_generate, "generate", _generate)
    run_dir = tasks.run(name, prompts, root, model_provider=_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return root, [r for r in records if "instrument" not in r and "error" not in r]


def test_every_sampled_record_carries_marker_density(tmp_path, monkeypatch):
    _root, records = _run(tmp_path, monkeypatch, "density")
    by_cell = {(r["condition"], r["promptID"]): r["markerDensity"] for r in records}
    assert by_cell == {
        ("baseline", "p0"): {"warm": 0.0, "plain": 0.0},
        ("baseline", "p1"): {"warm": 0.0, "plain": 0.0},
        ("steered", "p0"): {"warm": 0.75, "plain": 0.0},
        ("steered", "p1"): {"warm": pytest.approx(1 / 3), "plain": 0.0},
    }


def test_analyze_pairs_marker_density_for_this_engines_own_run(tmp_path, monkeypatch):
    root, _records = _run(tmp_path, monkeypatch, "paired")
    out = tasks.analyze("paired", root=root, log=lambda *_: None)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["stratifyBy"] == "pooled"]
    warm = next(row for row in rows if row["endpoint"] == "warmMarkerDensity")
    # Per-item differences 0.75 and 1/3: mean 13/24.
    assert warm["n"] == "2"
    assert float(warm["deltaMean"]) == pytest.approx(13 / 24, abs=1e-6)
    plain = next(row for row in rows if row["endpoint"] == "plainMarkerDensity")
    assert float(plain["deltaMean"]) == 0.0
    with open(os.path.join(out, "outcome-coverage.json"), encoding="utf-8") as handle:
        coverage = json.load(handle)
    assert not [entry for entry in coverage["outcomes"]
                if entry["family"] == "markerDensity"
                and entry["status"] != "computed"]


def test_a_study_with_no_concept_records_an_empty_object():
    assert condition_execution.marker_density({}, "warm words") == {}


def test_marker_density_reads_each_concepts_rubric():
    rubrics = {"warm": MarkerRubric.from_markers(["warm"], ""), "plain": None}
    assert condition_execution.marker_density(rubrics, "naïve warm") == {
        "warm": 0.5, "plain": 0.0}
