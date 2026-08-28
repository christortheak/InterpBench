"""The sweep's qualitative record: ``dev-generations.jsonl``.

The field incident this closes: ``_dev_texts`` generated every dev-prompt
text per sweep cell, computed the constraint numbers from them, logged a
150-char preview each — and dropped them. An entire dose ladder's only prose
evidence was Slurm log previews. Now every dev generation is durably appended
to the run directory as it happens ({kind, concept, layer, alpha,
promptIndex, text}): baseline, grid cells, and matched-norm controls alike,
with a resume never duplicating what an earlier attempt already recorded.
The evidence packager walks the whole run directory, so the record travels in
the bundle for free — asserted here anyway, because that is the property the
incident was about. Swift twin: ``SweepDevGenerationsTests``.

No GPU, no network: extraction and generation are faked (the
``test_sweep_checkpoint`` harness).
"""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment import resume, tasks
from steerlab_server.steering.vector_store import ConceptVectors


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _workspace(root, name, *, alphas=(0.1, 0.2), selection=None,
               dev_prompts=('{"text": "Write about the town."}\n'
                            '{"text": "Describe the harbor."}\n')):
    d = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    _write(os.path.join(d, "markers.json"), json.dumps({"words": ["dread"]}))
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"), dev_prompts)
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    d = es.load_raw(name, root)
    d["sweep"] = {
        "layerFractions": [0.5], "alphas": list(alphas),
        "devPromptsFile": "prompts/dev/dev.jsonl",
        "batteryFile": "prompts/batteries/b.jsonl", "maxTokens": 16,
        "selection": selection
        or {"objective": {"metric": "markerDensity"}}}
    es.save_raw(d, root)


@contextmanager
def _fake_model(model_id, revision):
    yield SimpleNamespace(revision=revision)


CONCEPT_VECTOR = [1.0, 0.0]


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[list(CONCEPT_VECTOR)] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


def _fake_generate(model, prompt, *, injections=None, **kwargs):
    """Three distinguishable proses: baseline, concept-steered, and (for the
    matched-norm control's random vector) control."""
    if not injections:
        return "the town woke slowly 2"
    if injections[0].vector == CONCEPT_VECTOR:
        return "dread filled the town 2"
    return "a random breeze passed 2"


def _records(run_dir):
    with open(os.path.join(run_dir, tasks.DEV_GENERATIONS_FILE),
              encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(record):
    return (record["kind"], record["concept"], record["layer"],
            record["alpha"], record["promptIndex"])


def _run_sweep(root, name, **kwargs):
    return tasks.sweep(name, root, model_provider=_fake_model,
                       log=lambda *_: None, **kwargs)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate)


def test_every_dev_generation_is_persisted_per_cell(tmp_path):
    """Baseline + every grid cell × every dev prompt, full text, in
    generation order — the record a reader opens INSTEAD of the Slurm log."""
    root = str(tmp_path)
    _workspace(root, "dg")
    run_dir = _run_sweep(root, "dg")

    records = _records(run_dir)
    assert [_key(r) for r in records] == [
        ("baseline", None, -1, 0.0, 0),
        ("baseline", None, -1, 0.0, 1),
        ("cell", "fear", 2, 0.1, 0),
        ("cell", "fear", 2, 0.1, 1),
        ("cell", "fear", 2, 0.2, 0),
        ("cell", "fear", 2, 0.2, 1),
    ]
    for record in records:
        assert set(record) == {"kind", "concept", "layer", "alpha",
                               "promptIndex", "text"}
        assert record["text"] == ("the town woke slowly 2"
                                  if record["kind"] == "baseline"
                                  else "dread filled the town 2")


def test_control_generations_are_recorded_too(tmp_path):
    """The matched-norm control's texts are evidence with the same standing
    as the grid's — the control verdict is unreadable without them."""
    root = str(tmp_path)
    _workspace(root, "dgc", alphas=(0.1,), selection={
        "objective": {"metric": "markerDensity"},
        "controls": {"matchedNormRandomMargin": 0}})
    run_dir = _run_sweep(root, "dgc")

    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as handle:
        assert json.load(handle)["fear"]["winningCell"] == {
            "layer": 2, "alpha": 0.1}
    controls = [r for r in _records(run_dir) if r["kind"] == "control"]
    assert [_key(r) for r in controls] == [
        ("control", "fear", 2, 0.1, 0), ("control", "fear", 2, 0.1, 1)]
    assert all(r["text"] == "a random breeze passed 2" for r in controls)


def test_resumed_sweep_does_not_duplicate_records(tmp_path, monkeypatch):
    """A checkpointed sweep's record survives the park, and the resume
    appends only the cells it actually regenerates — one line per
    generation, ever."""
    root = str(tmp_path)
    _workspace(root, "dgr")
    # Trip after the first cell finishes: baseline (2 dev + 1 battery) +
    # cell 1 (2 dev + 1 battery) = 6 generations.
    flag = resume.CheckpointFlag()
    calls: list = []

    def tripping_generate(model, prompt, *, injections=None, **kwargs):
        calls.append(prompt)
        if len(calls) == 6:
            flag.request()
        return _fake_generate(model, prompt, injections=injections, **kwargs)
    monkeypatch.setattr(tasks, "generate", tripping_generate)

    with pytest.raises(resume.CheckpointRequested) as caught:
        _run_sweep(root, "dgr", checkpoint=flag)
    run_dir = caught.value.run_directory
    parked = _records(run_dir)
    assert [_key(r) for r in parked] == [
        ("baseline", None, -1, 0.0, 0), ("baseline", None, -1, 0.0, 1),
        ("cell", "fear", 2, 0.1, 0), ("cell", "fear", 2, 0.1, 1)]

    monkeypatch.setattr(tasks, "generate", _fake_generate)
    out_dir = _run_sweep(root, "dgr", run_directory=run_dir)
    assert out_dir == run_dir
    records = _records(run_dir)
    assert len(records) == len({_key(r) for r in records}) == 6
    assert records[:4] == parked


def test_overlong_generation_is_truncated_with_a_flag(tmp_path, monkeypatch):
    """The size rail: a decohered cell looping forever must not balloon the
    record — the capped text says it was capped."""
    root = str(tmp_path)
    _workspace(root, "dgt", alphas=(0.1,),
               dev_prompts='{"text": "Write about the town."}\n')
    monkeypatch.setattr(
        tasks, "generate",
        lambda model, prompt, *, injections=None, **kwargs:
        "dread " * 8000 if injections else "the town woke slowly 2")
    run_dir = _run_sweep(root, "dgt")

    cell = next(r for r in _records(run_dir) if r["kind"] == "cell")
    assert cell["truncated"] is True
    assert len(cell["text"]) == tasks.DEV_GENERATION_TEXT_LIMIT
    baseline = next(r for r in _records(run_dir) if r["kind"] == "baseline")
    assert "truncated" not in baseline


def test_legacy_specless_sweep_persists_its_cell_texts(tmp_path):
    """The legacy single-prompt grid scored a text per cell and dropped it —
    same gap, same record (one prompt, so promptIndex is always 0)."""
    root = str(tmp_path)
    d = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    _write(os.path.join(d, "markers.json"), json.dumps({"words": ["dread"]}))
    es.create("lg", model_id="org/m", revision="abc", root=root)
    es.attach("lg", ["fear"], root=root)
    run_dir = _run_sweep(root, "lg", layer_fractions=(0.5,), alphas=(2.0,))

    records = _records(run_dir)
    assert [_key(r) for r in records] == [("cell", "fear", 2, 2.0, 0)]
    assert records[0]["text"] == "dread filled the town 2"


def test_evidence_bundle_carries_the_record(tmp_path):
    """The packager walks the run directory wholesale; this pins the
    property the field incident was about — the prose comes home."""
    root = str(tmp_path)
    _workspace(root, "dgb")
    run_dir = _run_sweep(root, "dgb")

    meta = bundles.package_evidence(run_dir, root=root)
    run_id = os.path.basename(run_dir)
    assert f"runs/{run_id}/{tasks.DEV_GENERATIONS_FILE}" in [
        entry["path"] for entry in meta["entries"]]
