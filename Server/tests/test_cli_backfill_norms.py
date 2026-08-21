"""``steerlab-server vectors backfill-norms`` — the CLI form of the
norm-backfill route (parameter-for-parameter: vectorID/neutralCorpusPath/
outputName/redenominate/modelID/revision), with the Swift twin's path
resolution and result-JSON keys identical to the API route's. Model-free —
the loader and the activation measurement are monkeypatched exactly as in
test_norm_backfill.py; the cheap refusals must fire BEFORE any load."""

import json
import os
from types import SimpleNamespace

from steerlab_server import cli
from steerlab_server.steering import extractor, model_loader, vector_store


def _write_corpus(path, texts=("a", "b", "c", "d")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}) + "\n")
    return path


def _save_artifact(directory, name="fear", *, residual=None, model_id="org/m"):
    vectors = vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 2.0]])
    sidecar = vector_store.SteeringVectorSidecar.make(
        model_id=model_id, concept=name, stimulus_set_hash="stim-hash",
        vectors=vectors, revision="abc", extraction_method="saeFeature",
        residual_norm_per_layer=residual,
        residual_norm_source=("neutral-corpus" if residual else None))
    vector_store.save(vectors, sidecar, directory, name)
    return directory


class _FakeActs:
    def __init__(self, norms):
        self.values = [[[0.0, 0.0]]]
        self.residual_norm_per_layer = list(norms)


def _harness(tmp_path, monkeypatch, norms=(7.5, 8.25)):
    """Workspace root + fake loader/measurement. Returns the list of
    (model_id, revision, dtype, device) tuples the loader was asked for."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.setattr(extractor, "activations",
                        lambda model, texts, position: _FakeActs(list(norms)))
    loads = []

    def fake_load(model_id, revision=None, *, dtype=None, device=None):
        loads.append((model_id, revision, dtype, device))
        return SimpleNamespace(model_id=model_id, revision=revision,
                               dtype="float32")

    monkeypatch.setattr(model_loader, "load", fake_load)
    monkeypatch.setattr(model_loader, "resolve_device",
                        lambda device=None: device or "cpu")
    return loads


def _find_run_dir(tmp_path, slug):
    runs = os.path.join(str(tmp_path), "runs")
    hits = [d for d in os.listdir(runs) if d.endswith(slug)]
    assert len(hits) == 1, (slug, os.listdir(runs))
    return os.path.join(runs, hits[0])


# --- happy paths ----------------------------------------------------------------


def test_backfill_happy_path(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    orig = _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    with open(os.path.join(orig, "fear.json"), "rb") as handle:
        orig_sidecar_bytes = handle.read()
    _write_corpus(os.path.join(str(tmp_path), "prompts", "n.jsonl"))

    rc = cli.main(["vectors", "backfill-norms", "runs/legacy/fear",
                   "--corpus", "prompts/n.jsonl"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # The loader got the artifact's OWN model at its OWN pinned revision.
    assert loads == [("org/m", "abc", None, "cpu")]

    # Result JSON is key-identical to the API route's result.
    result = json.loads(captured.out)
    run_dir = _find_run_dir(tmp_path, "-backfill-norms-fear")
    assert result == {"runDirectory": run_dir,
                      "artifact": os.path.join(run_dir, "fear"),
                      "residualNormSource": "neutral-corpus",
                      "layerCount": 2}
    assert "source artifact untouched" in captured.err

    # Canonical closed-key config.json, run type norm-backfill (same as the
    # route), and the dtype the model ACTUALLY ran in.
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["runType"] == "norm-backfill"
    assert config["modelID"] == "org/m"
    assert config["revision"] == "abc"
    assert config["dtype"] == "float32"

    # The new artifact carries the measured norms; the source is untouched.
    _, sidecar = vector_store.load(run_dir, "fear")
    assert sidecar.residualNormPerLayer == [7.5, 8.25]
    with open(os.path.join(orig, "fear.json"), "rb") as handle:
        assert handle.read() == orig_sidecar_bytes


def test_backfill_default_corpus_and_bare_reference(tmp_path, monkeypatch,
                                                    capsys):
    # No --corpus → prompts/neutral/corpus.jsonl; a bare <run>/<name>
    # reference resolves under runs/ (the Swift twin's rule).
    _harness(tmp_path, monkeypatch)
    _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    _write_corpus(os.path.join(str(tmp_path), "prompts", "neutral",
                               "corpus.jsonl"))
    rc = cli.main(["vectors", "backfill-norms", "legacy/fear"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert json.loads(captured.out)["layerCount"] == 2


def test_backfill_output_name_and_redenominate(tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    orig = _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"),
                          residual=[3.0, 4.0])
    # Legacy concept-dependent denominator — the case redenomination exists for.
    sidecar_path = os.path.join(orig, "fear.json")
    with open(sidecar_path, encoding="utf-8") as handle:
        d = json.load(handle)
    d["residualNormSource"] = "extraction-stimuli"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(d, handle, sort_keys=True, indent=2)
    _write_corpus(os.path.join(str(tmp_path), "prompts", "n.jsonl"))

    rc = cli.main(["vectors", "backfill-norms", "runs/legacy/fear",
                   "--corpus", "prompts/n.jsonl",
                   "--output-name", "fear-normed", "--redenominate"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    run_dir = _find_run_dir(tmp_path, "-backfill-norms-fear-normed")
    _, sidecar = vector_store.load(run_dir, "fear-normed")
    assert sidecar.residualNormPerLayer == [7.5, 8.25]
    assert sidecar.normBackfill["replacedNormSource"] == "extraction-stimuli"
    # The original keeps its stimulus-denominated norms (immutability).
    with open(sidecar_path, encoding="utf-8") as handle:
        assert json.load(handle)["residualNormPerLayer"] == [3.0, 4.0]


# --- refusals (all before any model load) ---------------------------------------


def test_backfill_usage(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    for argv in (["vectors", "backfill-norms"],
                 ["vectors", "backfill-norms", "--corpus", "x"]):
        assert cli.main(argv) == 64
        assert "backfill-norms" in capsys.readouterr().err
    assert loads == []


def test_backfill_refuses_missing_artifact(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    rc = cli.main(["vectors", "backfill-norms", "runs/absent/fear"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no vector artifact" in err and "no extension" in err
    assert loads == []


def test_backfill_refuses_non_vector_artifact(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    run = os.path.join(str(tmp_path), "runs", "odd")
    os.makedirs(run)
    for suffix, payload in ((".json", "{}"), (".safetensors", "x")):
        with open(os.path.join(run, "thing" + suffix), "w",
                  encoding="utf-8") as handle:
            handle.write(payload)
    rc = cli.main(["vectors", "backfill-norms", "runs/odd/thing"])
    assert rc == 2
    assert "not a steering-vector artifact" in capsys.readouterr().err
    assert loads == []


def test_backfill_refuses_present_norms_without_flag(tmp_path, monkeypatch,
                                                     capsys):
    loads = _harness(tmp_path, monkeypatch)
    _save_artifact(os.path.join(str(tmp_path), "runs", "normed"),
                   residual=[3.0, 4.0])
    _write_corpus(os.path.join(str(tmp_path), "prompts", "n.jsonl"))
    rc = cli.main(["vectors", "backfill-norms", "runs/normed/fear",
                   "--corpus", "prompts/n.jsonl"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "backfill never overwrites" in err and "--redenominate" in err
    assert loads == []   # refused before paying the model load


def test_backfill_refuses_wrong_model(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    _write_corpus(os.path.join(str(tmp_path), "prompts", "n.jsonl"))
    rc = cli.main(["vectors", "backfill-norms", "runs/legacy/fear",
                   "--corpus", "prompts/n.jsonl", "--model", "org/other"])
    assert rc == 2
    assert "per-model measurement" in capsys.readouterr().err
    assert loads == []


def test_backfill_refuses_missing_corpus(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    rc = cli.main(["vectors", "backfill-norms", "runs/legacy/fear",
                   "--corpus", "prompts/absent.jsonl"])
    assert rc == 2
    assert "neutral corpus not found" in capsys.readouterr().err
    assert loads == []


def test_backfill_refuses_pathy_output_name(tmp_path, monkeypatch, capsys):
    loads = _harness(tmp_path, monkeypatch)
    _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    _write_corpus(os.path.join(str(tmp_path), "prompts", "n.jsonl"))
    rc = cli.main(["vectors", "backfill-norms", "runs/legacy/fear",
                   "--corpus", "prompts/n.jsonl", "--output-name", "a/b"])
    assert rc == 64
    assert "plain file-name component" in capsys.readouterr().err
    assert loads == []


def test_backfill_measurement_failure_exits_two(tmp_path, monkeypatch, capsys):
    # A non-finite measurement (fp16 overflow) refuses with exit 2 — the
    # module's guard surfaces through the verb.
    _harness(tmp_path, monkeypatch, norms=(1.0, float("nan")))
    _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    _write_corpus(os.path.join(str(tmp_path), "prompts", "n.jsonl"))
    rc = cli.main(["vectors", "backfill-norms", "runs/legacy/fear",
                   "--corpus", "prompts/n.jsonl"])
    assert rc == 2
    assert "non-finite" in capsys.readouterr().err
