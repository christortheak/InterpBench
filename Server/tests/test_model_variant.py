"""Model variant: injection composition (band, norm-units, neutral projection)
and artifact round-trip. No model required."""

import pytest

from steerlab_server.experiment import model_variant
from steerlab_server.steering import vector_math as vm
from steerlab_server.steering.vector_store import ConceptVectors, SteeringVectorSidecar, save


def _save_vector(tmp_path, name="french"):
    vectors = ConceptVectors(per_layer=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept=name, stimulus_set_hash="h", vectors=vectors,
        residual_norm_per_layer=[10.0, 10.0, 10.0, 10.0], residual_norm_source="neutral-corpus")
    save(vectors, sidecar, str(tmp_path), name)
    return f"{tmp_path}/{name}"


def test_variant_injection_norm_units(tmp_path):
    vid = _save_vector(tmp_path)
    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/m", band_width=1, alpha_in_norm_units=True,
        injections=[{"concept": "french", "vectorArtifactID": vid, "layer": 1, "alpha": 2.0}])
    cells = model_variant.variant_injections(variant)
    assert len(cells) == 1 and cells[0].layer == 1
    # alpha_eff = alpha * residual / ||v|| = 2 * 10 / 2 = 10
    assert cells[0].alpha == pytest.approx(10.0)


def test_variant_band_width_expands_layers(tmp_path):
    vid = _save_vector(tmp_path)
    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/m", band_width=3, alpha_in_norm_units=False,
        injections=[{"concept": "french", "vectorArtifactID": vid, "layer": 1, "alpha": 2.0}])
    cells = model_variant.variant_injections(variant)
    assert sorted(c.layer for c in cells) == [0, 1, 2]   # center 1 ± 1
    assert all(c.alpha == 2.0 for c in cells)


def _save_normless_vector(tmp_path, name="nonorm", layers=2):
    v = ConceptVectors(per_layer=[[float(i + 1), 0.0] for i in range(layers)])
    save(v, SteeringVectorSidecar.make(model_id="m", concept="c", stimulus_set_hash="h", vectors=v),
         str(tmp_path), name)
    return f"{tmp_path}/{name}"


def test_variant_norm_units_without_norms_raises(tmp_path):
    vid = _save_normless_vector(tmp_path)
    variant = model_variant.ModelVariant(
        name="v", base_model_id="m", alpha_in_norm_units=True,
        injections=[{"concept": "c", "vectorArtifactID": vid, "layer": 0, "alpha": 1.0}])
    with pytest.raises(ValueError):
        model_variant.variant_injections(variant)


def test_norm_units_refusal_names_both_remedies(tmp_path):
    # The refusal is a dead end unless it names BOTH ways out: backfill the
    # denominator, or switch the variant to raw alpha (field report
    # 2026-08-06 — "no cryptic blockers").
    vid = _save_normless_vector(tmp_path)
    variant = model_variant.ModelVariant(
        name="v", base_model_id="m", alpha_in_norm_units=True,
        injections=[{"concept": "c", "vectorArtifactID": vid, "layer": 0, "alpha": 1.0}])
    with pytest.raises(ValueError) as excinfo:
        model_variant.variant_injections(variant)
    message = str(excinfo.value)
    assert "backfill-norms" in message      # POST /api/vectors/backfill-norms
    assert "raw alpha" in message


def test_variant_ablation_without_norms_composes(tmp_path):
    # Ablation-without-norms round trip (field report 2026-08-06): λ is a
    # plain fraction of what is present — it never converts through the
    # residual-norm denominator, so an ablating injection on a norms-less
    # vector (J-lens/reader-derived directions) must compose under
    # alpha_in_norm_units=True instead of hitting the missing-norms refusal.
    vid = _save_normless_vector(tmp_path, name="lens", layers=3)
    variant = model_variant.ModelVariant(
        name="v", base_model_id="m", alpha_in_norm_units=True,
        injections=[{"concept": "c", "vectorArtifactID": vid, "layer": 1,
                     "alpha": 0.75, "mode": "ablate"}])
    cells = model_variant.variant_injections(variant)
    # Whole-network coverage, λ passed through untouched, mode preserved.
    assert sorted(c.layer for c in cells) == [0, 1, 2]
    assert all(c.mode == "ablate" for c in cells)
    assert all(c.alpha == pytest.approx(0.75) for c in cells)


def test_from_dict_absent_alpha_units_reads_raw():
    # Cross-engine decode rule (mirrors Swift ModelVariantArtifact's decoder):
    # a spec that OMITS alphaInNormUnits is a legacy raw-α artifact — the
    # conservative literal α·v reading. Same bytes, same injection semantics
    # on both engines.
    variant = model_variant.ModelVariant.from_dict({"baseModelID": "org/m"})
    assert variant.alpha_in_norm_units is False


def test_from_dict_explicit_alpha_units_honored():
    for flag in (True, False):
        variant = model_variant.ModelVariant.from_dict(
            {"baseModelID": "org/m", "alphaInNormUnits": flag})
        assert variant.alpha_in_norm_units is flag


def test_to_dict_always_stamps_alpha_units():
    # The encoder writes the key explicitly — that stamping is what makes the
    # absent-key decode default safe for artifacts this engine produces.
    d = model_variant.ModelVariant(name="v", base_model_id="m").to_dict()
    assert d["alphaInNormUnits"] is True  # programmatic default: norm units
    # And an explicit raw spec round-trips as raw (never reinterpreted).
    rt = model_variant.ModelVariant.from_dict(
        {"baseModelID": "m", "alphaInNormUnits": False}).to_dict()
    assert rt["alphaInNormUnits"] is False


def test_apply_adapter_none_for_pure_steering():
    # Pure-steering variants (no adapters) skip adapter loading entirely.
    variant = model_variant.ModelVariant(name="v", base_model_id="m", injections=[])
    assert model_variant.apply_adapter(object(), variant) is None
    model_variant.remove_adapter(object(), None)  # no-op


def test_apply_adapter_missing_dir_raises(tmp_path):
    variant = model_variant.ModelVariant(
        name="v", base_model_id="m",
        adapters=[{"adapterDirectory": str(tmp_path / "nope")}])
    with pytest.raises(ValueError):
        model_variant.apply_adapter(object(), variant)


def test_save_and_reload_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    variant = model_variant.ModelVariant(
        name="myvar", base_model_id="org/m", temperature=0.7, prompt_mode="rawCompletion",
        injections=[{"concept": "fear", "vectorArtifactID": "x/fear", "layer": 5, "alpha": 1.0}])
    info = model_variant.save_variant(variant)
    reloaded = model_variant.ModelVariant.from_file(info["path"])
    assert reloaded.name == "myvar" and reloaded.temperature == 0.7
    assert reloaded.prompt_mode == "rawCompletion"
    assert model_variant.list_variants(str(tmp_path))[0]["name"] == "myvar"


# --- injection-time artifact guards (finiteness + substrate) -----------------

def _write_sidecar_patch(tmp_path, name, **patch):
    import json, os
    path = os.path.join(str(tmp_path), f"{name}.json")
    with open(path, encoding="utf-8") as handle:
        d = json.load(handle)
    for key, value in patch.items():
        if value is None:
            d.pop(key, None)
        else:
            d[key] = value
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(d, handle)


def _variant_for(vid, **kwargs):
    return model_variant.ModelVariant(
        name="v", base_model_id="org/m", band_width=1, alpha_in_norm_units=False,
        injections=[{"concept": "french", "vectorArtifactID": vid,
                     "layer": 1, "alpha": 2.0}], **kwargs)


def test_variant_injections_refuse_foreign_substrate(tmp_path):
    vid = _save_vector(tmp_path)
    _write_sidecar_patch(tmp_path, "french", substrate="swift-mlx")
    with pytest.raises(ValueError) as err:
        model_variant.variant_injections(_variant_for(vid))
    msg = str(err.value)
    assert "swift-mlx" in msg and "python-hf-transformers" in msg
    assert "re-extract" in msg


def test_variant_injections_accept_native_and_unstamped_legacy(tmp_path):
    vid = _save_vector(tmp_path)
    # Native stamp (written by save()) passes.
    assert len(model_variant.variant_injections(_variant_for(vid))) == 1
    # Unstamped legacy (pre-stamp artifact) passes — absent is never guessed.
    _write_sidecar_patch(tmp_path, "french", substrate=None)
    assert len(model_variant.variant_injections(_variant_for(vid))) == 1


def test_variant_injections_refuse_nonfinite_artifact(tmp_path):
    import json, math, os
    import numpy as np
    from safetensors.numpy import save_file
    # Poisoned artifact written directly (bypasses save()'s guard) — the
    # legacy pre-guard class of artifact.
    save_file({"layer_0": np.asarray([1.0, 0.0], dtype=np.float32),
               "layer_1": np.asarray([math.inf, 0.0], dtype=np.float32)},
              os.path.join(str(tmp_path), "poison.safetensors"))
    with open(os.path.join(str(tmp_path), "poison.json"), "w", encoding="utf-8") as h:
        json.dump({"modelID": "org/m", "concept": "french", "stimulusSetHash": "h",
                   "layerCount": 2, "hiddenSize": 2, "normsPerLayer": [1.0, 1.0],
                   "extractionDate": "2026-01-01T00:00:00Z"}, h)
    with pytest.raises(ValueError, match="non-finite"):
        model_variant.variant_injections(_variant_for(f"{tmp_path}/poison"))


# --- foreign-machine reference rebase (paths.resolve_artifact) ------------------

def test_variant_injections_rebase_foreign_absolute_reference(tmp_path, monkeypatch):
    """An agent promoted on another machine records that machine's absolute
    path (`/Users/…/Workspace/runs/<run>/<leaf>`). When the artifact is
    actually present under THIS workspace's runs/, the reference rebases and
    the injection loads — the 2026-08-04 cluster failure mode, fixed."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    run_dir = tmp_path / "runs" / "20260803T225631751-exp-sweep"
    run_dir.mkdir(parents=True)
    _save_vector(run_dir, "fear")
    foreign = "/Users/someone/SteerLab Workspace/runs/20260803T225631751-exp-sweep/fear"
    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/m", band_width=1, alpha_in_norm_units=False,
        injections=[{"concept": "fear", "vectorArtifactID": foreign,
                     "layer": 1, "alpha": 2.0}])
    cells = model_variant.variant_injections(variant)
    assert len(cells) == 1 and cells[0].layer == 1


def test_missing_artifacts_reports_unresolvable_and_accepts_rebased(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    _save_vector(run_dir, "fear")
    present = model_variant.ModelVariant(
        name="ok", base_model_id="org/m",
        injections=[{"concept": "fear",
                     "vectorArtifactID": "/elsewhere/runs/r1/fear",
                     "layer": 1, "alpha": 1.0}])
    assert model_variant.missing_artifacts(present) == []
    absent = model_variant.ModelVariant(
        name="broken", base_model_id="org/m",
        injections=[{"concept": "fear",
                     "vectorArtifactID": "/elsewhere/runs/nope/fear",
                     "layer": 1, "alpha": 1.0}])
    missing = model_variant.missing_artifacts(absent)
    assert [m["kind"] for m in missing] == ["vector"]
    assert missing[0]["reference"] == "/elsewhere/runs/nope/fear"
