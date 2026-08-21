"""OptVec gradient survey (WP-S4a): the α→0 probe, the dose ladder, the
linearity summary, the survey run directory, and the mint round-trip.

The load-bearing test here is the finite-difference property: the probe's
gradient must BE the derivative of the margin with respect to an additive delta
at the answer position, not merely a plausible vector — everything the survey
reports, and the ``cosineToGradient`` stamp the training driver adds for
per-item runs, is a claim about that derivative.

All CPU, all on the same tiny in-memory Llama the other OptVec suites use.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import optvec_eval, optvec_gradient
from steerlab_server.experiment.optvec_gradient import (
    OptVecGradientConfig, OptVecGradientConfigError, OptVecGradientDataError)
from steerlab_server.experiment.optvec_train import DatasetRef
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.steering.trainable_injector import (AdditiveDeltaProbe,
                                                         TrainableVectorInjector)
from tests.test_optvec_train import (HIDDEN, LAYERS, _target_rows,
                                     _tiny_steered_model, _write_rows)

LAYER = 2
ALPHA = 4.0


# ------------------------------------------------------------------ fixtures


def _config(tmp_path, *, rows=None, **overrides) -> OptVecGradientConfig:
    ref = overrides.pop("target_train", None) or _write_rows(
        tmp_path / "grad-train.jsonl", rows or _target_rows("g", 3))
    kwargs = dict(model_id="test/tiny", revision="rev-tiny", layer=LAYER,
                  name="probe", target_train=ref, alpha_absolute=ALPHA,
                  dose_ladder=(0.5, 1.0, 2.0), prompt_mode=RAW_COMPLETION)
    kwargs.update(overrides)
    return OptVecGradientConfig(**kwargs)


def _prepared_item(model, config):
    """One prepared item plus its baseline record and contrast index."""
    from steerlab_server.experiment import optvec_train

    rows = optvec_train.load_dataset(config.target_train, "targetTrain")
    items = optvec_train.prepare_items(model, rows, role="target",
                                       split="train", config=config,
                                       declared="targetTrain")
    baselines = optvec_train.baseline_prepass(model, items, config)
    item = items[0]
    return item, baselines, optvec_gradient.contrast_index_for(item, baselines)


# ------------------------------------------------------------- the property


def test_probe_gradient_matches_finite_differences(tmp_path):
    """The probe's gradient IS the derivative: along several random
    directions, ``g·r`` must match a central finite difference of the margin
    under a delta of ``±ε·r`` at the same position."""
    model = _tiny_steered_model()
    config = _config(tmp_path)
    optvec_gradient.freeze_model(model)
    item, baselines, contrast = _prepared_item(model, config)
    assert contrast is not None

    gradient, margin = optvec_gradient.item_gradient(
        model, item, contrast, layer=LAYER)
    assert gradient.shape == (HIDDEN,)
    assert float(torch.linalg.vector_norm(gradient)) > 0

    # The zero-delta forward is the unsteered forward: the probe's margin must
    # equal the margin with no probe armed at all.
    assert optvec_gradient.margin_at_delta(
        model, item, contrast, torch.zeros(HIDDEN), layer=LAYER) == \
        pytest.approx(margin, abs=1e-6)

    generator = torch.Generator().manual_seed(3)
    epsilon = 1e-2
    for _ in range(4):
        r = torch.randn(HIDDEN, generator=generator, dtype=torch.float32)
        r = r / r.norm()
        plus = optvec_gradient.margin_at_delta(model, item, contrast,
                                               epsilon * r, layer=LAYER)
        minus = optvec_gradient.margin_at_delta(model, item, contrast,
                                                -epsilon * r, layer=LAYER)
        finite = (plus - minus) / (2 * epsilon)
        analytic = float(torch.dot(gradient, r))
        assert finite == pytest.approx(analytic, rel=2e-2, abs=1e-3), \
            (finite, analytic)


def test_zero_delta_probe_leaves_the_forward_bit_identical(tmp_path):
    """``h + 0`` is exactly ``h``: arming the probe must change nothing about
    what the model computes, or the 'baseline and gradient from one pass'
    design is a lie."""
    model = _tiny_steered_model()
    ids = torch.randint(1, 255, (2, 5))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        bare = model.model(input_ids=ids, attention_mask=mask,
                           use_cache=False).logits.clone()
    probe = AdditiveDeltaProbe(layer=LAYER, hidden_size=HIDDEN)
    probe.set_batch(answer_positions=torch.tensor([4, 4]), attention_mask=mask)
    with torch.no_grad():
        with model.hooked.session([probe]):
            armed = model.model(input_ids=ids, attention_mask=mask,
                                use_cache=False).logits
    assert torch.equal(bare, armed)


def test_the_shared_position_machinery_still_serves_the_injector():
    """The refactor that gave the probe its positions must not have moved the
    injector: same mask, same refusals."""
    injector = TrainableVectorInjector(layer=LAYER, hidden_size=HIDDEN,
                                       alpha_absolute=2.0)
    h = torch.zeros((2, 3, HIDDEN))
    with pytest.raises(Exception):
        injector.apply(h, LAYER, 0)          # no batch declared
    injector.set_batch(answer_positions=torch.tensor([2, 1]))
    out = injector.apply(h, LAYER, 0).detach()
    assert float(out[0, 2].norm()) == pytest.approx(2.0, rel=1e-5)
    assert float(out[0, 0].norm()) == 0.0
    assert float(out[1, 1].norm()) == pytest.approx(2.0, rel=1e-5)
    # And an untouched layer is returned unchanged.
    assert torch.equal(injector.apply(h, LAYER + 1, 0), h)


def test_mean_margin_gradient_is_the_mean_of_item_gradients(tmp_path):
    from steerlab_server.experiment import optvec_train

    model = _tiny_steered_model()
    config = _config(tmp_path)
    optvec_gradient.freeze_model(model)
    rows = optvec_train.load_dataset(config.target_train, "targetTrain")
    items = optvec_train.prepare_items(model, rows, role="target",
                                       split="train", config=config,
                                       declared="targetTrain")
    baselines = optvec_train.baseline_prepass(model, items, config)
    per_item = [optvec_gradient.item_gradient(
        model, item, optvec_gradient.contrast_index_for(item, baselines),
        layer=LAYER)[0] for item in items]
    expected = optvec_gradient.unit(sum(per_item) / len(per_item))
    got = optvec_gradient.mean_margin_gradient(model, items, baselines,
                                               layer=LAYER)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert sum(x * x for x in got) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------- arithmetic


def test_linearity_summary_formula_on_hand_built_curves():
    """predicted(a) = m0 + a·‖g‖; deviation = observed − predicted;
    maxRelativeDeviation divides by the largest PROMISED movement."""
    doses = [{"alphaFraction": 0.5, "alphaAbsolute": 1.0, "margin": 2.0},
             {"alphaFraction": 1.0, "alphaAbsolute": 2.0, "margin": 4.5}]
    summary = optvec_gradient.linearity_summary(0.0, 2.0, doses)
    assert [row["predictedMargin"] for row in summary["perDose"]] == [2.0, 4.0]
    assert [row["deviation"] for row in summary["perDose"]] == \
        pytest.approx([0.0, 0.5])
    assert summary["maxAbsDeviation"] == pytest.approx(0.5)
    assert summary["maxRelativeDeviation"] == pytest.approx(0.5 / 4.0)
    # A perfectly linear curve has zero deviation…
    linear = optvec_gradient.linearity_summary(
        1.0, 3.0, [{"alphaFraction": 1.0, "alphaAbsolute": 2.0, "margin": 7.0}])
    assert linear["maxAbsDeviation"] == pytest.approx(0.0, abs=1e-12)
    assert linear["maxRelativeDeviation"] == pytest.approx(0.0, abs=1e-12)
    # …and a zero-gradient promise leaves the relative reading undefined
    # rather than dividing by zero.
    flat = optvec_gradient.linearity_summary(
        1.0, 0.0, [{"alphaFraction": 1.0, "alphaAbsolute": 2.0, "margin": 1.5}])
    assert flat["maxRelativeDeviation"] is None
    assert flat["maxAbsDeviation"] == pytest.approx(0.5)


def test_unit_refuses_to_normalize_a_zero_vector():
    assert optvec_gradient.unit([0.0, 0.0]) is None
    assert optvec_gradient.unit([3.0, 4.0]) == pytest.approx([0.6, 0.8])


# --------------------------------------------------------------- config rules


def test_config_refuses_all_positions_unknown_keys_and_selection_splits(tmp_path):
    ref = _write_rows(tmp_path / "t.jsonl", _target_rows("t", 2))
    base = {"modelID": "test/tiny", "layer": 1, "alphaAbsolute": 1.0,
            "datasets": {"targetTrain": ref.to_dict()}}
    assert OptVecGradientConfig.from_dict(base).layer == 1

    with pytest.raises(OptVecGradientConfigError) as exc:
        OptVecGradientConfig.from_dict({**base, "positionMode": "all"})
    assert "from_response" in str(exc.value)

    with pytest.raises(OptVecGradientConfigError):
        OptVecGradientConfig.from_dict({**base, "doseLader": [1.0]})
    with pytest.raises(OptVecGradientConfigError):
        OptVecGradientConfig.from_dict({**base, "doseLadder": []})
    with pytest.raises(OptVecGradientConfigError):
        OptVecGradientConfig.from_dict({**base, "doseLadder": [0.5, 0.0]})
    with pytest.raises(OptVecGradientConfigError) as exc:
        OptVecGradientConfig.from_dict(
            {**base, "datasets": {"targetTrain": ref.to_dict(),
                                  "targetVal": ref.to_dict()}})
    assert "TRAIN split only" in str(exc.value)
    # α still needs a denominator, exactly as in training.
    without_alpha = {k: v for k, v in base.items() if k != "alphaAbsolute"}
    with pytest.raises(OptVecGradientConfigError):
        OptVecGradientConfig.from_dict(without_alpha)
    # The canonical payload carries itemFilter only when it is used.
    assert "itemFilter" not in OptVecGradientConfig.from_dict(base).to_dict()
    filtered = OptVecGradientConfig.from_dict({**base, "itemFilter": ["t-0"]})
    assert filtered.to_dict()["itemFilter"] == ["t-0"]


def test_config_round_trips_through_a_file(tmp_path):
    ref = _write_rows(tmp_path / "t.jsonl", _target_rows("t", 2))
    path = tmp_path / "gradient.json"
    payload = {"modelID": "test/tiny", "layer": 2, "alphaAbsolute": 3.0,
               "doseLadder": [0.5, 1.0],
               "datasets": {"targetTrain": ref.to_dict()}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = optvec_gradient.load_config(str(path))
    assert config.dose_ladder == (0.5, 1.0)
    assert optvec_gradient.OptVecGradientConfig.from_dict(
        config.to_dict()).to_dict() == config.to_dict()


# -------------------------------------------------------------- the survey


def _run_survey(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    model = _tiny_steered_model()
    config = _config(tmp_path, **overrides)
    result = optvec_gradient.survey(config, model=model)
    return model, config, result


def test_survey_writes_one_immutable_run_directory(tmp_path, monkeypatch):
    model, config, result = _run_survey(tmp_path, monkeypatch)
    run_dir = result["runDirectory"]

    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-gradient"
    assert config_json["modelID"] == "test/tiny"
    assert config_json["revision"] == "rev-tiny"
    notes = config_json["notes"]
    assert notes["stage"] == "complete"
    assert notes["claim"] == "localSensitivity"
    assert notes["itemCount"] == 3 and notes["directionCount"] == 3

    # ONE survey file + ONE tensor file, not one artifact per item.
    written = sorted(os.listdir(run_dir))
    assert written == ["config.json", "gradient-survey.json",
                       "gradients.safetensors"]

    readout, directions = optvec_gradient.load_survey(run_dir)
    assert readout["runType"] == "optvec-gradient"
    assert readout["split"] == "train"
    assert readout["layer"] == LAYER
    assert readout["alphaAbsolute"] == pytest.approx(ALPHA)
    assert readout["model"]["numLayers"] == LAYERS
    assert readout["model"]["hiddenSize"] == HIDDEN
    assert readout["config"] == config.to_dict()
    assert set(directions) == {"g-0", "g-1", "g-2"}
    for values in directions.values():
        assert sum(x * x for x in values) == pytest.approx(1.0, rel=1e-5)

    for record in readout["items"]:
        assert record["contrast"] in record["options"]
        assert record["gradientNorm"] > 0
        assert len(record["baselineProbabilities"]) == 2
        # The stepped instrument and the teacher-forced probe pass compute the
        # same margin two ways; they must agree.
        assert record["probePassMargin"] == pytest.approx(
            record["baselineMargin"], abs=1e-4)
        assert set(record["baselineLogOdds"]) == set(record["options"])
        assert [d["alphaFraction"] for d in record["doses"]] == [0.5, 1.0, 2.0]
        assert all(d["deltaLogOddsTarget"] == pytest.approx(
            d["logOddsTarget"] - record["baselineLogOdds"][record["target"]])
            for d in record["doses"])
        assert [d["alphaAbsolute"] for d in record["doses"]] == \
            pytest.approx([0.5 * ALPHA, ALPHA, 2 * ALPHA])
        linearity = record["linearity"]
        assert linearity["maxAbsDeviation"] >= 0
        assert "predicted(a) = margin(0) + a·‖g‖" in linearity["formula"]
        # The recorded per-dose predictions follow the stated formula.
        for row in linearity["perDose"]:
            assert row["predictedMargin"] == pytest.approx(
                record["baselineMargin"]
                + row["alphaAbsolute"] * record["gradientNorm"])


def test_survey_honors_the_item_filter(tmp_path, monkeypatch):
    _model, _config, result = _run_survey(tmp_path, monkeypatch,
                                          item_filter=["g-2"])
    readout, directions = optvec_gradient.load_survey(result["runDirectory"])
    assert [r["id"] for r in readout["items"]] == ["g-2"]
    assert set(directions) == {"g-2"}
    assert readout["datasets"]["itemFilter"] == ["g-2"]
    assert readout["datasets"]["itemFilterApplication"]["selectedCount"] == 1
    # The file's hash pin is untouched — the filter is config data, not an
    # edit to the pinned bytes: the recorded hash is still the WHOLE file's.
    import hashlib
    whole_file = hashlib.sha256(
        open(tmp_path / "grad-train.jsonl", "rb").read()).hexdigest()
    assert readout["datasets"]["files"]["targetTrain"]["sha256"] == whole_file


def test_survey_refuses_an_unknown_filter_id(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _config(tmp_path, item_filter=["g-0", "nope"])
    with pytest.raises(OptVecGradientDataError) as exc:
        optvec_gradient.survey(config, model=model)
    assert "nope" in str(exc.value)


def test_survey_refuses_a_layer_the_model_does_not_have(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _config(tmp_path, layer=LAYERS + 3)
    with pytest.raises(OptVecGradientConfigError):
        optvec_gradient.survey(config, model=model)


# ----------------------------------------------------------------- the mint


def test_mint_round_trips_through_require_optvec(tmp_path, monkeypatch):
    model, _config, result = _run_survey(tmp_path, monkeypatch)
    minted = optvec_gradient.mint(result["runDirectory"], "g-1")

    reference = minted["vectorArtifactID"]
    artifact = optvec_eval.load_artifact(reference)
    target = optvec_eval.require_optvec(artifact)
    assert target.layer == LAYER
    assert target.alpha_absolute == pytest.approx(ALPHA)
    # The eval verb's model check passes too — same model, same revision.
    assert optvec_eval.verify_model(target, model)["revisionChecked"] is True

    sidecar = json.load(open(f"{reference}.json"))
    assert sidecar["extractionMethod"] == "optvec"
    assert sidecar["stimulusSetHash"].startswith("optvec-gradient:")
    assert sidecar["layerCount"] == LAYERS and sidecar["hiddenSize"] == HIDDEN
    # Born without a residual-norm denominator, like every other derived
    # direction: backfill comes before any alphaInNormUnits condition.
    assert "residualNormPerLayer" not in sidecar
    block = sidecar["optvec"]
    assert block["method"] == "gradientDirection"
    assert block["optimized"] is False
    assert block["claim"] == "localSensitivity"
    assert block["sourceItem"] == "g-1"
    # …and the flat per-item marker the geometry readouts group on.
    from steerlab_server.experiment import optvec_geometry
    assert optvec_geometry.item_of(block) == "g-1"
    assert block["sourceSurveyRun"] == os.path.basename(
        result["runDirectory"])
    assert block["linearity"]["maxAbsDeviation"] >= 0
    assert block["gradientNorm"] > 0

    vectors = artifact.vectors
    assert vectors.norm(LAYER) == pytest.approx(ALPHA, rel=1e-5)
    assert all(vectors.norm(i) == 0.0 for i in range(LAYERS) if i != LAYER)

    # Its own immutable run directory, with the canonical stamp.
    from tests.test_run_config import CONTRACT_KEYS
    mint_config = json.load(open(os.path.join(minted["runDirectory"],
                                              "config.json")))
    assert sorted(mint_config.keys()) == CONTRACT_KEYS
    assert mint_config["runType"] == "optvec-gradient-mint"
    # The survey run itself was not written into.
    assert sorted(os.listdir(result["runDirectory"])) == [
        "config.json", "gradient-survey.json", "gradients.safetensors"]


def test_dose_ladder_margins_reproduce_the_deployed_injection_path(
        tmp_path, monkeypatch):
    """Parity spot check: scoring the MINTED artifact through the eval verb's
    own dose path must reproduce the margin the survey recorded at that dose."""
    from steerlab_server.experiment import logprob

    model, config, result = _run_survey(tmp_path, monkeypatch)
    record = next(r for r in result["items"] if r["id"] == "g-0")
    dose = record["doses"][1]                     # the 1.0× rung
    minted = optvec_gradient.mint(result["runDirectory"], "g-0")

    artifact = optvec_eval.load_artifact(minted["vectorArtifactID"])
    target = optvec_eval.require_optvec(artifact)
    injection = optvec_eval.cell_injection(target, dose["alphaFraction"])
    scored = logprob.score_options(
        model, next(r for r in _target_rows("g", 3)
                    if r["id"] == "g-0")["prompt"],
        list(record["options"]), injections=[injection],
        prompt_mode=RAW_COMPLETION)
    logprobs = {s.option: s.logprob for s in scored.options}
    margin = logprobs[record["target"]] - logprobs[record["contrast"]]
    assert margin == pytest.approx(dose["margin"], rel=1e-4, abs=1e-4)


def test_mint_refuses_an_unsurveyed_item(tmp_path, monkeypatch):
    _model, _config, result = _run_survey(tmp_path, monkeypatch)
    with pytest.raises(OptVecGradientDataError) as exc:
        optvec_gradient.mint(result["runDirectory"], "not-an-item")
    assert "not-an-item" in str(exc.value)
    with pytest.raises(OptVecGradientDataError):
        optvec_gradient.mint(str(tmp_path), "g-0")


def test_mint_names_are_single_path_components(tmp_path, monkeypatch):
    _model, _config, result = _run_survey(tmp_path, monkeypatch)
    with pytest.raises(OptVecGradientDataError):
        optvec_gradient.mint(result["runDirectory"], "g-0",
                             name="nested/name")
