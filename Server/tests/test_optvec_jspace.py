"""OptVec propagated J-space decomposition (WP8): the exact decomposition at
the injection layer, the linearity identity, the teacher-forced pairing, the
matched-norm null discipline, the family tables, and the mandatory tier stamp.

All CPU, all on the same tiny in-memory Llama the other OptVec suites use, with
a hand-built lens record standing in for an imported one (a ``StubBackend``
checkpoint converted through the real ``importer.convert_to_per_layer``, so the
store, the loader and the readout are the production ones — only the numbers
are synthetic). No downloads, no GPU, seconds.
"""

import hashlib
import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import optvec_jspace, paths
from steerlab_server.experiment.optvec_eval import FileRef
from steerlab_server.experiment.optvec_jspace import (OptVecJSpaceConfig,
                                                      OptVecJSpaceConfigError,
                                                      OptVecJSpaceError)
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.jlens import backend, importer, lens_store, schemas
from tests.test_optvec_eval import (HIDDEN, LAYERS, OPTVEC_LAYER,
                                    _tiny_steered_model, write_optvec_artifact)

LENS_ID = "test-lens"


# ------------------------------------------------------------------ fixtures


class _RandomLensSource:
    """A lens whose ``J_l`` is a seeded DENSE matrix, not a scaled identity.

    ``StubBackend`` uses ``eye(d)·(l+1)``, which is ideal for catching a
    mis-mapped layer and useless for catching a readout that forgot to
    transport at all — with a diagonal J, ``J x`` and ``x`` point the same way.
    Where a test's claim is about transport, it uses this instead.
    """

    def __init__(self, *, d_model=HIDDEN, source_layers=(0, 1, 2), seed=7):
        generator = torch.Generator().manual_seed(seed)
        self._d_model = d_model
        self._source_layers = sorted(source_layers)
        self._j = {layer: torch.randn(d_model, d_model, generator=generator,
                                      dtype=torch.float32) * 0.2
                   for layer in self._source_layers}

    @property
    def d_model(self):
        return self._d_model

    @property
    def n_prompts(self):
        return 5

    @property
    def source_layers(self):
        return list(self._source_layers)

    def jacobian(self, layer):
        return self._j[layer]


def _write_lens(tmp_path, *, source=None, model_id="test/tiny",
                lens_id=LENS_ID, root=None):
    """Save a lens record into a workspace lens library.

    The bytes go through the production converter and the production store, so
    ``lens_store.load_layer`` and ``LensReadout.build`` are exercised for real.
    """
    root = root or str(tmp_path / "ws")
    source = source or backend.StubBackend(d_model=HIDDEN,
                                           source_layers=[0, 1, 2])
    directory = paths.jlens_lens_directory(lens_id, root)
    os.makedirs(directory, exist_ok=True)
    converted = importer.convert_to_per_layer(source, directory)
    record = schemas.JLensRecord(
        lensID=lens_id,
        source=schemas.SourceRef(repo="test/lens", folder="f",
                                 tensorFile="t.pt", configFile="c.yaml",
                                 tensorSHA256="deadbeef"),
        fit=schemas.FitProvenance(modelID=model_id, dtype="float32"),
        sourceLayers=source.source_layers, dModel=source.d_model,
        targetLayer=source.source_layers[-1] + 1, nPrompts=source.n_prompts,
        converted=converted)
    lens_store.save(record, root)
    return record, root


def _write_probe(path, count=3) -> FileRef:
    rows = [{"id": f"p-{i}", "prompt": f"case number {i} the ruling is",
             "options": ["alpha", "beta"], "target": "alpha"}
            for i in range(count)]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return FileRef(path=str(path), sha256=digest)


def _config(tmp_path, artifacts, **overrides) -> OptVecJSpaceConfig:
    kwargs = dict(vector_artifacts=list(artifacts), lens_id=LENS_ID,
                  probe_items=_write_probe(tmp_path / "probe.jsonl"),
                  microbatch_size=2, top_k=3, prompt_mode=RAW_COMPLETION)
    kwargs.update(overrides)
    return OptVecJSpaceConfig(**kwargs)


def _probe_items(model, tmp_path, count=3):
    """Rendered probe items, through the production loader + renderer."""
    from steerlab_server.experiment.optvec_eval import (load_dataset,
                                                        prepare_items)

    ref = _write_probe(tmp_path / "probe-items.jsonl", count)
    rows = load_dataset(ref, "probeItems")
    config = _config(tmp_path, ["runs/x/toy"])
    return optvec_jspace.prepare_probe_items(
        model, prepare_items(rows, role="probe", dataset="probeItems"), config)


# ------------------------------------------------------------ config strictness


def test_config_refuses_unknown_keys_and_requires_lens_and_probe_items(tmp_path):
    probe = _write_probe(tmp_path / "probe.jsonl").to_dict()
    base = {"vectorArtifacts": ["runs/x/toy"], "lensID": LENS_ID,
            "probeItems": probe}
    assert OptVecJSpaceConfig.from_dict(base).lens_id == LENS_ID

    with pytest.raises(OptVecJSpaceConfigError) as exc:
        OptVecJSpaceConfig.from_dict({**base, "observationLayer": [4]})
    assert "observationLayer" in str(exc.value)

    with pytest.raises(OptVecJSpaceConfigError) as exc:
        OptVecJSpaceConfig.from_dict(
            {k: v for k, v in base.items() if k != "lensID"})
    assert "lensID" in str(exc.value)

    with pytest.raises(OptVecJSpaceConfigError) as exc:
        OptVecJSpaceConfig.from_dict(
            {k: v for k, v in base.items() if k != "probeItems"})
    assert "probeItems" in str(exc.value)

    with pytest.raises(OptVecJSpaceConfigError):
        OptVecJSpaceConfig.from_dict({**base, "vectorArtifacts": []})
    # The same artifact twice would put a duplicated row in the family matrix.
    with pytest.raises(OptVecJSpaceConfigError):
        OptVecJSpaceConfig.from_dict(
            {**base, "vectorArtifacts": ["runs/x/toy", "runs/x/toy"]})
    # α=0 makes the steered pass the baseline; every delta would be zero.
    with pytest.raises(OptVecJSpaceConfigError):
        OptVecJSpaceConfig.from_dict({**base, "alphaMultiple": 0.0})
    with pytest.raises(OptVecJSpaceConfigError):
        OptVecJSpaceConfig.from_dict({**base, "observationLayers": []})


def test_config_defaults_and_round_trip(tmp_path):
    probe = _write_probe(tmp_path / "probe.jsonl").to_dict()
    config = OptVecJSpaceConfig.from_dict(
        {"vectorArtifacts": ["runs/x/toy"], "lensID": LENS_ID,
         "probeItems": probe, "observationLayers": [8, 4, 4]})
    assert config.alpha_multiple == 1.0
    assert config.null_seed == optvec_jspace.NULL_SEED
    assert config.top_k == optvec_jspace.DEFAULT_TOP_K
    # Deduplicated and sorted, so two configs naming the same layers are one
    # configuration rather than two orderings of it.
    assert config.observation_layers == [4, 8]
    payload = config.to_dict()
    assert payload["probeItems"] == probe
    assert payload["lensID"] == LENS_ID
    assert OptVecJSpaceConfig.from_dict(payload).to_dict() == payload


def test_config_file_round_trips(tmp_path):
    probe = _write_probe(tmp_path / "probe.jsonl").to_dict()
    path = tmp_path / "jspace.json"
    path.write_text(json.dumps({"vectorArtifacts": ["runs/x/toy"],
                                "lensID": LENS_ID, "probeItems": probe,
                                "alphaMultiple": 2.0}), encoding="utf-8")
    config = optvec_jspace.load_config(str(path))
    assert config.alpha_multiple == 2.0


# ------------------------------------------------------------------- the lens


def test_missing_lens_refuses_and_names_both_acquisition_verbs(tmp_path):
    with pytest.raises(OptVecJSpaceError) as exc:
        optvec_jspace.resolve_lens("no-such-lens", str(tmp_path / "ws"))
    message = str(exc.value)
    assert "no-such-lens" in message
    # Acquisition and import are separate steps, and a researcher who has done
    # neither cannot tell which one is missing from "no imported lens".
    assert "jlens acquire" in message and "jlens import" in message


def test_evidence_tier_comes_from_the_lens_not_the_run(tmp_path):
    record, _ = _write_lens(tmp_path, model_id="google/gemma-3-27b-it")
    assert optvec_jspace.evidence_tier_for(record) == "evidence"

    record.fit.modelID = "google/gemma-3-4b-it"
    assert optvec_jspace.evidence_tier_for(record) == "testing"

    # A lens whose fit model is not in the supported table is stamped unknown
    # rather than defaulted into a tier it never earned.
    record.fit.modelID = "test/tiny"
    assert optvec_jspace.evidence_tier_for(record) == "unknown"
    record.fit.modelID = None
    assert optvec_jspace.evidence_tier_for(record) == "unknown"


def test_the_null_is_the_same_distribution_the_study_path_controls_with():
    """The matched-norm random control has one stamped algorithm across the
    engine; a J-space null drawn from a different distribution would not be
    comparable with any other control in the program."""
    from steerlab_server.experiment import tasks

    assert optvec_jspace.NULL_ALGORITHM == tasks.RANDOM_VECTOR_ALGORITHM


# ------------------------------------------------------ observation-layer rule


def test_default_observation_ladder_is_L_then_stride_capped_at_the_lens(tmp_path):
    source = _RandomLensSource(source_layers=tuple(range(0, 14)))
    record, _ = _write_lens(tmp_path, source=source)
    # L, L+4, L+8 … while fitted, plus the deepest fitted layer if the stride
    # missed it (13 here: 2, 6, 10 then 13).
    assert optvec_jspace.observation_layers(2, record) == [2, 6, 10, 13]
    assert optvec_jspace.observation_layers(12, record) == [12, 13]
    # The deepest fitted layer is never duplicated when the stride lands on it.
    assert optvec_jspace.observation_layers(13, record) == [13]
    assert optvec_jspace.DEFAULT_LAYER_STRIDE == 4


def test_observation_layers_below_the_injection_or_off_the_lens_refuse(tmp_path):
    record, _ = _write_lens(tmp_path)                # fitted 0,1,2; target 3
    with pytest.raises(OptVecJSpaceError) as exc:
        optvec_jspace.observation_layers(1, record, [0, 1, 2])
    assert "below the injection layer" in str(exc.value)

    with pytest.raises(OptVecJSpaceError) as exc:
        optvec_jspace.observation_layers(1, record, [1, 3])
    # The target layer has no Jacobian by construction.
    assert "3" in str(exc.value) and "target layer" in str(exc.value)

    # An injection layer the lens never fitted cannot carry the direct term.
    with pytest.raises(OptVecJSpaceError) as exc:
        optvec_jspace.observation_layers(3, record)
    assert "injection layer 3" in str(exc.value)

    assert optvec_jspace.observation_layers(1, record, [2, 1]) == [1, 2]


# ----------------------------------------------------------- paired capture


def _capture_pair(model, items, *, layers, direction, norm, layer=OPTVEC_LAYER):
    baseline = optvec_jspace.capture_answer_residuals(
        model, items, layers=layers, injector=None, microbatch_size=2)
    steered = optvec_jspace.capture_answer_residuals(
        model, items, layers=layers,
        injector=optvec_jspace.injector_for(model, layer, direction, norm),
        microbatch_size=2)
    return baseline, steered


def test_delta_at_the_injection_layer_is_exactly_alpha_v(tmp_path):
    """The decomposition's foundation: capture is armed after the injector, so
    at ``ℓ = L`` the steered−baseline delta IS the dosed vector — the emergent
    term is zero there, and anything else means the capture is reading the
    pre-intervention residual (or the wrong position)."""
    model = _tiny_steered_model()
    items = _probe_items(model, tmp_path)
    generator = torch.Generator().manual_seed(4)
    raw = torch.randn(HIDDEN, generator=generator)
    direction = [float(x) for x in raw]
    norm = 6.0
    dosed = torch.tensor(direction, dtype=torch.float32)
    dosed = dosed * (norm / float(dosed.norm()))

    baseline, steered = _capture_pair(model, items,
                                      layers=[OPTVEC_LAYER, LAYERS - 1],
                                      direction=direction, norm=norm)
    for item in items:
        delta = steered[item.id][OPTVEC_LAYER] - baseline[item.id][OPTVEC_LAYER]
        assert torch.allclose(delta, dosed, atol=1e-5), item.id
        emergent = delta - dosed
        assert float(emergent.norm()) < 1e-4
        # Downstream the model has computed something from it: the delta is no
        # longer the vector.
        deep = steered[item.id][LAYERS - 1] - baseline[item.id][LAYERS - 1]
        assert float((deep - dosed).norm()) > 1e-3, item.id


def test_paired_passes_are_deterministic_and_teacher_forced(tmp_path):
    """Two preconditions of per-position pairing, both asserted rather than
    assumed: the baseline is reproducible, and the steered pass ran the SAME
    token sequence (so every layer BELOW the injection is bit-identical — the
    only thing that differs is the intervention)."""
    model = _tiny_steered_model()
    items = _probe_items(model, tmp_path)
    generator = torch.Generator().manual_seed(9)
    direction = [float(x) for x in torch.randn(HIDDEN, generator=generator)]

    layers = [0, 1, OPTVEC_LAYER, LAYERS - 1]
    baseline, steered = _capture_pair(model, items, layers=layers,
                                      direction=direction, norm=5.0)
    again = optvec_jspace.capture_answer_residuals(
        model, items, layers=layers, injector=None, microbatch_size=2)

    for item in items:
        for layer in layers:
            assert torch.equal(baseline[item.id][layer],
                               again[item.id][layer]), (item.id, layer)
        for layer in (0, 1):
            # Below the injection nothing can differ unless the sequences did.
            assert torch.equal(baseline[item.id][layer],
                               steered[item.id][layer]), (item.id, layer)
        assert not torch.equal(baseline[item.id][OPTVEC_LAYER],
                               steered[item.id][OPTVEC_LAYER])


def test_capture_is_read_only_and_refuses_a_second_firing():
    capture = optvec_jspace.AnswerPositionCapture([0])
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    before = hidden.clone()
    capture.set_batch(torch.tensor([2, 1]))
    out = capture.apply(hidden, 0, 0)
    assert out is hidden                       # identity, not a copy
    assert torch.equal(hidden, before)         # no in-place mutation
    assert torch.equal(capture.rows[0][0], hidden[0, 2])
    assert torch.equal(capture.rows[0][1], hidden[1, 1])
    assert capture.apply(hidden, 1, 0) is hidden and 1 not in capture.rows
    with pytest.raises(OptVecJSpaceError):
        capture.apply(hidden, 0, 0)            # one pass per microbatch


def test_capture_without_positions_refuses_rather_than_guessing():
    capture = optvec_jspace.AnswerPositionCapture([0])
    with pytest.raises(OptVecJSpaceError):
        capture.apply(torch.zeros(1, 2, 4), 0, 0)


# ------------------------------------------------------------------ linearity


def test_transported_delta_equals_direct_plus_emergent(tmp_path):
    """``J`` is linear, so the decomposition is an identity, not an
    approximation — measured through the production readout with a DENSE
    Jacobian (a diagonal one could hide a missing transport)."""
    from steerlab_server.jlens.readout import LensReadout, ReadoutConfig

    model = _tiny_steered_model()
    record, root = _write_lens(tmp_path, source=_RandomLensSource())
    readout = LensReadout.build(
        record=record,
        config=ReadoutConfig(layers=[OPTVEC_LAYER], topK=3,
                             topKLayers=[OPTVEC_LAYER],
                             logitLensCompanion=False),
        model=model, root=root)

    items = _probe_items(model, tmp_path)
    generator = torch.Generator().manual_seed(21)
    direction = [float(x) for x in torch.randn(HIDDEN, generator=generator)]
    norm = 4.0
    dosed = torch.tensor(direction, dtype=torch.float32)
    dosed = dosed * (norm / float(dosed.norm()))
    baseline, steered = _capture_pair(model, items, layers=[OPTVEC_LAYER],
                                      direction=direction, norm=norm)

    transported_direct = optvec_jspace._transport(readout, dosed, OPTVEC_LAYER)
    # The dense Jacobian actually rotates: transport is doing work.
    assert float(transported_direct.norm()) > 0
    cosine_with_raw = optvec_jspace.cosine(
        transported_direct, dosed.to(transported_direct.device))
    assert abs(cosine_with_raw) < 0.99

    for item in items:
        delta = steered[item.id][OPTVEC_LAYER] - baseline[item.id][OPTVEC_LAYER]
        emergent = delta - dosed
        left = optvec_jspace._transport(readout, delta, OPTVEC_LAYER)
        right = transported_direct + optvec_jspace._transport(
            readout, emergent, OPTVEC_LAYER)
        assert torch.allclose(left, right, atol=1e-4)


# ------------------------------------------------------------- null discipline


def test_energy_block_and_the_schema_refuse_an_energy_without_its_null():
    block = optvec_jspace.energy_block("meanDelta", 4.0, 2.0)
    assert block == {"meanDeltaEnergy": 4.0, "meanDeltaEnergyNull": 2.0,
                     "meanDeltaEnergyNullRatio": 2.0}
    optvec_jspace.validate_energy_pairing({"layers": [block]})

    for broken in ({"meanDeltaEnergy": 4.0},
                   {"meanDeltaEnergy": 4.0, "meanDeltaEnergyNull": 2.0},
                   {"a": {"b": [{"emergentEnergy": 1.0}]}}):
        with pytest.raises(OptVecJSpaceError) as exc:
            optvec_jspace.validate_energy_pairing(broken)
        assert "null" in str(exc.value)

    # A zero null is not a divide-by-zero: the ratio is undefined, and the
    # energy is still reported beside it.
    assert optvec_jspace.energy_block("direct", 1.0,
                                      0.0)["directEnergyNullRatio"] is None


def test_every_energy_in_a_real_report_carries_its_null(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch)["report"]
    optvec_jspace.validate_energy_pairing(report)
    keys = _energy_keys(report)
    # Not vacuous: the report does contain energies.
    assert {"meanDeltaEnergy", "directEnergy", "meanEmergentEnergy"} <= keys
    for key in keys:
        assert key + "Null" in _all_keys(report)
        assert key + "NullRatio" in _all_keys(report)


def test_the_null_is_deterministic_under_its_seed(tmp_path, monkeypatch):
    first = _run(tmp_path / "a", monkeypatch)["report"]
    second = _run(tmp_path / "b", monkeypatch)["report"]
    changed = _run(tmp_path / "c", monkeypatch, null_seed=999)["report"]

    def nulls(report):
        return [entry["meanDeltaEnergyNull"]
                for vector in report["vectors"] for entry in vector["layers"]]

    assert nulls(first) == nulls(second)
    assert nulls(first) != nulls(changed)
    assert first["null"]["seedBase"] == optvec_jspace.NULL_SEED
    assert changed["null"]["seedBase"] == 999
    # The null's norm is matched to the dose, not merely to the raw vector.
    for vector in first["vectors"]:
        assert vector["null"]["matchedNorm"] == \
            pytest.approx(vector["appliedNorm"])


# ------------------------------------------------------------ multiple draws


def test_null_draws_refuses_anything_that_is_not_a_count(tmp_path):
    """A draw count is a number of forward passes, so a typo is refused rather
    than rounded: 2.7 draws and "3" draws are not requests this verb can honor,
    and coercing either would put unrequested compute on a cluster job."""
    probe = _write_probe(tmp_path / "probe.jsonl").to_dict()
    base = {"vectorArtifacts": ["runs/x/toy"], "lensID": LENS_ID,
            "probeItems": probe}
    assert OptVecJSpaceConfig.from_dict(base).null_draws == 1

    for bad in (0, -1, 2.7, "3", True, None):
        with pytest.raises(OptVecJSpaceConfigError) as exc:
            OptVecJSpaceConfig.from_dict({**base, "nullDraws": bad})
        assert "nullDraws" in str(exc.value)

    config = OptVecJSpaceConfig.from_dict({**base, "nullDraws": 4})
    assert config.null_draws == 4
    assert config.to_dict()["nullDraws"] == 4


def test_three_null_draws_stamp_three_seeds_and_carry_spread(tmp_path,
                                                              monkeypatch):
    """Three draws are three INDEPENDENT comparator passes: three distinct
    seeds, three per-draw energies, a spread across them, and a count of how
    many met or exceeded the observed energy — the rank statistic that is the
    only thing k draws can support (NULL_INFERENCE_NOTE)."""
    out = _run(tmp_path, monkeypatch, null_draws=3)
    report, records = out["report"], out["records"]

    null = report["vectors"][0]["null"]
    assert null["drawCount"] == 3
    seeds = [draw["seed"] for draw in null["draws"]]
    assert [draw["draw"] for draw in null["draws"]] == [0, 1, 2]
    assert len(set(seeds)) == 3
    # Draw 0 keeps the historical seed; the others are strided off it.
    assert seeds[0] == optvec_jspace.NULL_SEED == null["seed"]
    assert seeds == [optvec_jspace.NULL_SEED + k * optvec_jspace.NULL_DRAW_STRIDE
                     for k in range(3)]
    assert report["null"]["drawCount"] == 3

    entry = report["vectors"][0]["layers"][0]
    assert entry["nullDrawCount"] == 3
    for name in ("meanDeltaEnergy", "directEnergy", "meanEmergentEnergy"):
        draws = entry[name + "NullDraws"]
        assert len(draws) == 3
        # Three different random directions, so three different energies.
        assert len(set(draws)) == 3
        assert entry[name + "Null"] == pytest.approx(sum(draws) / 3, rel=1e-12)
        assert entry[name + "NullSD"] > 0.0
        assert entry[name + "NullExceedanceCount"] == sum(
            1 for value in draws if value >= entry[name])

    # Per item: the mean over draws, with the draws beside it and no spread —
    # a three-draw SD on one item would read as that item's error bar.
    for record in records:
        assert record["nullDrawCount"] == 3
        assert record["nullSeeds"] == seeds
        assert record["nullSeed"] == seeds[0]
        assert len(record["deltaEnergyNullDraws"]) == 3
        assert record["deltaEnergyNull"] == pytest.approx(
            sum(record["deltaEnergyNullDraws"]) / 3, rel=1e-12)
        assert "deltaEnergyNullSD" not in record

    # Still a comparator, and the artifact says so in its own words.
    assert report["null"]["inferenceNote"] == optvec_jspace.NULL_INFERENCE_NOTE
    assert "1/(k+1)" in optvec_jspace.NULL_INFERENCE_NOTE
    assert report["null"]["seedRule"] == optvec_jspace.NULL_DRAW_SEED_RULE
    assert str(optvec_jspace.NULL_DRAW_STRIDE) in \
        optvec_jspace.NULL_DRAW_SEED_RULE


def _without_root(payload, root):
    """The payload with one workspace root spelled ``ROOT``, so two runs in two
    workspaces can be compared value by value."""
    return json.loads(json.dumps(payload, sort_keys=True).replace(root, "ROOT"))


def test_one_null_draw_reproduces_the_report_that_had_no_draws_option(
        tmp_path, monkeypatch):
    """The compatibility guarantee behind the seed derivation: asking for one
    draw is the default, and the default is what this verb has always done. Not
    "the same shape" — the same VALUES, key for key, including the null note,
    which stays the sentence it was for a single-draw run because that sentence
    is still true of it."""
    default = _run(tmp_path / "a", monkeypatch)
    explicit = _run(tmp_path / "b", monkeypatch, null_draws=1)

    left = _without_root(default["report"], str(tmp_path / "a"))
    right = _without_root(explicit["report"], str(tmp_path / "b"))
    for report in (left, right):
        # The run id is a timestamp; everything else must match.
        report.pop("runID")
    assert left == right
    assert _without_root(default["records"], str(tmp_path / "a")) == \
        _without_root(explicit["records"], str(tmp_path / "b"))

    assert left["null"]["drawCount"] == 1
    assert left["null"]["note"] == (
        "one independent matched-norm random direction per artifact (seedBase "
        "+ artifact ordinal), pushed through the identical pipeline; every "
        "energy is reported only beside its null")
    entry = left["vectors"][0]["layers"][0]
    # One draw has no spread, and 0.0 would read as "measured, and it was zero".
    assert entry["meanDeltaEnergyNullSD"] is None
    assert entry["meanDeltaEnergyNullDraws"] == [entry["meanDeltaEnergyNull"]]


# --------------------------------------------------------- energy partition


def test_the_energy_partition_is_three_termed_and_stamped_verbatim(tmp_path,
                                                                    monkeypatch):
    """A squared norm is not additive over a sum, so directEnergy and
    emergentEnergy do not add to deltaEnergy — the cross term is the difference
    and is reported, precisely so ``directEnergy / deltaEnergy`` is not read as
    the share of the effect the vector accounts for."""
    assert optvec_jspace.ENERGY_PARTITION == (
        "deltaEnergy = directEnergy + emergentEnergy + crossTerm, where "
        "crossTerm = 2*<J_l(direct), J_l(emergent)> and may be NEGATIVE (so it "
        "carries no null ratio). The three terms are an arithmetic partition "
        "of one squared norm: directEnergy/deltaEnergy is NOT a causal share, "
        "not a percentage of the effect, and not comparable across doses")

    # Observed ABOVE the injection layer: at ℓ = L the emergent term is zero
    # and the identity holds however the cross term is computed.
    out = _run(tmp_path, monkeypatch, injection_layer=1)
    report, records = out["report"], out["records"]
    assert report["observationLayers"] == [1, 2]
    assert report["instrument"]["energyPartition"] == \
        optvec_jspace.ENERGY_PARTITION

    material = 0
    for vector in report["vectors"]:
        for entry in vector["layers"]:
            total = (entry["directEnergy"] + entry["meanEmergentEnergy"]
                     + entry["meanDeltaEnergyCrossTerm"])
            assert entry["meanDeltaEnergy"] == pytest.approx(total, rel=1e-6,
                                                             abs=1e-9)
            null_total = (entry["directEnergyNull"]
                          + entry["meanEmergentEnergyNull"]
                          + entry["meanDeltaEnergyCrossTermNull"])
            assert entry["meanDeltaEnergyNull"] == pytest.approx(
                null_total, rel=1e-6, abs=1e-9)
            if abs(entry["meanDeltaEnergyCrossTerm"]) > 1e-3:
                material += 1
    # Not vacuous: at least one layer's cross term is far from zero, so the
    # identity is testing an addend rather than confirming that 0 == 0.
    assert material >= 1
    material_rows = 0
    for record in records:
        total = (record["directEnergy"] + record["emergentEnergy"]
                 + record["deltaEnergyCrossTerm"])
        assert record["deltaEnergy"] == pytest.approx(total, rel=1e-6,
                                                      abs=1e-9)
        if abs(record["deltaEnergyCrossTerm"]) > 1e-3:
            material_rows += 1
    assert material_rows >= 1

    # A cross term is signed, so it takes a null but never a ratio: a quotient
    # of two quantities either side of zero compares nothing.
    keys = _all_keys(report) | _all_keys(records)
    assert "meanDeltaEnergyCrossTermNull" in keys
    assert not [k for k in keys if k.endswith("CrossTermNullRatio")]
    optvec_jspace.validate_energy_pairing(report)


def test_the_cross_term_is_twice_the_inner_product_of_the_transported_terms(
        tmp_path, monkeypatch):
    """The reported cross term is computed as the residual of the partition;
    this checks it against its closed form, ``2<J(direct), J(emergent)>``,
    through the production readout with a DENSE Jacobian and at a layer where
    the emergent term is not zero."""
    from steerlab_server.jlens.readout import LensReadout, ReadoutConfig
    from steerlab_server.steering import vector_math as vm

    tmp = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", tmp)
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(HIDDEN, generator=generator)
    direction = [float(x) for x in raw / raw.norm() * 6.0]
    reference = write_optvec_artifact(os.path.join(tmp, "runs", "train"), "v0",
                                      layer=1, direction=direction)
    record, _ = _write_lens(tmp_path, source=_RandomLensSource(), root=tmp)
    model = _tiny_steered_model()
    readout = LensReadout.build(
        record=record,
        config=ReadoutConfig(layers=[1, 2], topK=3, topKLayers=[1, 2],
                             logitLensCompanion=False),
        model=model, root=tmp)
    items = _probe_items(model, tmp_path)

    norm = 6.0
    dosed = torch.tensor(direction, dtype=torch.float32)
    dosed = dosed * (norm / float(dosed.norm()))
    null_direction = vm.random_vector(HIDDEN, norm, seed=3)
    baseline, steered = _capture_pair(model, items, layers=[1, 2],
                                      direction=direction, norm=norm, layer=1)
    steered_null = optvec_jspace.capture_answer_residuals(
        model, items, layers=[1, 2],
        injector=optvec_jspace.injector_for(model, 1, null_direction, norm),
        microbatch_size=2)

    passes = optvec_jspace.VectorPass(
        target=optvec_jspace._resolve_targets([reference])[0], direct=dosed,
        null_direct=torch.tensor(null_direction, dtype=torch.float32),
        null_seed=3, steered=steered, steered_null=steered_null)
    aggregates, records, _ = optvec_jspace.decompose_vector(
        readout, model.tokenizer, passes=passes, baseline=baseline,
        items=items, layers=[1, 2], top_k=3)

    direct_t = optvec_jspace._transport(readout, dosed, 2)
    deep = [r for r in records if r["layer"] == 2]
    assert len(deep) == len(items)
    nonzero = 0
    for record_row, item in zip(deep, items):
        delta = steered[item.id][2] - baseline[item.id][2]
        emergent_t = optvec_jspace._transport(readout, delta - dosed, 2)
        closed_form = 2.0 * float(direct_t.dot(emergent_t))
        assert record_row["deltaEnergyCrossTerm"] == pytest.approx(
            closed_form, rel=1e-3, abs=1e-4)
        if abs(closed_form) > 1e-3:
            nonzero += 1
    # Not vacuous: at ℓ > L the model has computed something, and the cross
    # term of two aligned-but-not-parallel vectors is not zero.
    assert nonzero == len(items)

    entry = {a["layer"]: a for a in aggregates}[2]
    assert entry["meanDeltaEnergy"] == pytest.approx(
        entry["directEnergy"] + entry["meanEmergentEnergy"]
        + entry["meanDeltaEnergyCrossTerm"], rel=1e-6)


# -------------------------------------------------------- realized dose


def test_dtype_epsilon_is_none_unless_the_dtype_is_a_known_float():
    """An unrecognized dtype gets no epsilon rather than float32's: silently
    reporting 1.2e-7 for a bf16 or quantized runtime would understate the noise
    floor by three orders of magnitude."""
    assert optvec_jspace.dtype_epsilon("float32") == \
        pytest.approx(float(torch.finfo(torch.float32).eps))
    assert optvec_jspace.dtype_epsilon("bfloat16") == \
        pytest.approx(float(torch.finfo(torch.bfloat16).eps))
    assert optvec_jspace.dtype_epsilon("bfloat16") > \
        optvec_jspace.dtype_epsilon("float32")
    for unknown in (None, "", "int8", "nf4", "auto"):
        assert optvec_jspace.dtype_epsilon(unknown) is None


def test_the_realized_dose_block_reports_the_noise_floor_at_the_injection_layer(
        tmp_path, monkeypatch):
    """Converting a capture to float32 does not recover precision the runtime
    already discarded, so the nominal dose and the change the model actually
    realized are reported side by side. On this fp32 fixture the gap is at the
    fp32 floor — the point of the block is that the number is MEASURED, not
    that it is small; a bf16 runtime is where it stops being."""
    report = _run(tmp_path, monkeypatch)["report"]
    vector = report["vectors"][0]
    block = vector["realizedDose"]

    assert block["layer"] == report["injectionLayer"]
    assert block["captureDtype"] == "float32"
    assert block["dtypeEpsilon"] == pytest.approx(
        float(torch.finfo(torch.float32).eps))
    assert block["nominalDirectNorm"] == pytest.approx(vector["appliedNorm"])
    # At ℓ = L the delta IS the dosed vector, so the realized norm matches and
    # the residue is realization noise.
    assert block["meanRealizedDeltaNorm"] == pytest.approx(
        block["nominalDirectNorm"], rel=1e-5)
    assert 0.0 <= block["meanRealizationErrorNorm"] < 1e-4
    assert block["realizationErrorToDoseRatio"] == pytest.approx(
        block["meanRealizationErrorNorm"] / block["nominalDirectNorm"])
    # The dose is only interpretable against the stream it is added to.
    assert block["meanBaselineResidualNorm"] > 0.0
    assert block["note"] == optvec_jspace.PRECISION_NOTE
    assert report["instrument"]["precision"] == optvec_jspace.PRECISION_NOTE
    assert optvec_jspace.PRECISION_NOTE == (
        "residual rows are captured at the runtime dtype (captureDtype) and "
        "cast to float32 inside the pass; the cast does not recover precision "
        "the runtime dtype already discarded. meanRealizationErrorNorm is the "
        "mean ||delta_L - direct|| AT THE INJECTION LAYER, where the delta is "
        "the dosed vector by construction and any difference is realization "
        "noise: it is the empirical noise floor for THIS dose and THIS dtype, "
        "and a low-dose result must be read against it rather than against "
        "zero")


def test_the_injection_layer_is_captured_even_when_it_is_not_observed(
        tmp_path, monkeypatch):
    """The realized-dose block is only meaningful at L, so L is captured even
    when the declared observation layers skip it — without adding a row to
    observationLayers, which stays exactly what was asked for."""
    tmp = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", tmp)
    generator = torch.Generator().manual_seed(11)
    raw = torch.randn(HIDDEN, generator=generator)
    reference = write_optvec_artifact(
        os.path.join(tmp, "runs", "train"), "v0", layer=1,
        direction=[float(x) for x in raw / raw.norm() * 6.0])
    _write_lens(tmp_path, source=_RandomLensSource(source_layers=(0, 1, 2)),
                root=tmp)
    config = _config(tmp_path, [reference], observation_layers=[2])
    result = optvec_jspace.analyze(config, model=_tiny_steered_model(),
                                   root=tmp)

    assert result["observationLayers"] == [2]
    assert [entry["layer"] for entry in result["vectors"][0]["layers"]] == [2]
    block = result["vectors"][0]["realizedDose"]
    assert block["layer"] == 1
    assert block["meanRealizedDeltaNorm"] == pytest.approx(
        block["nominalDirectNorm"], rel=1e-5)


# ------------------------------------------------------- what the numbers are


def test_the_quantities_block_is_stamped_verbatim(tmp_path, monkeypatch):
    """The separation the review asked for, in the artifact rather than in a
    reviewer's memory: what each table is, and what it is not."""
    report = _run(tmp_path, monkeypatch)["report"]
    assert report["quantities"] == optvec_jspace.QUANTITIES
    assert optvec_jspace.QUANTITIES == {
        "topKDelta": (
            "the lens's NORMALIZED readout of the MEAN over items of the "
            "(steered - baseline) residual at the answer position: a readout "
            "of a DIFFERENCE vector. It is not the difference between the "
            "normalized readouts of the two states (normalization does not "
            "distribute over a subtraction), and it is not an observed change "
            "in the model's logits"),
        "topKEmergent": (
            "the same normalized readout applied to the mean of (delta - "
            "direct): a readout of a DIFFERENCE vector, not the difference "
            "between two readouts, and not an observed change in the model's "
            "logits"),
        "cosineDeltaDirect": (
            "cosine between J_l(delta) and J_l(direct) in the lens's output "
            "basis -- an angle between two transported vectors, per item and "
            "averaged over items; it says how aligned the propagated delta "
            "stayed with the dosed vector, not how much of the delta the "
            "vector caused"),
        "energies": (optvec_jspace.ENERGY_DEFINITION + ". "
                     + optvec_jspace.ENERGY_PARTITION),
        "linearityResidualL2": (
            "||J_l(delta) - [J_l(direct) + J_l(emergent)]||_2: an ARITHMETIC "
            "identity check. J(x+y) = J(x) + J(y) holds for any linear map and "
            "any split of the delta, so a low residual confirms the transport "
            "was computed correctly and is NOT evidence that the lens explains "
            "the model's behavior"),
        "lens": (
            "J_l is an AVERAGED JACOBIAN ESTIMATOR fitted on the lens's own "
            "prompts; whether it transfers to this probe set's prompt "
            "distribution at this dose is an empirical question, answered by "
            "jlens qualify and by nothing in this report"),
        "scope": (
            "no field in this report measures the intervention's effect on "
            "the model's OUTPUTS -- token probabilities, choices, or generated "
            "text. That is a separate behavioral run; this verb reads residual "
            "streams"),
    }


def test_the_producing_revision_is_stamped_in_the_report_and_the_notes(
        tmp_path, monkeypatch):
    """A future impact ledger has to classify this artifact by the code that
    produced it, and the only place that can come from is the artifact."""
    from steerlab_server import build_identity

    out = _run(tmp_path, monkeypatch)
    expected = {"version": build_identity.engine_version(),
                "buildCommit": build_identity.build_commit()}
    assert out["report"]["engine"] == expected
    assert expected["version"].startswith("steerlab-server ")

    notes = json.load(open(os.path.join(out["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["engine"] == expected
    assert notes["nullDraws"] == 1


# ---------------------------------------------------------------- family mode


def test_family_tables_report_both_statistics_on_hand_built_deltas():
    """Shallow vs deep multiplicity is exactly this comparison: two directions
    that scatter raw (orthogonal, PR 2) but whose propagated deltas are
    parallel (PR 1) are one mechanism differently parameterized. The engine
    reports both numbers and no verdict."""
    entries = [{"reference": "runs/a/v1"}, {"reference": "runs/b/v2"}]
    raw = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    propagated = {4: [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],       # collapsed
                  8: [[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]]}       # separated
    table = optvec_jspace.family_tables(entries, raw, propagated, layer=2)

    assert table["layer"] == 2 and table["count"] == 2
    assert table["entries"] == entries
    assert table["raw"]["cosineMatrix"][0][1] == pytest.approx(0.0, abs=1e-12)
    assert table["raw"]["participationRatioUnitNormalized"] == \
        pytest.approx(2.0, rel=1e-9)

    by_layer = {row["layer"]: row for row in table["byObservationLayer"]}
    assert sorted(by_layer) == [4, 8]
    assert by_layer[4]["cosineMatrix"][0][1] == pytest.approx(1.0, abs=1e-9)
    assert by_layer[4]["participationRatioUnitNormalized"] == \
        pytest.approx(1.0, rel=1e-9)
    assert by_layer[8]["cosineMatrix"][0][1] == pytest.approx(0.0, abs=1e-12)
    assert by_layer[8]["participationRatioUnitNormalized"] == \
        pytest.approx(2.0, rel=1e-9)
    # Numbers only — no field asserts which reading the family shows.
    assert "shallow" not in json.dumps(table) and "deep" not in json.dumps(table)


def test_a_single_artifact_reports_no_family_and_two_do(tmp_path, monkeypatch):
    one = _run(tmp_path / "one", monkeypatch)["report"]
    assert one["family"] is None

    two = _run(tmp_path / "two", monkeypatch, artifacts=2)["report"]
    family = two["family"]
    assert family["count"] == 2
    assert family["layer"] == OPTVEC_LAYER
    assert [e["reference"] for e in family["entries"]] == \
        [v["artifact"]["reference"] for v in two["vectors"]]
    assert len(family["raw"]["cosineMatrix"]) == 2
    assert [row["layer"] for row in family["byObservationLayer"]] == \
        two["observationLayers"]
    for row in family["byObservationLayer"]:
        assert len(row["cosineMatrix"]) == 2
        assert row["participationRatio"] > 0


def test_artifacts_at_different_layers_refuse(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    a = write_optvec_artifact(tmp_path / "runs" / "a", "v1", layer=OPTVEC_LAYER)
    b = write_optvec_artifact(tmp_path / "runs" / "b", "v2", layer=1)
    _write_lens(tmp_path, root=str(tmp_path))
    config = _config(tmp_path, [a, b])
    with pytest.raises(OptVecJSpaceError) as exc:
        optvec_jspace.analyze(config, model=_tiny_steered_model(),
                              root=str(tmp_path))
    assert "different optvec layers" in str(exc.value)


# ----------------------------------------------------------------- end to end


def _run(tmp_path, monkeypatch, *, artifacts=1, injection_layer=OPTVEC_LAYER,
         **overrides):
    """One full analyze() on the tiny model, in its own workspace root.

    ``injection_layer`` exists so a test can ask for a run that OBSERVES above
    the injection: at ℓ = L the emergent term is zero by construction and every
    identity involving it is trivially satisfied, which is exactly the shape a
    test must not settle for.
    """
    root = str(tmp_path)
    os.makedirs(root, exist_ok=True)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    references = []
    for index in range(artifacts):
        generator = torch.Generator().manual_seed(31 + index)
        raw = torch.randn(HIDDEN, generator=generator)
        direction = [float(x) for x in raw / raw.norm() * 6.0]
        references.append(write_optvec_artifact(
            os.path.join(root, "runs", f"train-{index}"), f"v{index}",
            layer=injection_layer, direction=direction))
    _write_lens(tmp_path, source=_RandomLensSource(), root=root)
    config = _config(tmp_path, references, **overrides)
    result = optvec_jspace.analyze(config, model=_tiny_steered_model(),
                                   root=root)
    run_dir = result["runDirectory"]
    report = json.load(open(os.path.join(run_dir, optvec_jspace.JSPACE_JSON)))
    records = [json.loads(line) for line in
               open(os.path.join(run_dir, optvec_jspace.JSPACE_RECORDS))
               if line.strip()]
    return {"result": result, "report": report, "records": records,
            "runDirectory": run_dir}


def _all_keys(payload, into=None):
    into = set() if into is None else into
    if isinstance(payload, dict):
        into.update(payload)
        for value in payload.values():
            _all_keys(value, into)
    elif isinstance(payload, list):
        for value in payload:
            _all_keys(value, into)
    return into


def _energy_keys(payload):
    return {k for k in _all_keys(payload) if k.endswith("Energy")}


def test_a_full_run_writes_an_immutable_run_directory(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    run_dir, report, records = out["runDirectory"], out["report"], out["records"]

    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-jspace"
    assert config_json["modelID"] == "test/tiny"
    notes = config_json["notes"]
    assert notes["stage"] == "complete"
    assert notes["lensID"] == LENS_ID
    assert notes["recordCount"] == len(records)

    assert report["runType"] == "optvec-jspace"
    assert report["injectionLayer"] == OPTVEC_LAYER
    assert report["observationLayers"] == [OPTVEC_LAYER]   # lens fits 0..2
    assert report["observationLayerRule"] == optvec_jspace.LAYER_RULE
    assert report["probeItems"]["itemCount"] == 3
    assert report["lens"]["lensID"] == LENS_ID
    assert report["lens"]["readoutConvention"] == schemas.CANONICAL_READOUT
    assert report["lens"]["compatibility"]["qualified"] is False

    assert len(report["vectors"]) == 1
    entry = report["vectors"][0]["layers"][0]
    assert entry["itemCount"] == 3 and entry["isInjectionLayer"] is True
    # At ℓ = L the emergent term is zero and the delta IS the direct term.
    assert entry["meanEmergentEnergy"] == pytest.approx(0.0, abs=1e-6)
    assert entry["meanDeltaEnergy"] == pytest.approx(entry["directEnergy"],
                                                     rel=1e-5)
    assert entry["meanCosineDeltaDirect"] == pytest.approx(1.0, abs=1e-5)
    assert entry["maxLinearityResidualL2"] < 1e-3
    assert len(entry["topKDelta"]) == 3
    assert all(isinstance(row["tokenID"], int) for row in entry["topKDelta"])
    # The run seed moves nothing here (no sampling) and says so, rather than
    # sitting in the record looking causal.
    assert report["seed"] == {
        "seed": 0, "seedInert": True,
        "note": "no sampling occurs in this verb; the only RNG is the null "
                "draw, which uses nullSeed"}

    assert len(records) == 3
    assert {r["itemID"] for r in records} == {"p-0", "p-1", "p-2"}
    for record in records:
        assert record["layer"] == OPTVEC_LAYER
        assert record["linearityResidualL2"] < 1e-3
        assert record["deltaEnergyNull"] > 0


def test_the_tier_and_the_qualification_caveat_are_stamped_verbatim(tmp_path,
                                                                    monkeypatch):
    """Plan §7 WP8: an unqualified runtime's J-space readout is exploratory by
    stamp — written by the engine, not by whoever remembers to say so in the
    paper.

    The stamp was reworded when Stage 4 landed (2026-08-15): it used to say
    qualification was UNIMPLEMENTED, which is now false, and a stamp that
    misdescribes why a readout is uncitable is worse than none. The default is
    still 'unqualified', because this fixture's lens has no qualification for
    this runtime."""
    out = _run(tmp_path, monkeypatch)
    report = out["report"]
    assert report["qualification"] == (
        "unqualified for this runtime — no passing jlens qualification "
        "(steerlab-server jlens qualify); exploratory analysis, not citable "
        "evidence")
    assert report["qualification"] == optvec_jspace.QUALIFICATION_STAMP
    # The lens here is fitted on a model outside the supported table.
    assert report["evidenceTier"] == "unknown"
    assert report["claim"] == "exploratory"

    notes = json.load(open(os.path.join(out["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["qualification"] == optvec_jspace.QUALIFICATION_STAMP
    assert notes["evidenceTier"] == "unknown"


def test_deeper_observation_layers_show_a_nonzero_emergent_term(tmp_path,
                                                                monkeypatch):
    """The whole point of the verb: at ``ℓ = L`` the emergent term is zero, and
    a layer later it is not — the model has computed something from the
    vector."""
    tmp = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", tmp)
    generator = torch.Generator().manual_seed(5)
    raw = torch.randn(HIDDEN, generator=generator)
    reference = write_optvec_artifact(
        os.path.join(tmp, "runs", "train"), "v0", layer=1,
        direction=[float(x) for x in raw / raw.norm() * 6.0])
    _write_lens(tmp_path, source=_RandomLensSource(source_layers=(0, 1, 2)),
                root=tmp)
    config = _config(tmp_path, [reference], observation_layers=[1, 2])
    result = optvec_jspace.analyze(config, model=_tiny_steered_model(),
                                   root=tmp)
    layers = {entry["layer"]: entry for entry in result["vectors"][0]["layers"]}
    assert layers[1]["meanEmergentEnergy"] == pytest.approx(0.0, abs=1e-6)
    assert layers[2]["meanEmergentEnergy"] > 0.0
    # And it is reported against the null the same pipeline produced.
    assert layers[2]["meanEmergentEnergyNull"] > 0.0
    assert layers[2]["meanEmergentEnergyNullRatio"] == pytest.approx(
        layers[2]["meanEmergentEnergy"] / layers[2]["meanEmergentEnergyNull"],
        rel=1e-9)


def test_a_lens_for_another_model_refuses_before_any_pass(tmp_path, monkeypatch):
    tmp = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", tmp)
    reference = write_optvec_artifact(os.path.join(tmp, "runs", "train"), "v0")
    _write_lens(tmp_path, model_id="google/gemma-3-27b-it", root=tmp)
    config = _config(tmp_path, [reference])
    with pytest.raises(OptVecJSpaceError) as exc:
        optvec_jspace.analyze(config, model=_tiny_steered_model(), root=tmp)
    message = str(exc.value)
    assert "cannot be used against this runtime" in message
    assert "google/gemma-3-27b-it" in message
