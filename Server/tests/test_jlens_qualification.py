"""Stage 4: the qualification producer, and the gate it makes satisfiable.

Every test here is CPU-only and model-free. The checks that need a resident
model are exercised on the cluster, not in CI (plan §12.3) — what CI must pin
is the part a live run cannot: that an unresolvable or differently-typed
runtime is REFUSED rather than inherited, that a testing-tier record can never
become an evidence-tier one, and that the freeze gate now has a producer to
satisfy it.
"""

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import experiment_store
from steerlab_server.jlens import (backend, importer, lens_store,
                                   qualification, schemas)

EVIDENCE_MODEL = "google/gemma-3-27b-it"
TESTING_MODEL = "google/gemma-3-4b-it"
REV = "005ad3404e59d6023443cb575daa05336842228a"


def _import(tmp_path, model_id, *, layers=(0, 1, 2)):
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=list(layers)).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {model_id}\n")
    return importer.import_lens(model_id, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


class FakeConfig:
    def __init__(self, quantization=None):
        if quantization is not None:
            self.quantization_config = quantization


class FakeInner:
    def __init__(self, *, dtype=torch.bfloat16, quantization=None):
        self.config = FakeConfig(quantization)
        self._param = torch.zeros(2, dtype=dtype)

    def parameters(self):
        yield self._param


class FakeModel:
    """The minimum surface :func:`resolve_runtime` reads. Deliberately not a
    SteeredModel: the point of the function is that it works off what the
    object actually presents, not off what a loader promised."""

    def __init__(self, *, dtype="bfloat16", param_dtype=torch.bfloat16,
                 quantization=None):
        self.dtype = dtype
        self.model = FakeInner(dtype=param_dtype, quantization=quantization)


# --- runtime resolution ------------------------------------------------------

def test_a_resolvable_bf16_runtime_resolves(tmp_path):
    assert qualification.resolve_runtime(FakeModel()) == ("bfloat16", None)


def test_an_unquantized_runtime_is_None_not_the_string_none():
    """The freeze gate matches ``(q.quantization or None) == (declared or
    None)``, and manifests declare no quantization key at all. Stamping the
    STRING "none" would make every unquantized qualification unmatchable while
    looking perfectly reasonable in the artifact."""
    _dtype, quantization = qualification.resolve_runtime(FakeModel())
    assert quantization is None


def test_an_unresolvable_dtype_refuses():
    """Absent is not a match (plan §3.3). A qualification keyed by numerics
    that could not be established would be inherited by whatever the runtime
    happened to be."""
    model = FakeModel(dtype=None)
    model.model = object()          # no parameters, no config
    with pytest.raises(qualification.QualificationRefused,
                       match="could not resolve the runtime dtype"):
        qualification.resolve_runtime(model)


def test_a_declared_dtype_that_is_not_in_the_vocabulary_falls_back_to_the_params():
    """The parameters are the ground truth; a stamped label is a claim."""
    model = FakeModel(dtype="banana", param_dtype=torch.float16)
    assert qualification.resolve_runtime(model) == ("float16", None)


def test_a_named_quantizer_is_recorded():
    model = FakeModel(quantization={"quant_method": "bitsandbytes_8bit"})
    assert qualification.resolve_runtime(model) == ("bfloat16",
                                                    "bitsandbytes_8bit")


def test_an_unnameable_quantizer_refuses_rather_than_reporting_unquantized():
    """'There is a quantizer here and I cannot say which' must never be
    recorded as 'not quantized' — that is the one reading that lets a
    differently-quantized runtime inherit the qualification."""
    model = FakeModel(quantization={"bits": 4})
    with pytest.raises(qualification.QualificationRefused,
                       match="cannot name"):
        qualification.resolve_runtime(model)


# --- individual checks -------------------------------------------------------

def _pin_tokenizer(monkeypatch, value):
    """Stand in for the snapshot read. Patched on the imported MODULE rather
    than by dotted string: ``steerlab_server.jlens`` does not re-export its
    submodules, so a string target resolves only when something else happened
    to import ``derive`` first — which made these tests pass alone and fail in
    the suite."""
    from steerlab_server.jlens import derive

    monkeypatch.setattr(derive, "tokenizer_identity_hash",
                        lambda *a, **k: value)


class GeometryModel(FakeModel):
    def __init__(self, *, num_layers=4, hidden_size=8, vocab=64, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model.lm_head = type("Head", (), {})()
        self.model.lm_head.weight = torch.zeros(vocab, hidden_size)
        self.model.config.vocab_size = vocab


def test_geometry_passes_on_a_matching_runtime(tmp_path, monkeypatch):
    record = _import(tmp_path, EVIDENCE_MODEL)          # target layer 3, d 8
    _pin_tokenizer(monkeypatch, "t" * 64)
    result = qualification._check_geometry(
        record, GeometryModel(), model_id=EVIDENCE_MODEL, revision=REV)
    assert result.passed is True
    assert result.measured["tokenizerHash"] == "t" * 64
    assert result.measured["vocabSize"] == 64


def test_geometry_fails_when_the_output_head_and_config_disagree(tmp_path,
                                                                monkeypatch):
    """The readout is indexed by token ID, so two answers to 'how many tokens
    are there' means its coordinate system is ambiguous."""
    record = _import(tmp_path, EVIDENCE_MODEL)
    _pin_tokenizer(monkeypatch, "t" * 64)
    model = GeometryModel()
    model.model.config.vocab_size = 65
    result = qualification._check_geometry(record, model,
                                           model_id=EVIDENCE_MODEL,
                                           revision=REV)
    assert result.passed is False
    assert "coordinate system is ambiguous" in result.detail


def test_geometry_fails_when_the_vocabulary_cannot_be_pinned(tmp_path,
                                                             monkeypatch):
    """An unpinnable tokenizer means a later reader cannot verify that the
    trace's token IDs still name the pieces they named."""
    record = _import(tmp_path, EVIDENCE_MODEL)
    _pin_tokenizer(monkeypatch, None)
    result = qualification._check_geometry(record, GeometryModel(),
                                           model_id=EVIDENCE_MODEL,
                                           revision=REV)
    assert result.passed is False
    assert "unpinnable vocabulary" in result.detail


def test_a_fit_dtype_divergence_is_stamped_not_refused(tmp_path):
    """The whole point of qualification is EMPIRICAL acceptance: refusing on a
    config field would decide by declaration what the checks decide by
    measurement. The divergence must be impossible to discover later only by
    inference, so it is stamped."""
    record = _import(tmp_path, EVIDENCE_MODEL)
    record.fit.dtype = "bfloat16"
    diverged = qualification._check_runtime_numerics(
        record, dtype="float16", quantization=None)
    assert diverged.passed is True
    assert diverged.measured["divergesFromFitDtype"] is True
    assert "DIVERGES" in diverged.detail

    matched = qualification._check_runtime_numerics(
        record, dtype="bfloat16", quantization=None)
    assert matched.measured["divergesFromFitDtype"] is False


def test_a_nonfinite_jacobian_fails_the_finiteness_check(tmp_path):
    from safetensors.torch import load_file, save_file

    from steerlab_server.experiment import paths

    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    assert qualification._check_jacobian_finite(record, root=root).passed

    path = paths.resolve(record.converted.path, root)
    tensors = load_file(path)
    tensors["layer_1"][0][0] = float("nan")
    save_file(tensors, path)
    result = qualification._check_jacobian_finite(record, root=root)
    assert result.passed is False
    assert "layer 1 contains non-finite" in result.detail


def test_reference_agreement_FAILS_when_the_reference_package_is_absent(
        tmp_path, monkeypatch):
    """'We could not check' and 'we checked and it agreed' are not the same
    evidence, and a qualification is the artifact that decides which one a
    study is standing on — so absence fails rather than skips quietly."""
    from steerlab_server.jlens import backend as backend_mod
    from steerlab_server.jlens.schemas import JLensError

    record = _import(tmp_path, EVIDENCE_MODEL)

    def _absent():
        raise JLensError("the J-lens reference package is not installed")

    monkeypatch.setattr(backend_mod, "require_reference", _absent)
    result = qualification._check_reference_agreement(
        record, None, None, [0], root=str(tmp_path / "ws"))
    assert result.passed is False and result.skipped is True
    assert "not installed" in result.detail


# --- tier --------------------------------------------------------------------

def test_tier_comes_from_the_supported_table():
    assert qualification.tier_for(EVIDENCE_MODEL) == "evidence"
    assert qualification.tier_for(TESTING_MODEL) == "testing"
    assert qualification.tier_for("mistral/whatever") == "unknown"


def test_a_testing_tier_record_is_refused_by_freeze_and_cannot_be_upgraded(
        tmp_path):
    """`jlens qualify` RUNS on 4B on purpose — rehearsing the mechanics before
    a 27B node is booked is the cheap pass. What must not happen is that
    rehearsal becoming citable."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, TESTING_MODEL)
    entry = schemas.Qualification(
        qualificationID="q-testing", modelID=TESTING_MODEL, revision=REV,
        dtype="bfloat16", tier=qualification.tier_for(TESTING_MODEL),
        passed=True)
    qualification.append_qualification(record, entry, root=root)
    assert entry.tier == "testing"

    block = {"lensID": record.lensID, "lensSHA256": record.source.tensorSHA256,
             "layers": [0, 1], "watchlist": [23648], "configHash": "c" * 64,
             "tokenizerHash": "t" * 64, "qualificationID": "q-testing"}
    d = {"name": "s", "modelID": TESTING_MODEL, "modelRevision": REV,
         "dtype": "bfloat16", "concepts": [], "jlensReadout": block}
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="testing-tier"):
        experiment_store._check_jlens_readout("s", d, root)

    # Editing the stored stamp is not a supported path, and the tier a NEW
    # entry carries is recomputed from the table rather than copied.
    reloaded = lens_store.resolve(record.lensID, root)
    reloaded.qualifications[0].tier = "evidence"
    lens_store.save(reloaded, root)
    fresh = schemas.Qualification(
        qualificationID="q-testing-2", modelID=TESTING_MODEL, revision=REV,
        dtype="bfloat16", tier=qualification.tier_for(TESTING_MODEL),
        passed=True)
    assert fresh.tier == "testing"
    # …and freeze reads the SUPPORTED table itself, so a doctored stamp buys
    # nothing.
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="testing-tier"):
        experiment_store._check_jlens_readout("s", d, root)


# --- append semantics --------------------------------------------------------

def test_qualifications_are_appended_never_replaced(tmp_path):
    """A re-run that overwrote its predecessor would erase a failure and leave
    a pass where the history said something else."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    first = _bound(record, qualificationID="q-1", modelID=EVIDENCE_MODEL,
                   revision=REV, dtype="bfloat16", passed=False)
    qualification.append_qualification(record, first, root=root)
    with pytest.raises(qualification.QualificationRefused,
                       match="appended, never replaced"):
        qualification.append_qualification(record, first, root=root)

    second = _bound(record, qualificationID="q-2", modelID=EVIDENCE_MODEL,
                    revision=REV, dtype="bfloat16", passed=True)
    qualification.append_qualification(record, second, root=root)
    reloaded = lens_store.resolve(record.lensID, root)
    assert [q.qualificationID for q in reloaded.qualifications] == ["q-1", "q-2"]
    # The failure stays readable; the gate looks for the PASSING one.
    assert reloaded.qualification_for(EVIDENCE_MODEL, REV,
                                      "bfloat16").qualificationID == "q-2"


def test_a_record_with_qualifications_round_trips(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    entry = schemas.Qualification(
        qualificationID="q-round", modelID=EVIDENCE_MODEL, revision=REV,
        dtype="bfloat16", quantization=None, tier="evidence", passed=True,
        checks={"checkNames": list(qualification.CHECKS),
                "blockingFailures": [],
                "results": {"geometry": {"passed": True, "detail": "ok",
                                         "measured": {"vocabSize": 262144}}}})
    qualification.append_qualification(record, entry, root=root)
    reloaded = lens_store.resolve(record.lensID, root)
    stored = reloaded.qualifications[0]
    assert stored.checks["results"]["geometry"]["measured"]["vocabSize"] == 262144
    assert stored.checks["checkNames"] == list(qualification.CHECKS)
    assert qualification.summarize(reloaded)[0]["qualificationID"] == "q-round"


def test_the_id_is_content_addressed_over_the_runtime(tmp_path):
    """Two runtimes must never collide onto one id, and the same runtime
    re-qualified later must not collide with its own earlier entry."""
    common = dict(lens_id="l", lens_sha="a" * 64, model_id=EVIDENCE_MODEL,
                  revision=REV, qualified_at="2026-08-15T00:00:00Z")
    bf16 = qualification.qualification_id(dtype="bfloat16", quantization=None,
                                          **common)
    fp16 = qualification.qualification_id(dtype="float16", quantization=None,
                                          **common)
    quantized = qualification.qualification_id(
        dtype="bfloat16", quantization="bitsandbytes_8bit", **common)
    later = qualification.qualification_id(
        **{**common, "qualified_at": "2026-08-16T00:00:00Z"},
        dtype="bfloat16", quantization=None)
    assert len({bf16, fp16, quantized, later}) == 4


# --- the gate ----------------------------------------------------------------

def _bound(record, **kw):
    """A qualification carrying the bindings `jlens qualify` writes. Unbound
    records license NOTHING since the second review — absent bindings used to
    read as "unconstrained"."""
    kw.setdefault("lensSHA256", record.source.tensorSHA256)
    kw.setdefault("convertedSHA256",
                  record.converted.sha256 if record.converted else None)
    kw.setdefault("layers", list(record.sourceLayers))
    return schemas.Qualification(**kw)


def test_a_geometry_identical_float16_runtime_does_not_pass_a_bf16_record(
        tmp_path):
    """Geometry cannot see dtype: a float16 27B has the same layer count,
    hidden size, vocabulary, and head shape as the bf16 one, and every shape
    check passes identically while the residual's numerics differ."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    qualification.append_qualification(
        record, _bound(record,
                       qualificationID="q-bf16", modelID=EVIDENCE_MODEL,
                       revision=REV, dtype="bfloat16", tier="evidence",
                       passed=True), root=root)

    block = {"lensID": record.lensID, "lensSHA256": record.source.tensorSHA256,
             "layers": [0, 1], "watchlist": [23648], "configHash": "c" * 64,
             "tokenizerHash": "t" * 64, "qualificationID": "q-bf16"}
    d = {"name": "s", "modelID": EVIDENCE_MODEL, "modelRevision": REV,
         "concepts": [], "jlensReadout": block}
    experiment_store._check_jlens_readout("s", {**d, "dtype": "bfloat16"}, root)
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification"):
        experiment_store._check_jlens_readout(
            "s", {**d, "dtype": "float16"}, root)
    # A quantized runtime of the identical geometry is likewise not covered.
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification"):
        experiment_store._check_jlens_readout(
            "s", {**d, "dtype": "bfloat16",
                  "quantization": "bitsandbytes_8bit"}, root)


def test_the_refusal_now_names_the_verb_that_would_fix_it(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    block = {"lensID": record.lensID, "lensSHA256": record.source.tensorSHA256,
             "layers": [0], "watchlist": [1], "configHash": "c" * 64,
             "tokenizerHash": "t" * 64, "qualificationID": "q-none"}
    with pytest.raises(experiment_store.ExperimentStoreError) as exc:
        experiment_store._check_jlens_readout(
            "s", {"name": "s", "modelID": EVIDENCE_MODEL,
                  "modelRevision": REV, "dtype": "bfloat16", "concepts": [],
                  "jlensReadout": block}, root)
    assert "jlens qualify" in str(exc.value)


# --- fixtures and constants --------------------------------------------------

def test_the_committed_prompt_fixtures_load_and_carry_their_roles():
    rows = qualification.load_fixture_prompts()
    assert rows and all(r.get("prompt") for r in rows)
    assert any(r.get("causalSmoke") for r in rows)
    assert any(r.get("carriesUnspokenIntermediate") for r in rows)
    # Expectations are deliberately absent until a 27B run measures them: a
    # check that passes against an invented expectation certifies nothing.
    assert not any(r.get("expectTopKPieces") for r in rows)


def test_the_capability_tolerance_matches_the_sweeps(tmp_path):
    """'The dose range is usable' must mean one thing across the engine."""
    from steerlab_server.experiment import sweep_selection

    assert qualification.CAPABILITY_TOLERANCE == \
        sweep_selection.DEFAULT_CAPABILITY_TOLERANCE


def test_every_check_is_blocking_and_named_in_order():
    """The check list is the artifact's contract: a record naming fewer checks
    than this was written by an older engine, and a reader can tell."""
    assert set(qualification.BLOCKING) == set(qualification.CHECKS)
    assert qualification.CHECKS[0] == "geometry"
    assert qualification.CHECKS[-1] == "capabilityGuard"


def test_qualifying_a_lens_fitted_on_another_model_refuses(tmp_path):
    """Nonsense rather than merely unqualified — and the two must not produce
    the same message."""
    root = str(tmp_path / "ws")
    _import(tmp_path, EVIDENCE_MODEL)
    with pytest.raises(qualification.QualificationRefused,
                       match="qualification was asked for"):
        qualification.qualify(importer.lens_id_for(EVIDENCE_MODEL),
                              TESTING_MODEL, root=root,
                              model=FakeModel())


def test_the_default_layers_sit_inside_the_paper_s_workspace_band(tmp_path):
    """The default used to be "middle third", which is CLAUDE.md's heuristic
    for CAA STEERING layers — a different question from where the lens READS.
    That put L20 of 62 (32% depth) below the workspace onset (~38%).

    Corroborated locally on gemma-3-4b-it (2026-08-15): layer 4 (12%) read as
    noise, layer 12 (35%) already carried abstract content.
    """
    record = _import(tmp_path, EVIDENCE_MODEL, layers=tuple(range(0, 61)))
    record.targetLayer = 61                       # 62-layer runtime
    picked = qualification._default_layers(record)
    assert all(l in record.sourceLayers for l in picked)
    low, high = qualification.WORKSPACE_BAND
    for layer in picked:
        assert low < layer / 62 < high, f"layer {layer} is outside the band"
    # Interior points, evenly spaced — a default should not sit on a boundary
    # where the functional regions blur.
    assert picked == [31, 40, 48]


def test_the_band_is_the_readout_band_not_the_steering_heuristic():
    """Pinned so the two never get conflated again: the workspace band starts
    at 38% of depth, well above the middle-third heuristic's 33%."""
    assert qualification.WORKSPACE_BAND == (0.38, 0.92)


def test_a_short_lens_qualifies_every_layer_it_has(tmp_path):
    record = _import(tmp_path, EVIDENCE_MODEL, layers=(0, 1, 2))
    assert qualification._default_layers(record) == [0, 1, 2]
