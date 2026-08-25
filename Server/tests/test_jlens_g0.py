"""The G0 1b gate: the split that keeps a small model from vetoing 27B, and
the arm independence that keeps one capability from taking the other down.

The gate's live half runs on a node with a model on it. What CI pins is the
half that decides what the numbers MEAN — which checks may block, which may
only inform, and which arm each licenses — because that is the part a live run
cannot check and the part the original AND-together wording got wrong.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.jlens import backend, g0, importer
from steerlab_server.jlens.qualification import CheckResult
from steerlab_server.jlens.readout import ReadoutConfig

EVIDENCE_MODEL = "google/gemma-3-27b-it"
TESTING_MODEL = "google/gemma-3-4b-it"


def _results(**overrides) -> dict:
    """Every check passing, unless a test says otherwise."""
    out = {name: CheckResult(name=name, passed=True).to_dict()
           for name in g0.MECHANICAL_CHECKS + g0.SCIENTIFIC_CHECKS}
    for name, passed in overrides.items():
        out[name] = CheckResult(name=name, passed=passed,
                                detail="planted").to_dict()
    return out


# --- the mechanical / scientific split ---------------------------------------

def test_the_split_is_the_schema_not_the_prose():
    """A reader must not be able to mistake a 4B scientific result for a gate.
    The sections are separate objects with independent verdicts."""
    assert set(g0.MECHANICAL_CHECKS) & set(g0.SCIENTIFIC_CHECKS) == set()
    assert "alignment" in g0.MECHANICAL_CHECKS
    assert "causalDoseResponse" in g0.SCIENTIFIC_CHECKS


def test_a_testing_tier_scientific_failure_is_informational(tmp_path):
    """The paper's own finding is that early-layer readouts are noise and some
    workspace-band cells resist interpretation even at Claude scale, so a 4B
    null is uninformative, not disqualifying. Blocking the 27B run on it would
    let a small model veto the only measurement that counts."""
    results = _results(causalDoseResponse=False, referenceReadouts=False)
    scientific = g0._verdict(results, g0.SCIENTIFIC_CHECKS, tier="testing",
                             scientific=True)
    assert scientific["verdict"] == "informational"
    assert scientific["failed"] == ["causalDoseResponse", "referenceReadouts"]
    assert "never block" in scientific["reason"]


def test_a_testing_tier_MECHANICAL_failure_still_fails(tmp_path):
    """A defect here is scale-independent and would waste cluster time, so the
    4B rehearsal blocks the 27B run on exactly these."""
    results = _results(alignment=False)
    mechanical = g0._verdict(results, g0.MECHANICAL_CHECKS, tier="testing",
                             scientific=False)
    assert mechanical["verdict"] == "fail"
    assert mechanical["failed"] == ["alignment"]


def test_an_evidence_tier_scientific_section_is_verdicted():
    scientific = g0._verdict(_results(), g0.SCIENTIFIC_CHECKS,
                             tier="evidence", scientific=True)
    assert scientific["verdict"] == "pass"


# --- the two arms ------------------------------------------------------------

def test_the_arms_verdict_independently():
    """Passing one and failing the other is a real outcome: a study may steer
    with J-lens directions while making no readout claims, or read without
    steering."""
    verdicts = g0._arm_verdicts(_results(referenceReadouts=False),
                               tier="evidence")
    assert verdicts["steering"]["verdict"] == "pass"
    assert verdicts["steering"]["licensed"] is True
    assert verdicts["readout"]["verdict"] == "fail"
    assert verdicts["conjunction"]["licensed"] is False


# --- the readout arm licenses READING; instruments declare their own plumbing

def test_the_readout_arm_is_the_shared_science_only():
    """It used to be (alignment, referenceReadouts, costMeasurement) — three
    checks about different objects under one verdict, so passing it licensed
    claims from the ONLINE trace while looking like a blanket licence."""
    assert g0.ARM_CHECKS["readout"] == g0.READOUT_SCIENCE
    assert "alignment" not in g0.ARM_CHECKS["readout"]
    assert "referenceReadouts" in g0.READOUT_SCIENCE


def test_an_alignment_failure_no_longer_sinks_a_slice_claim():
    """The offline slice uses none of the alignment machinery — you index
    positions directly. An alignment failure must fail the ONLINE instrument
    and leave the slice's own verdict intact."""
    verdicts = g0._arm_verdicts(_results(alignment=False), tier="evidence")
    assert verdicts["readout"]["verdict"] == "pass"          # fidelity intact
    assert verdicts["instruments"]["onlineTrace"]["plumbing"] == "fail"
    assert verdicts["instruments"]["onlineTrace"]["usable"] is False
    assert verdicts["instruments"]["offlineSlice"]["plumbing"] == "pass"
    assert verdicts["instruments"]["offlineSlice"]["usable"] is True
    # The steering arm still needs alignment — a run whose sampled tokens
    # changed under a read-only recorder invalidates what it was paired with.
    assert verdicts["steering"]["verdict"] == "fail"


def test_a_slice_positioning_failure_does_not_sink_the_online_trace():
    verdicts = g0._arm_verdicts(_results(slicePositioning=False),
                                tier="evidence")
    assert verdicts["instruments"]["offlineSlice"]["usable"] is False
    assert verdicts["instruments"]["onlineTrace"]["usable"] is True


def test_failing_the_shared_science_makes_NO_instrument_usable():
    """Both paths compute the identical readout, so if it means nothing here
    it means nothing however it was obtained."""
    verdicts = g0._arm_verdicts(_results(referenceReadouts=False),
                                tier="evidence")
    for name in g0.INSTRUMENT_CHECKS:
        assert verdicts["instruments"][name]["usable"] is False


def test_no_instrument_is_usable_at_testing_tier():
    """Both rest on the readout arm, which is null at testing tier."""
    verdicts = g0._arm_verdicts(_results(), tier="testing")
    for name in g0.INSTRUMENT_CHECKS:
        assert verdicts["instruments"][name]["usable"] is False


def test_slice_positioning_is_mechanical_not_scientific():
    """An off-by-one in position indexing is scale-independent, so a 4B
    failure blocks the 27B run exactly like the online aligner's."""
    assert "slicePositioning" in g0.MECHANICAL_CHECKS
    assert "slicePositioning" not in g0.SCIENTIFIC_CHECKS


def test_the_anti_lexical_control_can_sink_the_steering_arm_alone():
    """A monotonic dose response is fully consistent with the model merely
    SAYING the token more often. The control is what separates vocabulary from
    disposition, and failing it must fail the arm even with a clean curve."""
    verdicts = g0._arm_verdicts(_results(antiLexicalControl=False),
                                tier="evidence")
    assert verdicts["steering"]["verdict"] == "fail"
    assert verdicts["steering"]["failed"] == ["antiLexicalControl"]
    assert verdicts["readout"]["verdict"] == "pass"


def test_alignment_is_a_precondition_for_STEERING_and_the_online_trace():
    """Superseded 2026-08-15. This test used to assert alignment sank BOTH
    arms, which is what the old one-verdict readout arm did — and it was the
    defect: alignment is a property of the online recorder, and the offline
    slice does not use it. It remains a steering precondition (a run whose
    sampled tokens changed under a read-only recorder invalidates the readout
    it was paired with) and an ONLINE-trace precondition."""
    verdicts = g0._arm_verdicts(_results(alignment=False), tier="evidence")
    assert verdicts["steering"]["verdict"] == "fail"
    assert verdicts["instruments"]["onlineTrace"]["usable"] is False
    # …but reading itself is still licensed, via the slice.
    assert verdicts["readout"]["verdict"] == "pass"
    assert verdicts["instruments"]["offlineSlice"]["usable"] is True


def test_a_testing_tier_run_refuses_to_verdict_either_arm():
    """Null, not pass: both arms rest on scale-bound checks, and a stamped
    pass from 4B would be a licence a reader could not detect as false."""
    verdicts = g0._arm_verdicts(_results(), tier="testing")
    for arm in ("steering", "readout"):
        assert verdicts[arm]["verdict"] is None
        assert verdicts[arm]["licensed"] is False
        assert "prior, not a licence" in verdicts[arm]["reason"]
    assert verdicts["conjunction"]["licensed"] is False


def test_the_conjunction_cannot_be_licensed_while_informativeness_is_unknown():
    """It used to AND the two arms' booleans. But the conjunction is "the
    direction steers AND the readout says why", and the second half is exactly
    what no preregistered criterion decides yet — so a `true` there overstated
    the readout side no matter how the checks came out (external review round
    2). Everything passing is not enough."""
    for results in (_results(referenceReadouts=False),
                    _results(antiLexicalControl=False),
                    _results()):
        verdicts = g0._arm_verdicts(results, tier="evidence")
        assert verdicts["conjunction"]["licensed"] is False
    assert g0._arm_verdicts(_results(), tier="evidence")["conjunction"][
        "blockedBy"] == "readout.scientificInformativeness"


def test_the_readout_arm_emits_no_bare_licensed_boolean():
    """Downstream code reads booleans, not prose. The scoped field cannot be
    mistaken for the broader claim; the broader claim is explicitly unknown."""
    readout = g0._arm_verdicts(_results(), tier="evidence")["readout"]
    assert "licensed" not in readout
    assert readout["implementationFidelityPassed"] is True
    assert readout["scientificInformativeness"] == "unknown"
    # The steering arm keeps its licence: dose response, the anti-lexical
    # control and the resource figures are a substantive claim.
    assert g0._arm_verdicts(_results(), tier="evidence")["steering"]["licensed"]


# --- the endpoint ------------------------------------------------------------

def test_an_endpoint_file_round_trips_with_its_hash(tmp_path):
    """The endpoint is a measurement-side input, so the report has to be able
    to say which bytes were scored."""
    path = tmp_path / "endpoint.jsonl"
    path.write_text(json.dumps({"id": "cf-1", "prompt": "Award damages?",
                                "options": ["Yes", "No"],
                                "target": "Yes"}) + "\n")
    endpoint = g0.Endpoint.load(str(path))
    assert len(endpoint.rows) == 1 and endpoint.rows[0].target == "Yes"
    assert len(endpoint.hash) == 64


def test_the_gate_reads_the_STUDYS_choice_rows_not_a_parallel_format(tmp_path):
    """The gate parses the study's own ``choicePromptsFile`` with the study's
    own loader, so it can be pointed at the choice rows that already exist.

    Both study conventions must survive: ``text`` as an alias for ``prompt``,
    and ``target`` defaulting to ``options[0]``. A private parser would have
    honoured neither, and the gate would have measured a different endpoint
    than the study it licenses.
    """
    from steerlab_server.experiment import sweep_selection

    path = tmp_path / "cf-choices.jsonl"
    path.write_text(json.dumps({"id": "cf-1", "text": "Award damages?",
                                "options": ["Yes", "No"]}) + "\n")
    endpoint = g0.Endpoint.load(str(path))
    assert endpoint.rows[0].prompt == "Award damages?"
    assert endpoint.rows[0].target == "Yes"          # options[0]
    # Byte-identical to what the sweep would compute for the same file.
    _rows, digest = sweep_selection.load_choice_rows(str(path), str(path))
    assert endpoint.hash == digest


def test_an_endpoint_target_outside_its_options_refuses(tmp_path):
    path = tmp_path / "endpoint.jsonl"
    path.write_text(json.dumps({"id": "bad", "prompt": "p",
                                "options": ["Yes", "No"],
                                "target": "Maybe"}) + "\n")
    with pytest.raises(g0.G0Error, match="not one of its options"):
        g0.Endpoint.load(str(path))


def test_an_endpoint_row_with_too_few_options_refuses(tmp_path):
    path = tmp_path / "endpoint.jsonl"
    path.write_text(json.dumps({"id": "bad", "prompt": "p",
                                "options": ["Yes"]}) + "\n")
    with pytest.raises(g0.G0Error, match="at least 2 options"):
        g0.Endpoint.load(str(path))


def test_without_an_endpoint_the_steering_arm_is_skipped_not_assumed(tmp_path):
    """Marker density is NOT a substitute — it is the surface-prose confound
    the control exists to detect — so an undeclared endpoint leaves the arm
    unmeasured rather than measured by something else."""
    dose, control = g0._steering_arm_science(
        record=None, model=None, readout=None, layer=0, token_id=1, piece=" x",
        endpoint=None, ladder=(0.04,), carrier="", root=None)
    for result in (dose, control):
        assert result.skipped is True and result.passed is False
        assert "Marker density is NOT a substitute" in result.detail


# --- the preflight-mechanics check -------------------------------------------

def test_the_preflight_check_requires_an_over_budget_config_to_refuse():
    """A preflight that produced numbers and let everything through would be a
    decoration. The compute ceiling is the one that binds."""
    config = ReadoutConfig(layers=[5, 9], watchlist=[10, 20], topK=10,
                           topKLayers=[5, 9], logitLensCompanion=True)
    result = g0._check_preflight_mechanics(config)
    assert result.passed is True
    estimate = result.measured["estimate"]
    assert estimate["steps"] == 50_000
    # Two armed top-k layers, doubled by the companion.
    assert estimate["fullVocabProjections"] == 200_000
    assert result.measured["refusalProblems"]


# --- the acquisition check ---------------------------------------------------

def _import(tmp_path, model_id, *, layers=(0, 1, 2), d_model=8):
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=d_model,
                        source_layers=list(layers)).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {model_id}\n")
    return importer.import_lens(model_id, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


def test_acquisition_rehashes_the_derived_cache(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    assert g0._check_acquisition(record, root=root).passed is True

    # A derived cache rebuilt from different upstream bytes keeps its path and
    # its record entry, and every number it produces would be different.
    from steerlab_server.experiment import paths

    with open(paths.resolve(record.converted.path, root), "ab") as handle:
        handle.write(b"\0")
    result = g0._check_acquisition(record, root=root)
    assert result.passed is False
    assert "hash drifted" in result.detail


def test_load_path_reads_one_layer_at_a_time(tmp_path):
    """The access pattern IS the feature: at 27B the whole lens is ~6.6 GiB
    promoted and one layer is ~58 MB."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)

    class Model:
        hidden_size = 8

    result = g0._check_load_path(record, Model(), root=root)
    assert result.passed is True
    assert result.measured["layersLoaded"] == 3
    assert result.measured["wholeLensBytesIfPromoted"] > \
        result.measured["perLayerBytes"]


def test_load_path_notices_a_width_mismatch(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)

    class Model:
        hidden_size = 4096

    result = g0._check_load_path(record, Model(), root=root)
    assert result.passed is False and "hidden size" in result.detail


# --- the resolution check's gain convention ----------------------------------
#
# The gate here used to be `gain_min <= 0`, which could not pass on ANY
# supported model: "gain" is not a lens-artifact field at all — it is read live
# from the RUNTIME's final RMSNorm as 1 + norm.weight — and Gemma-3's gammas go
# negative at every published size (4B min -0.0546875 with 2/2560 non-positive
# dims, 12B -0.117 with 8/3840, 27B -0.25). The sign gate indicted Google's
# weights. What replaces it are the convention checks it was standing in for.

D_MODEL_FOR_GAIN = 256          # wide enough that "a handful" is a fraction


def _resolution_model(num_layers=4):
    return type("Runtime", (), {"num_layers": num_layers})()


def _readout_with_gain(gain, softcap=None):
    return type("Readout", (), {"gain": gain, "softcap": softcap})()


def _gemma_shaped_gain(width=D_MODEL_FOR_GAIN, negatives=2):
    """A gain shaped like a real one: a body around 7-10, a long right tail,
    and the model's own handful of dimensions below zero (measured on
    gemma-3-4b-it: min -0.0546875, max 53.5, median 7.19)."""
    gain = torch.full((width,), 7.19)
    gain[0] = 53.5
    for i in range(negatives):
        gain[1 + i] = -0.0546875
    return gain


def test_a_realistic_gain_with_a_few_negative_dims_PASSES(tmp_path):
    """The 4B/12B/27B runtimes all present one. A check they cannot pass is not
    a check."""
    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    result = g0._check_resolution(record, _resolution_model(),
                                  _readout_with_gain(_gemma_shaped_gain()))
    assert result.passed is True
    assert result.measured["gainMin"] < 0            # and it does not matter
    assert result.measured["gainNonPositive"] == 2
    assert result.measured["gainWidth"] == D_MODEL_FOR_GAIN
    # The diagnostics survive the gate's removal — they were never the problem.
    assert result.measured["gainMax"] == pytest.approx(53.5)


def test_the_sign_gate_is_GONE_and_its_fixture_now_passes(tmp_path):
    """Pinned against reintroduction. This is exactly the 4B fixture the old
    `gain_min <= 0` failed on, and it is a healthy runtime."""
    import inspect

    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    # The BODY, not the docstring — which names the deleted gate on purpose,
    # so a later reader knows what was removed and why.
    body = inspect.getsource(g0._check_resolution).split('"""')[2]
    assert "gain_min <= 0" not in body
    assert "non-positive entries (min" not in body

    result = g0._check_resolution(record, _resolution_model(),
                                  _readout_with_gain(_gemma_shaped_gain()))
    assert result.passed is True
    # The negative dims are REPORTED, in the passing detail, rather than judged.
    assert "2 non-positive" in result.detail


def test_a_gain_that_is_ENTIRELY_non_positive_fails(tmp_path):
    """The realistic sign defect: the stored weight read as the gain itself, so
    `1 + w` was never applied and the whole vector flips."""
    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    gain = torch.full((D_MODEL_FOR_GAIN,), -0.5)
    result = g0._check_resolution(record, _resolution_model(),
                                  _readout_with_gain(gain))
    assert result.passed is False
    assert "whole-vector sign flip" in result.detail
    assert result.measured["gainNonPositive"] == D_MODEL_FOR_GAIN


def test_an_all_zero_gain_fails_as_a_dead_read(tmp_path):
    """An unloaded or zeroed norm buffer reads as a gain of exactly nothing;
    saying 'sign' about it would name the wrong defect."""
    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    result = g0._check_resolution(
        record, _resolution_model(),
        _readout_with_gain(torch.zeros(D_MODEL_FOR_GAIN)))
    assert result.passed is False
    assert "all zeros" in result.detail


def test_a_non_positive_POPULATION_fails_where_a_handful_does_not(tmp_path):
    """The measured populations are 0.08% (4B) and 0.21% (12B). A tenth of the
    dimensions below zero is a different object, not a noisier one."""
    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    many = _gemma_shaped_gain(negatives=26)          # ~10%
    result = g0._check_resolution(record, _resolution_model(),
                                  _readout_with_gain(many))
    assert result.passed is False
    assert "degeneracy limit" in result.detail
    assert result.measured["gainNonPositiveFraction"] > \
        g0.GAIN_NONPOSITIVE_FRACTION


def test_a_gain_of_the_wrong_WIDTH_fails(tmp_path):
    """The bug the sign gate was reaching for: a gain resolved off some other
    norm module. It shows up as a width, never as a sign."""
    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    result = g0._check_resolution(
        record, _resolution_model(),
        _readout_with_gain(_gemma_shaped_gain(width=D_MODEL_FOR_GAIN // 2)))
    assert result.passed is False
    assert "d_model" in result.detail and "norm module" in result.detail


def test_an_unresolved_gain_still_fails(tmp_path):
    """Nothing about the sign ruling makes 'no gain at all' acceptable."""
    record = _import(tmp_path, EVIDENCE_MODEL, d_model=D_MODEL_FOR_GAIN)
    result = g0._check_resolution(record, _resolution_model(),
                                  _readout_with_gain(None))
    assert result.passed is False
    assert "no final-norm gain" in result.detail


def test_the_docstring_states_the_RANGE_not_the_distributions_body():
    """It quoted '~6.6-9.5' as the range; the measured 4B span is -1.055..53.5
    (median 7.19). Reading the body as the range is what made a sign gate look
    reasonable in the first place."""
    doc = g0._check_resolution.__doc__
    assert "53.5" in doc and "1.055" in doc
    assert "1 + norm.weight" in doc


# --- thresholds are declared, not inferred -----------------------------------

def test_the_control_thresholds_are_named_constants():
    """Both halves of 'the endpoint moved and the frequency did not' need a
    stated threshold, or the verdict is a vibe."""
    assert g0.ENDPOINT_MOVEMENT_NATS > 0
    assert 0 < g0.FREQUENCY_MOVEMENT_FRACTION < 1


# --- measurements (stamped outputs, never verdicts) --------------------------

def test_the_measurements_are_not_checks():
    """Neither has a threshold anyone could justify. The resolution limit is a
    property of the runtime; the band is a study-design input. Making either a
    pass/fail would be inventing a number."""
    for name in ("replayFidelity", "readableBand"):
        assert name not in g0.MECHANICAL_CHECKS
        assert name not in g0.SCIENTIFIC_CHECKS
        for names in g0.ARM_CHECKS.values():
            assert name not in names


def test_token_rank_is_the_band_statistic_because_top1_is_not():
    """Regression for a measurement that would have shipped saying nothing.

    Untrained/reserved vocabulary entries carry large unembedding norms, and
    since logit = ‖z‖·‖g⊙u_t‖·cos(...) that constant lifts them at every layer
    and position. Measured on gemma-3-4b-it, the lens ranked ' Paris' FIFTH for
    "the capital city of France is" — behind 'ꗜ', ' PLDNN', '<unused338>' —
    while the model's own argmax was ' Paris'. A top-1 band chart read 0.00 at
    every layer including the motor end.
    """
    import inspect

    from steerlab_server.jlens.readout import LensReadout

    assert hasattr(LensReadout, "token_rank")
    band = inspect.getsource(g0._measure_readable_band)
    assert "token_rank" in band
    assert "medianNextTokenRank" in band
    # top-1 is still reported, but must not be the headline the caller reads.
    assert "nextTokenTop1" in band
    summary = inspect.getsource(g0.run)
    assert "medianNextTokenRank" in summary


def test_the_band_sweep_can_be_disabled():
    """It costs a full-vocabulary projection per (layer, position); a run that
    only wants the gate should not pay for it."""
    import inspect

    assert g0.DEFAULT_BAND_STRIDE > 0
    assert "band_stride" in inspect.signature(g0.run).parameters
    assert "--band-stride" in inspect.getsource(
        __import__("steerlab_server.cli", fromlist=["_jlens"])._jlens)


def test_the_readout_licence_says_what_it_does_not_establish():
    """`referenceReadouts` passes on numerical agreement with the reference and
    finite fixture readouts. The fixtures carry no declared expectations — by
    design, since inventing them would fabricate an acceptance criterion — so
    the unspoken-intermediate observations are RECORDED, never evaluated.

    An evidence-tier run could therefore 'license reading' without showing the
    readout anticipates anything (external review, 2026-08-16). The verdict now
    states its own scope rather than being read as the broader claim."""
    verdicts = g0._arm_verdicts(_results(), tier="evidence")
    licence = verdicts["readout"]["licences"]
    assert "implementation fidelity" in licence
    assert "NOT a claim" in licence
    assert "readableBand" in licence


def test_the_unspoken_intermediate_fixture_carries_no_expectations():
    """Pinned so the licence text and the fixture cannot drift apart: if
    expectations are ever added, the verdict's scope must widen with them."""
    from steerlab_server.jlens.qualification import load_fixture_prompts

    rows = load_fixture_prompts()
    assert any(r.get("carriesUnspokenIntermediate") for r in rows)
    assert not any(r.get("expectTopKPieces") for r in rows)
