"""The resampling stability diagnostic — the pure function and its verb.

WHAT THESE TESTS ARE FOR. A stability number is only usable if a reader knows
what a HIGH one and a LOW one look like, so the first block calibrates the
instrument against clouds whose answer is known in advance: a shared direction
plus small noise must come back near 1.0 with no sign flips, and two
independent noise clouds must not. The second block pins the two determinism
promises the artifact makes (same seed → same draws; same seed → same draws AT
EVERY LAYER). The third pins the refusals. The fourth drives the verb with a
monkeypatched capture and a monkeypatched loader — the ``test_cli_backfill_
norms.py`` harness shape — so the envelope, the document, and the
never-writes-into-runs promise are all checked without a model.

THE VERB IS TESTED THROUGH A MONKEYPATCHED CAPTURE, not a tiny real model.
``extractor.activations`` is the seam the verb reuses and the seam a real model
would reach; substituting it exercises every line the verb owns — manifest
resolution, the method gate, the per-layer assembly, the document, the
envelope — while leaving the capture itself to the tests that already cover it
(``test_extraction_reading_position.py`` runs the real recorder on an in-memory
Llama). What a real model would add here is a slower assertion about code this
file does not change.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from steerlab_server import cli
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import extract_stability
from steerlab_server.steering import extractor, model_loader
from steerlab_server.steering import vector_math as vm

PAIRED_METHODS = (vm.ExtractionMethod.MEAN_DIFFERENCE,
                  vm.ExtractionMethod.PAIRED_DIFFERENCE_PCA)


def _signal_rows(*, rows=24, dimension=48, noise=0.15, seed=11):
    """A clean contrast: both classes share a base cloud, the positive class
    is displaced along ONE direction. Every draw sees the same displacement, so
    a correct diagnostic reports near-perfect stability."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dimension)
    direction /= np.linalg.norm(direction)
    base = rng.standard_normal((rows, dimension)) * noise
    positive = base + direction * 2.0
    negative = base + rng.standard_normal((rows, dimension)) * noise
    return positive.tolist(), negative.tolist()


def _noise_rows(*, rows=24, dimension=48, seed=5):
    """Two independent noise clouds — no shared contrast for a draw to
    recover, so the direction is whatever the drawn rows happened to say."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, dimension)).tolist(),
            rng.standard_normal((rows, dimension)).tolist())


# --- calibration: what a stable and an unstable reading look like ---------------


@pytest.mark.parametrize("method", PAIRED_METHODS)
def test_a_clean_contrast_is_stable_and_never_flips_sign(method):
    positive, negative = _signal_rows()
    result = vm.direction_stability(positive, negative, method, resamples=24,
                                    fraction=0.5, seed=3, order_shuffles=4)
    assert result.min_cosine > 0.9, result.resample_cosines
    assert result.percentile5_cosine > 0.9
    assert result.sign_flips == 0
    assert result.degenerate_draws == []
    assert len(result.resample_cosines) == 24
    assert result.subsample_size == 12
    assert result.paired is True


@pytest.mark.parametrize("method", PAIRED_METHODS)
def test_pure_noise_is_not_stable(method):
    positive, negative = _noise_rows()
    result = vm.direction_stability(positive, negative, method, resamples=24,
                                    fraction=0.4, seed=3)
    signal = vm.direction_stability(*_signal_rows(), method, resamples=24,
                                    fraction=0.4, seed=3)
    assert result.min_cosine < 0.8, result.resample_cosines
    # The instrument SEPARATES the two cases, which is the property a reader
    # relies on — an absolute threshold would be a calibration this diagnostic
    # deliberately does not ship.
    assert result.mean_cosine < signal.mean_cosine - 0.1


def test_the_summary_statistics_describe_the_resample_draws():
    positive, negative = _signal_rows()
    result = vm.direction_stability(
        positive, negative, vm.ExtractionMethod.MEAN_DIFFERENCE,
        resamples=20, fraction=0.5, seed=9, order_shuffles=3)
    values = sorted(result.resample_cosines)
    assert result.min_cosine == pytest.approx(values[0])
    assert result.mean_cosine == pytest.approx(sum(values) / len(values))
    assert result.median_cosine == pytest.approx(float(np.median(values)))
    assert result.percentile5_cosine == pytest.approx(
        float(np.percentile(np.asarray(values, dtype=np.float64), 5.0)))
    # The order shuffles are a SEPARATE population and never enter the summary.
    assert len(result.order_shuffle_cosines) == 3
    assert result.min_cosine <= min(values)


# --- determinism ----------------------------------------------------------------


@pytest.mark.parametrize("method", PAIRED_METHODS)
def test_the_same_seed_redraws_the_same_subsamples(method):
    positive, negative = _signal_rows()
    kwargs = dict(resamples=12, fraction=0.5, order_shuffles=3)
    first = vm.direction_stability(positive, negative, method, seed=41, **kwargs)
    again = vm.direction_stability(positive, negative, method, seed=41, **kwargs)
    other = vm.direction_stability(positive, negative, method, seed=42, **kwargs)
    assert first.to_dict() == again.to_dict()
    assert first.resample_seeds == again.resample_seeds
    assert first.resample_seeds != other.resample_seeds


def test_every_layer_draws_the_same_subsamples():
    """One seed, one draw of STIMULI. A per-layer draw would make "layer 20 is
    the unstable one" unfalsifiable, because layer 20 would have been asked a
    different question."""
    positive, negative = _signal_rows()
    shifted = (np.asarray(positive) * 3.0).tolist()
    by_layer = vm.stability_by_layer(
        {0: (positive, negative), 1: (shifted, negative)},
        vm.ExtractionMethod.MEAN_DIFFERENCE, resamples=8, fraction=0.5, seed=7)
    assert sorted(by_layer) == [0, 1]
    assert by_layer[0].resample_seeds == by_layer[1].resample_seeds
    assert by_layer[0].subsample_size == by_layer[1].subsample_size


# --- the order-shuffle control --------------------------------------------------


def test_row_order_cannot_move_a_mean_difference():
    """The control. A mean is order-invariant, so every shuffle must return
    exactly 1.0 — and a control that is only run where it is expected to fail
    is not a control."""
    positive, negative = _signal_rows()
    result = vm.direction_stability(
        positive, negative, vm.ExtractionMethod.MEAN_DIFFERENCE, resamples=4,
        fraction=0.5, seed=2, order_shuffles=6)
    assert len(result.order_shuffle_cosines) == 6
    assert all(c == pytest.approx(1.0, abs=1e-6)
               for c in result.order_shuffle_cosines)


def test_row_order_moves_the_paired_difference_pca():
    """The paired-difference PCA feeds its normalized differences in
    ALTERNATING ± orientation (``vector_math.direction``), so the row ORDER is
    a recipe input. On a noise cloud, reordering it moves the answer — which is
    the fact the ``lat`` recipe's write-up has to state and the mean-difference
    recipe's does not."""
    positive, negative = _noise_rows()
    result = vm.direction_stability(
        positive, negative, vm.ExtractionMethod.PAIRED_DIFFERENCE_PCA,
        resamples=4, fraction=0.5, seed=2, order_shuffles=8)
    assert len(result.order_shuffle_cosines) == 8
    assert min(result.order_shuffle_cosines) < 0.999


# --- refusals -------------------------------------------------------------------


def test_too_few_resamples_is_refused():
    positive, negative = _signal_rows()
    with pytest.raises(vm.SteeringVectorError, match="at least 2 draws"):
        vm.direction_stability(positive, negative,
                               vm.ExtractionMethod.MEAN_DIFFERENCE,
                               resamples=1, fraction=0.5, seed=0)


@pytest.mark.parametrize("fraction", [0.0, -0.2, 1.5])
def test_a_fraction_outside_the_unit_interval_is_refused(fraction):
    positive, negative = _signal_rows()
    with pytest.raises(vm.SteeringVectorError, match=r"must be in \(0, 1\]"):
        vm.direction_stability(positive, negative,
                               vm.ExtractionMethod.MEAN_DIFFERENCE,
                               resamples=4, fraction=fraction, seed=0)


def test_too_few_rows_to_draw_two_is_refused():
    positive, negative = _signal_rows(rows=3)
    with pytest.raises(vm.SteeringVectorError, match="at least 2"):
        vm.direction_stability(positive, negative,
                               vm.ExtractionMethod.MEAN_DIFFERENCE,
                               resamples=4, fraction=0.2, seed=0)


def test_a_method_that_never_reaches_direction_is_refused():
    positive, negative = _signal_rows()
    with pytest.raises(vm.SteeringVectorError, match="never reaches"):
        vm.direction_stability(positive, negative,
                               vm.ExtractionMethod.GRAND_MEAN,
                               resamples=4, fraction=0.5, seed=0)


def test_unpaired_classes_are_refused_for_the_paired_difference_pca():
    positive, _ = _signal_rows(rows=10)
    _, negative = _signal_rows(rows=7, seed=12)
    with pytest.raises(vm.SteeringVectorError, match="unpairedStimuli"):
        vm.direction_stability(positive, negative,
                               vm.ExtractionMethod.PAIRED_DIFFERENCE_PCA,
                               resamples=4, fraction=0.5, seed=0)


def test_unpaired_classes_are_diagnosable_for_a_designated_reference():
    """``designatedReference`` compares two story corpora that need not be the
    same length, so the draw is per class and the reading says so."""
    positive, _ = _signal_rows(rows=14)
    _, negative = _signal_rows(rows=9, seed=12)
    result = vm.direction_stability(
        positive, negative, vm.ExtractionMethod.DESIGNATED_REFERENCE,
        resamples=6, fraction=0.5, seed=1, order_shuffles=2)
    assert result.paired is False
    assert result.pair_count == 9
    assert all(c == pytest.approx(1.0, abs=1e-6)
               for c in result.order_shuffle_cosines)


def test_a_degenerate_full_direction_is_refused():
    rows = [[0.0, 0.0, 0.0] for _ in range(6)]
    with pytest.raises(vm.SteeringVectorError, match="zero norm"):
        vm.direction_stability(rows, rows,
                               vm.ExtractionMethod.MEAN_DIFFERENCE,
                               resamples=4, fraction=0.5, seed=0)


# --- the artifact's key set -----------------------------------------------------


def test_the_document_halves_partition_the_dataclass_keys():
    """The per-layer half and the shared half must cover the dataclass exactly:
    a field added to :class:`DirectionStability` and forgotten in the writer
    would otherwise vanish from the artifact with nothing to notice it."""
    positive, negative = _signal_rows()
    keys = set(vm.direction_stability(
        positive, negative, vm.ExtractionMethod.MEAN_DIFFERENCE, resamples=4,
        fraction=0.5, seed=0).to_dict())
    assert set(extract_stability.LAYER_KEYS).isdisjoint(
        extract_stability.SHARED_KEYS)
    assert set(extract_stability.LAYER_KEYS) | set(
        extract_stability.SHARED_KEYS) == keys


# --- the verb -------------------------------------------------------------------


LAYERS = 3
HIDDEN = 16


def _study(root: str, name: str = "stab", *, method: str = "meanDifference",
           rows: int = 8) -> None:
    """One draft experiment with one paired concept of ``rows`` pairs."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    os.makedirs(concept_dir, exist_ok=True)
    for pole, word in (("positive", "dread"), ("negative", "calm")):
        with open(os.path.join(concept_dir, f"{pole}.jsonl"), "w",
                  encoding="utf-8") as handle:
            for i in range(rows):
                handle.write(json.dumps({"text": f"{word} {i}"}) + "\n")
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], method=method, root=root)


def _harness(tmp_path, monkeypatch, *, rows=8):
    """Workspace root, a fake loader, and a fake capture at the extractor seam.

    The captured rows carry a real contrast so the reading is a reading and not
    a shape check: every text gets the same per-layer displacement, keyed off
    its own index for a little spread.
    """
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    rng = np.random.default_rng(4)
    direction = rng.standard_normal(HIDDEN)
    direction /= np.linalg.norm(direction)
    calls = []

    def fake_activations(model, texts, position, rendering=None):
        calls.append((tuple(texts), position.label,
                      (rendering or SimpleNamespace(mode="raw")).mode))
        positive = texts and texts[0].startswith("dread")
        values = []
        for i, _text in enumerate(texts):
            layers = []
            for layer in range(LAYERS):
                row = rng.standard_normal(HIDDEN) * 0.1
                if positive:
                    row = row + direction * (1.0 + 0.05 * i)
                layers.append([float(x) for x in row])
            values.append(layers)
        return extractor.StimulusActivations(
            values=values, residual_norm_per_layer=[1.0] * LAYERS,
            resolutions=[])

    monkeypatch.setattr(extractor, "activations", fake_activations)
    monkeypatch.setattr(
        model_loader, "load",
        lambda model_id, revision=None, **kwargs: SimpleNamespace(
            model_id=model_id, revision=revision or "abc", dtype="float32"))
    monkeypatch.setattr(model_loader, "resolve_device",
                        lambda device=None: device or "cpu")
    _study(str(tmp_path), rows=rows)
    return calls


def test_the_verb_writes_a_stamped_document_and_answers_in_the_envelope(
        tmp_path, monkeypatch, capsys):
    calls = _harness(tmp_path, monkeypatch)
    rc = cli.main(["experiment", "extract-stability", "stab", "fear",
                   "--resamples", "6", "--fraction", "0.5", "--seed", "5",
                   "--order-shuffles", "2", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    envelope = json.loads(captured.out)
    assert envelope["state"] == "ready"
    assert envelope["verb"] == "experiment extract-stability"
    result = envelope["result"]
    assert result["experiment"] == "stab"
    assert result["concept"] == "fear"
    assert result["layerCount"] == LAYERS
    assert result["worstLayer"] in range(LAYERS)
    assert result["recipeIdentityHash"]
    assert result["stimulusDrift"] is False
    assert vm.STABILITY_DIAGNOSTIC_NOTE in result["diagnosticNote"]

    # ONE capture per class, at the concept's own position and rendering.
    assert len(calls) == 2
    assert [c[1] for c in calls] == ["last token", "last token"]
    assert [c[2] for c in calls] == ["raw", "raw"]

    with open(result["path"], encoding="utf-8") as handle:
        document = json.load(handle)
    assert set(document) == {
        "buildCommit", "concept", "diagnosticNote", "engineVersion",
        "experiment", "experimentStatus", "extractionMethod",
        "extractionRendering", "layerCount", "layers", "modelID",
        "modelRevision", "neutralProjectionApplied", "observedAt",
        "readingPosition", "readingPositionMode", "readingPositionParameter",
        "readingPositionResolution", "recipeIdentityHash",
        "recipeIdentityUnprovable", "resample", "schema", "stimulus", "verb"}
    assert document["diagnosticNote"] == vm.STABILITY_DIAGNOSTIC_NOTE
    assert document["modelID"] == "org/m"
    assert document["modelRevision"] == "abc"
    assert document["extractionMethod"] == "meanDifference"
    assert document["neutralProjectionApplied"] is False
    assert document["engineVersion"].startswith("steerlab-server ")
    assert document["resample"] == {
        "fraction": 0.5, "method": "meanDifference", "orderShuffleSeeds":
            document["resample"]["orderShuffleSeeds"], "orderShuffles": 2,
        "paired": True, "pairCount": 8,
        "resampleSeeds": document["resample"]["resampleSeeds"],
        "resamples": 6, "seed": 5, "subsampleSize": 4}
    assert len(document["resample"]["resampleSeeds"]) == 6
    assert len(document["resample"]["orderShuffleSeeds"]) == 2
    assert [row["layer"] for row in document["layers"]] == list(range(LAYERS))
    for row in document["layers"]:
        assert set(row) == {"layer", *extract_stability.LAYER_KEYS}
        assert len(row["resampleCosines"]) == 6
        assert len(row["orderShuffleCosines"]) == 2
    assert document["stimulus"]["stimulusHashLive"] == \
        document["stimulus"]["stimulusHashPinned"]


def test_the_document_lands_under_diagnostics_and_nowhere_else(
        tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    before_runs = sorted(os.listdir(os.path.join(str(tmp_path), "runs"))) \
        if os.path.isdir(os.path.join(str(tmp_path), "runs")) else []
    rc = cli.main(["experiment", "extract-stability", "stab", "fear",
                   "--resamples", "4"])
    capsys.readouterr()
    assert rc == 0
    runs = os.path.join(str(tmp_path), "runs")
    after_runs = sorted(os.listdir(runs)) if os.path.isdir(runs) else []
    assert after_runs == before_runs
    diagnostics = os.path.join(str(tmp_path), "diagnostics")
    entries = sorted(os.listdir(diagnostics))
    assert len(entries) == 1 and entries[0].startswith("extract-stability-fear-")
    assert sorted(os.listdir(os.path.join(diagnostics, entries[0]))) == \
        ["stability.json"]


def test_two_readings_of_the_same_concept_do_not_collide(
        tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    for _ in range(2):
        assert cli.main(["experiment", "extract-stability", "stab", "fear",
                         "--resamples", "3"]) == 0
    capsys.readouterr()
    entries = os.listdir(os.path.join(str(tmp_path), "diagnostics"))
    assert len(entries) == 2, entries


def test_an_unknown_concept_is_a_typed_not_found(tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    rc = cli.main(["experiment", "extract-stability", "stab", "hope", "--json"])
    captured = capsys.readouterr()
    assert rc == 66, captured.out
    envelope = json.loads(captured.out)
    assert envelope["state"] == "notFound"
    assert envelope["error"]["code"] == "notFound"
    assert "fear" in envelope["error"]["repairAction"]


def test_a_grand_mean_concept_is_refused_before_any_model_load(
        tmp_path, monkeypatch, capsys):
    """The gate runs before the loader, so an unsupported recipe costs nothing.
    The loader is replaced by one that FAILS the test if it is reached."""
    _harness(tmp_path, monkeypatch)
    stories = os.path.join(str(tmp_path), "prompts", "emotions", "joy")
    os.makedirs(stories, exist_ok=True)
    with open(os.path.join(stories, "stories.jsonl"), "w",
              encoding="utf-8") as handle:
        for i in range(6):
            handle.write(json.dumps({"text": f"a joyful day {i}"}) + "\n")
    es.attach("stab", ["joy"], method="emotionGrandMean", root=str(tmp_path))

    def refuse_to_load(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the method gate must refuse before the load")

    monkeypatch.setattr(model_loader, "load", refuse_to_load)
    rc = cli.main(["experiment", "extract-stability", "stab", "joy", "--json"])
    captured = capsys.readouterr()
    assert rc == 65, captured.out
    envelope = json.loads(captured.out)
    assert envelope["error"]["code"] == "unsupportedMethod"
    assert "never reaches direction()" in envelope["error"]["reason"]


@pytest.mark.parametrize("flags,fragment", [
    (["--resamples", "1"], "at least 2 draws"),
    (["--fraction", "1.4"], "must be in (0, 1]"),
    (["--fraction", "0.1"], "there is nothing to resample"),
])
def test_bad_resample_parameters_are_blocked_before_the_load(
        tmp_path, monkeypatch, capsys, flags, fragment):
    _harness(tmp_path, monkeypatch)

    def refuse_to_load(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the parameter gate must refuse before the load")

    monkeypatch.setattr(model_loader, "load", refuse_to_load)
    rc = cli.main(["experiment", "extract-stability", "stab", "fear", "--json",
                   *flags])
    captured = capsys.readouterr()
    assert rc == 64, captured.out
    envelope = json.loads(captured.out)
    assert envelope["state"] == "blocked"
    assert fragment in envelope["error"]["reason"]


def test_a_non_numeric_flag_value_is_a_malformed_invocation(
        tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    rc = cli.main(["experiment", "extract-stability", "stab", "fear",
                   "--resamples", "half"])
    captured = capsys.readouterr()
    assert rc == 64
    assert "--resamples" in captured.err


def test_the_verb_refuses_an_undeclared_flag(tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    rc = cli.main(["experiment", "extract-stability", "stab", "fear",
                   "--bootstrap", "9"])
    captured = capsys.readouterr()
    assert rc == 64
    assert "--bootstrap" in captured.err


def test_a_missing_concept_positional_prints_the_usage_line(
        tmp_path, monkeypatch, capsys):
    _harness(tmp_path, monkeypatch)
    rc = cli.main(["experiment", "extract-stability", "stab"])
    captured = capsys.readouterr()
    assert rc == 64
    assert "extract-stability <name> <concept>" in captured.err
    assert "diagnostics/" in captured.err
