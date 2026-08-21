"""The residual-norm DENOMINATOR CONVENTION — whole-corpus average, stamped.

Cross-engine twin: ``Tests/SteeringKitTests/ResidualNormConventionTests.swift``.
Both suites drive the SAME fixture (``FIXTURE_*`` below / ``fixture*`` there)
and assert the same three numbers, so a divergence in the averaging rule fails
on whichever engine drifted rather than surfacing months later as an
uncomparable alpha.

The fixture is chosen so the banked-positions mean and the whole-corpus mean
are DIFFERENT numbers: a test that passed under either rule would not have
caught the bug this convention closes.
"""

from __future__ import annotations

import json

import pytest

from steerlab_server.steering import residual_norm_convention as convention
from steerlab_server.steering import vector_store
from steerlab_server.steering.residual_norm_convention import ResidualNormTally

# ---------------------------------------------------------------- the fixture
#
# One layer, eight measured token positions. The row cap banked the four SMALL
# ones and excluded the four LARGE ones — the realistic shape, because the draw
# is blind to magnitude and any imbalance biases a banked-only mean.
FIXTURE_LAYER = 3
FIXTURE_BANKED = [10.0, 12.0, 14.0, 16.0]
FIXTURE_SKIPPED = [40.0, 44.0, 48.0, 52.0]

#: mean(banked) = 52/4 — the SUPERSEDED rule this engine used before 2026-08-20.
FIXTURE_BANKED_ONLY_MEAN = 13.0
#: mean(banked + skipped) = 236/8 — THE CONVENTION.
FIXTURE_WHOLE_CORPUS_MEAN = 29.5


def _tally() -> ResidualNormTally:
    tally = ResidualNormTally()
    for norm in FIXTURE_BANKED:
        tally.add(FIXTURE_LAYER, norm)
    for norm in FIXTURE_SKIPPED:
        tally.add(FIXTURE_LAYER, norm)
    return tally


def test_fixture_actually_discriminates_the_two_rules():
    """Guard on the guard: if these ever coincide, the suite below proves
    nothing."""
    assert FIXTURE_BANKED_ONLY_MEAN != FIXTURE_WHOLE_CORPUS_MEAN


def test_tally_averages_over_the_whole_corpus_not_the_draw():
    tally = _tally()
    assert tally.mean(FIXTURE_LAYER) == pytest.approx(FIXTURE_WHOLE_CORPUS_MEAN)
    # And is NOT the banked-only number the pre-ruling code produced.
    assert tally.mean(FIXTURE_LAYER) != pytest.approx(FIXTURE_BANKED_ONLY_MEAN)
    assert tally.count(FIXTURE_LAYER) == len(FIXTURE_BANKED) + len(FIXTURE_SKIPPED)
    assert tally.total_count == len(FIXTURE_BANKED) + len(FIXTURE_SKIPPED)
    assert tally.layers == [FIXTURE_LAYER]


def test_banked_only_reproduces_the_superseded_number():
    """Feeding ONLY the banked rows reproduces the old divergent value —
    which is what makes the assertion above a real discrimination and not an
    accident of the fixture."""
    tally = ResidualNormTally()
    for norm in FIXTURE_BANKED:
        tally.add(FIXTURE_LAYER, norm)
    assert tally.mean(FIXTURE_LAYER) == pytest.approx(FIXTURE_BANKED_ONLY_MEAN)


def test_unmeasured_layer_has_no_denominator():
    """0, never a borrowed neighbour: the downstream ``> 0`` guards refuse it
    rather than steering against a made-up denominator."""
    assert ResidualNormTally().mean(99) == 0.0


def test_stamp_is_the_pinned_cross_engine_string():
    assert convention.CURRENT == "wholeCorpusMean-v1"


def test_bank_driver_folds_the_excluded_positions_in():
    """The DRIVER, not just the tally: ``neutral_activation_bank`` must
    consume ``recorder.skipped_norms``. Driven with a stub model so no GPU or
    weights are needed."""
    from steerlab_server.steering import extractor

    bank = extractor.neutral_activation_bank(
        _StubModel(), ["aaaa", "bbbb"],
        reading_position=_position(),
        max_token_rows=2,           # 8 positions, cap 2 → 6 excluded
        downsample_seed=7)

    # Every position counted, not just the two banked ones.
    assert bank.token_row_count == 8
    assert bank.positions_total == 8
    assert bank.positions_kept == 2
    # The stub's norms are 1..8 across the two texts, so the whole-corpus mean
    # is 4.5 whichever two positions the draw happened to keep.
    assert bank.residual_norm_per_layer == [pytest.approx(4.5)]


def test_extraction_result_carries_the_convention():
    from steerlab_server.steering import extractor

    result = extractor.ExtractionResult(
        vectors=vector_store.ConceptVectors(per_layer=[[1.0, 0.0]]),
        residual_norm_per_layer=[1.0], residual_norm_source="neutral-corpus",
        options=extractor.ExtractionOptions())
    assert result.residual_norm_convention == convention.CURRENT


# --------------------------------------------------------------- the stamping


def _sidecar(**kwargs) -> vector_store.SteeringVectorSidecar:
    return vector_store.SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h",
        vectors=vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]]),
        **kwargs)


def test_fresh_measurement_stamps_the_convention(tmp_path):
    sidecar = _sidecar(residual_norm_per_layer=[2.0, 3.0],
                       residual_norm_source="neutral-corpus abc123",
                       residual_norm_convention=convention.CURRENT)
    vector_store.save(vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]]),
                      sidecar, str(tmp_path), "v")
    payload = json.loads((tmp_path / "v.json").read_text())
    assert payload["residualNormConvention"] == "wholeCorpusMean-v1"


def test_unstamped_is_ABSENT_not_null(tmp_path):
    """Family convention: an unknown field is omitted, never written as
    ``null``. A legacy artifact must be indistinguishable on the wire from
    one written before the field existed."""
    sidecar = _sidecar(residual_norm_per_layer=[2.0, 3.0],
                       residual_norm_source="extraction-stimuli")
    vector_store.save(vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]]),
                      sidecar, str(tmp_path), "v")
    payload = json.loads((tmp_path / "v.json").read_text())
    assert "residualNormConvention" not in payload


def test_legacy_sidecar_reads_exactly_as_before(tmp_path):
    """The never-retro-applied rule: a sidecar written before the stamp
    existed decodes with no convention, keeps its norms untouched, and raises
    nothing. No migration, no recompute, no warning storm."""
    legacy = {
        "schemaVersion": 2, "modelID": "m", "concept": "c",
        "stimulusSetHash": "h", "layerCount": 2, "hiddenSize": 2,
        "normsPerLayer": [1.0, 1.0], "extractionDate": "2026-01-01T00:00:00Z",
        "residualNormPerLayer": [2.0, 3.0],
        "residualNormSource": "neutral-corpus deadbeef",
    }
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    from safetensors.numpy import save_file
    import numpy as np
    save_file({"layer_0": np.asarray([1.0, 0.0], dtype=np.float32),
               "layer_1": np.asarray([0.0, 1.0], dtype=np.float32)},
              str(tmp_path / "legacy.safetensors"))

    _vectors, sidecar = vector_store.load(str(tmp_path), "legacy")
    assert sidecar.residualNormConvention is None
    assert sidecar.residualNormPerLayer == [2.0, 3.0]
    assert sidecar.residualNormSource == "neutral-corpus deadbeef"
    # And the display rule names it honestly rather than guessing a rule.
    assert convention.display_label(
        sidecar.residualNormPerLayer, sidecar.residualNormConvention
    ) == "legacy (pre-stamp)"


def test_display_label_rules():
    assert convention.display_label([1.0], convention.CURRENT) == "wholeCorpusMean-v1"
    assert convention.display_label([1.0], None) == "legacy (pre-stamp)"
    # No norms at all: there is no denominator to describe.
    assert convention.display_label(None, None) is None
    assert convention.display_label([], convention.CURRENT) is None


# ------------------------------------------------------------------ the stubs


def _position():
    from steerlab_server.steering.reading_position import mean_from_token
    return mean_from_token(0)


class _StubHooked:
    def __init__(self):
        self._interventions: list = []

    def session(self, interventions):
        outer = self

        class _Ctx:
            def __enter__(self):
                outer._interventions = list(interventions)
                return outer

            def __exit__(self, *_exc):
                outer._interventions = []
                return False

        return _Ctx()

    def reset_offsets(self):
        pass


class _StubTokenizer:
    """Four tokens per text, so two texts give eight bankable positions."""

    def __call__(self, _text, return_tensors=None):
        import torch

        class _Encoded:
            input_ids = torch.zeros((1, 4), dtype=torch.long)

        return _Encoded()


class _StubModel:
    """Emits rows whose L2 norms are 1,2,3,4 then 5,6,7,8 — a corpus whose
    mean (4.5) is stable under any draw, so the assertion isolates WHICH
    positions were counted rather than which were kept."""

    num_layers = 1
    hidden_size = 2
    device = "cpu"

    def __init__(self):
        self.hooked = _StubHooked()
        self.tokenizer = _StubTokenizer()
        self._pass = 0

    def model(self, input_ids=None, use_cache=False):
        import torch

        base = self._pass * 4
        self._pass += 1
        # [1, 4, 2]; row i has norm (base + i + 1) via (n, 0).
        values = torch.tensor(
            [[[float(base + i + 1), 0.0] for i in range(4)]], dtype=torch.float32)
        for intervention in self.hooked._interventions:
            intervention.apply(values, 0, 0)
        return None
