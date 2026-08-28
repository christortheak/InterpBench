"""The residual-norm DENOMINATOR CONVENTION — two averaging rules, two stamps.

Cross-engine twin: ``Tests/SteeringKitTests/ResidualNormConventionTests.swift``.
Both suites drive the SAME fixtures (``FIXTURE_*`` below / ``fixture*`` there)
and assert the same numbers, so a divergence in an averaging rule fails on
whichever engine drifted rather than surfacing months later as an uncomparable
alpha.

Two fixtures, each chosen so that a test passing under either rule would prove
nothing:

* the TALLY fixture separates the whole-corpus mean from the banked-only mean
  the server used before the 2026-08-20 ruling;
* the POOLED fixture (``FIXTURE_WINDOWS``) separates the two rules that shared
  one stamp until the 2026-08-28 audit (F1) — variable-length texts read at a
  pooled position, where the per-position mean and the mean of per-text
  window-means are different numbers.
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


def test_stamps_are_the_pinned_cross_engine_strings():
    """Twin literals. Each string names ONE averaging rule; the version suffix
    moves only when that rule does."""
    assert convention.WHOLE_CORPUS_MEAN == "wholeCorpusMean-v1"
    assert convention.PER_TEXT_MEAN == "perTextMean-v1"
    assert convention.WHOLE_CORPUS_MEAN != convention.PER_TEXT_MEAN


def test_the_legacy_stamp_is_grandfathered_onto_the_per_text_rule():
    """Every sidecar on disk stamped ``wholeCorpusMean-v1`` was written by a
    PER-TEXT writer — extraction or backfill — because the tally has never
    reached a sidecar. Frozen bytes are never rewritten, so the reader-side
    rule is that such a stamp means the per-text number it has always meant."""
    assert convention.GRANDFATHERED_PER_TEXT_STAMP == convention.WHOLE_CORPUS_MEAN


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


def test_extraction_result_carries_the_per_text_rule():
    """``extract`` measures every denominator through ``activations``, so the
    rule it stamps is the per-text one — not the tally's."""
    from steerlab_server.steering import extractor

    result = extractor.ExtractionResult(
        vectors=vector_store.ConceptVectors(per_layer=[[1.0, 0.0]]),
        residual_norm_per_layer=[1.0], residual_norm_source="neutral-corpus",
        options=extractor.ExtractionOptions())
    assert result.residual_norm_convention == convention.PER_TEXT_MEAN
    assert extractor.MultiConceptExtractionResult(
        per_concept={}, residual_norm_per_layer=[1.0],
        residual_norm_source="neutral-corpus"
    ).residual_norm_convention == convention.PER_TEXT_MEAN


def test_the_bank_states_the_rule_its_denominator_was_measured_under():
    """The tally's number is the PER-POSITION one, and the object it travels
    on says so — so a future writer stamps the rule it was handed instead of
    the extraction path's."""
    from steerlab_server.steering import extractor

    bank = extractor.NeutralActivationBank(
        layers=[0], rows_by_layer=[[[1.0]]], residual_norm_per_layer=[1.0],
        token_row_count=1)
    assert bank.residual_norm_convention == convention.WHOLE_CORPUS_MEAN


# ------------------------------------------------- the two rules, discriminated
#
# THE fixture the audit asked for (F1): variable-length texts at a POOLED
# reading position. Text A is read over 2 positions of norm 10; text B over 6
# positions of norm 2.
#
#   per-TEXT mean     = (10 + 2) / 2            = 6.0
#   per-POSITION mean = (10·2 + 2·6) / 8        = 4.0
#
# Both numbers are produced by REAL code below — ``extractor.activations`` for
# the per-text rule, ``extractor.neutral_activation_bank``'s tally for the
# per-position one — so the test pins the rules, not a restatement of them.
FIXTURE_WINDOWS = ((2, 10.0), (6, 2.0))
FIXTURE_POOLED_PER_TEXT_MEAN = 6.0
FIXTURE_POOLED_PER_POSITION_MEAN = 4.0


def test_pooled_fixture_actually_discriminates_the_two_rules():
    """Guard on the guard, again: equal-length texts (the shape the pre-audit
    fixtures used) cannot tell the two rules apart, which is exactly why no
    test caught the shared stamp."""
    assert FIXTURE_POOLED_PER_TEXT_MEAN != FIXTURE_POOLED_PER_POSITION_MEAN
    assert len({length for length, _ in FIXTURE_WINDOWS}) > 1


def test_activations_measure_the_per_text_rule_at_a_pooled_position():
    from steerlab_server.steering import extractor

    measured = extractor.activations(
        _VariableStubModel(), _variable_texts(), _position())
    assert measured.residual_norm_per_layer == \
        pytest.approx([FIXTURE_POOLED_PER_TEXT_MEAN])
    # And is NOT the per-position number, under a stamp that used to claim it.
    assert measured.residual_norm_per_layer != \
        pytest.approx([FIXTURE_POOLED_PER_POSITION_MEAN])


def test_the_tally_measures_the_per_position_rule_on_the_same_corpus():
    from steerlab_server.steering import extractor

    bank = extractor.neutral_activation_bank(
        _VariableStubModel(), _variable_texts(),
        reading_position=_position(), max_token_rows=None)
    assert bank.token_row_count == sum(n for n, _ in FIXTURE_WINDOWS)
    assert bank.residual_norm_per_layer == \
        pytest.approx([FIXTURE_POOLED_PER_POSITION_MEAN])


def test_the_two_rules_agree_at_a_single_position_reading():
    """Why the grandfathering is safe: at ``lastToken`` every text contributes
    exactly one position, so per-text and per-position weighting are the same
    arithmetic and a legacy stamp names a number both rules produce."""
    from steerlab_server.steering import extractor
    from steerlab_server.steering.reading_position import LAST_TOKEN

    measured = extractor.activations(
        _VariableStubModel(), _variable_texts(), LAST_TOKEN)
    tally = ResidualNormTally()
    for length, norm in FIXTURE_WINDOWS:
        # The last position of each text — one per text, either way round.
        tally.add(0, norm if length else 0.0)
    assert measured.residual_norm_per_layer == pytest.approx([tally.mean(0)])


# --------------------------------------------------------------- the stamping


def _sidecar(**kwargs) -> vector_store.SteeringVectorSidecar:
    return vector_store.SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h",
        vectors=vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]]),
        **kwargs)


def test_fresh_measurement_stamps_the_convention(tmp_path):
    sidecar = _sidecar(residual_norm_per_layer=[2.0, 3.0],
                       residual_norm_source="neutral-corpus abc123",
                       residual_norm_convention=convention.PER_TEXT_MEAN)
    vector_store.save(vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]]),
                      sidecar, str(tmp_path), "v")
    payload = json.loads((tmp_path / "v.json").read_text())
    assert payload["residualNormConvention"] == "perTextMean-v1"


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
    # Both stamps display as themselves — the ONLY consumer of the string on
    # either engine is display/provenance (this label, the α-default
    # convention note, the OptVec packaging advisory). Nothing gates on it, so
    # a new-stamp artifact and an old-stamp artifact cannot refuse against
    # each other, and a reader sees which rule they hold rather than a
    # normalized fiction.
    assert convention.display_label([1.0], convention.PER_TEXT_MEAN) == "perTextMean-v1"
    assert convention.display_label(
        [1.0], convention.WHOLE_CORPUS_MEAN) == "wholeCorpusMean-v1"
    assert convention.display_label([1.0], None) == "legacy (pre-stamp)"
    # No norms at all: there is no denominator to describe.
    assert convention.display_label(None, None) is None
    assert convention.display_label([], convention.PER_TEXT_MEAN) is None


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


def _variable_texts() -> list[str]:
    """One text per ``FIXTURE_WINDOWS`` entry, its LENGTH IN CHARACTERS equal
    to its token count — the stub tokenizer's whole contract."""
    return ["x" * length for length, _ in FIXTURE_WINDOWS]


class _VariableTokenizer:
    """One token per character, so the fixture's texts tokenize to different
    lengths — the shape at which the two averaging rules part company."""

    def __call__(self, text, return_tensors=None):
        import torch

        class _Encoded:
            input_ids = torch.zeros((1, len(text)), dtype=torch.long)

        return _Encoded()


class _VariableStubModel:
    """Emits, for the pass over text *i*, ``FIXTURE_WINDOWS[i][0]`` rows whose
    L2 norm is all ``FIXTURE_WINDOWS[i][1]`` — a flat window per text, so the
    only thing that can move the answer is HOW the windows are weighted."""

    num_layers = 1
    hidden_size = 2
    device = "cpu"

    def __init__(self):
        self.hooked = _StubHooked()
        self.tokenizer = _VariableTokenizer()

    def model(self, input_ids=None, use_cache=False):
        import torch

        length = int(input_ids.shape[1])
        norm = next(value for count, value in FIXTURE_WINDOWS if count == length)
        values = torch.tensor(
            [[[norm, 0.0] for _ in range(length)]], dtype=torch.float32)
        for intervention in self.hooked._interventions:
            intervention.apply(values, 0, 0)
        return None


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
