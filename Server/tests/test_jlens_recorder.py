"""J-lens Stage 5: prediction alignment, hook ordering, and the budget.

The alignment and ordering tests are engine-pure — they drive the recorder with
synthetic hidden states, so the arithmetic that decides which activation
predicted which token is pinned without a model.
"""

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import generate as gen
from steerlab_server.jlens import recorder as rec_mod
from steerlab_server.jlens.readout import Budget, ReadoutConfig, preflight
from steerlab_server.jlens.schemas import JLensError

D = 4


class _FakeReadout:
    """Returns the position's own marker so a mis-mapped row is visible."""

    def watched_scores(self, h, layer, *, use_jacobian=True):
        base = h.sum().reshape(1)
        return base if use_jacobian else base * 10

    def topk(self, h, layer, k, *, use_jacobian=True):
        ids = torch.arange(k) + (0 if use_jacobian else 100)
        return ids, torch.arange(k, dtype=torch.float32)


def _recorder(prompt_len, **cfg):
    config = ReadoutConfig(**{"layers": [0], "watchlist": [7], **cfg})
    return rec_mod.JLensReadoutRecorder(_FakeReadout(), config, prompt_len)


def _hidden(seq, value=1.0):
    return torch.full((1, seq, D), value)


# --- prediction alignment ----------------------------------------------------

def test_final_prompt_position_predicts_generated_token_zero():
    r = _recorder(5)
    r.apply(_hidden(5), 0, 0)          # prefill: positions 0..4
    assert [o.position for o in r.observations] == [4]
    assert [o.predictedIndex for o in r.observations] == [0]
    assert r.observations[0].passKind == "prefill"


def test_prompt_positions_before_the_last_are_not_recorded():
    """They predict prompt tokens, which nobody sampled — recording them would
    inflate the trace with rows no consumer can interpret."""
    r = _recorder(5)
    r.apply(_hidden(5), 0, 0)
    assert all(o.predictedIndex >= 0 for o in r.observations)
    assert len(r.observations) == 1


def test_each_decode_pass_predicts_the_next_token():
    r = _recorder(5)
    r.apply(_hidden(5), 0, 0)                    # -> predicts 0
    for step in range(3):
        r.apply(_hidden(1), 0, 5 + step)         # -> predicts 1, 2, 3
    assert [o.predictedIndex for o in r.observations] == [0, 1, 2, 3]
    assert [o.passKind for o in r.observations] == ["prefill"] + ["decode"] * 3


def test_alignment_is_uniform_across_a_chunked_prefill():
    """The formula is position-based, so a prefill split into passes gives the
    same answer as one pass — no special case, nothing to get wrong later."""
    whole = _recorder(6)
    whole.apply(_hidden(6), 0, 0)

    chunked = _recorder(6)
    chunked.apply(_hidden(4), 0, 0)              # positions 0..3: none recorded
    chunked.apply(_hidden(2), 0, 4)              # positions 4..5: 5 is the last
    assert [(o.position, o.predictedIndex) for o in chunked.observations] == \
           [(o.position, o.predictedIndex) for o in whole.observations]


def test_ids_are_joined_from_the_returned_sequence_not_from_text():
    r = _recorder(5)
    r.apply(_hidden(5), 0, 0)
    for step in range(2):
        r.apply(_hidden(1), 0, 5 + step)
    r.join_token_ids([111, 222, 333])
    assert [o.predictedTokenID for o in r.observations] == [111, 222, 333]
    assert r.complete is True


def test_a_short_sequence_leaves_the_trace_incomplete_and_says_why():
    """A failed or truncated trace must never masquerade as a complete
    readout — reportable consumers gate on this flag."""
    r = _recorder(5)
    r.apply(_hidden(5), 0, 0)
    for step in range(3):
        r.apply(_hidden(1), 0, 5 + step)
    r.join_token_ids([111, 222])                 # 4 rows, 2 ids
    assert r.complete is False
    assert "past the end" in r.failureReason


def test_no_ids_at_all_is_incomplete_not_empty_success():
    r = _recorder(5)
    r.apply(_hidden(5), 0, 0)
    r.join_token_ids([])
    assert r.complete is False
    assert "no token ids" in r.failureReason


# --- read-only ---------------------------------------------------------------

def test_the_recorder_returns_the_hidden_state_unchanged_and_unmutated():
    r = _recorder(2)
    h = _hidden(2, 3.0)
    before = h.clone()
    out = r.apply(h, 0, 0)
    assert out is h                      # identity, not a copy
    assert torch.equal(h, before)        # no in-place mutation


def test_unarmed_layers_are_skipped_entirely():
    r = _recorder(2)
    h = _hidden(2)
    assert r.apply(h, 5, 0) is h
    assert r.observations == []


def test_a_readout_failure_marks_the_trace_without_breaking_generation():
    class _Boom:
        def watched_scores(self, *a, **kw):
            raise RuntimeError("kaboom")

    config = ReadoutConfig(layers=[0], watchlist=[7])
    r = rec_mod.JLensReadoutRecorder(_Boom(), config, 1)
    h = _hidden(1)
    assert r.apply(h, 0, 0) is h         # generation continues
    assert "kaboom" in r.failureReason
    r.join_token_ids([5])
    assert r.complete is False


# --- the logit-lens companion ------------------------------------------------

def test_the_companion_is_recorded_alongside_by_default():
    r = _recorder(1)
    r.apply(_hidden(1), 0, 0)
    obs = r.observations[0]
    assert obs.watched and obs.watchedLogitLens
    assert obs.watched != obs.watchedLogitLens

    off = _recorder(1, logitLensCompanion=False)
    off.apply(_hidden(1), 0, 0)
    assert off.observations[0].watchedLogitLens == []


# --- hook ordering -----------------------------------------------------------

def test_observers_are_armed_after_injectors():
    """Order is the contract: an observer at the injection layer must see the
    POST-intervention residual, so it runs last."""
    import inspect

    src = inspect.getsource(gen._stream_rendered)
    assert "injectors + list(observers or [])" in src


# --- generated-token-id extraction -------------------------------------------

def test_generated_ids_drop_the_prompt_and_keep_special_tokens():
    seq = torch.tensor([[1, 2, 3, 40, 41, 2]])       # 2 is also an EOS-like id
    assert gen.generated_token_ids(seq, 3) == [40, 41, 2]


def test_generated_ids_handle_the_return_dict_shape():
    class _Out:
        sequences = torch.tensor([[1, 2, 9, 9]])

    assert gen.generated_token_ids(_Out(), 2) == [9, 9]
    assert gen.generated_token_ids(None, 2) == []


# --- budget ------------------------------------------------------------------

def test_watchlist_only_runs_are_not_charged_for_projections_they_never_do():
    est = preflight(ReadoutConfig(layers=[0, 1], watchlist=[1, 2]),
                    generations=100, max_new_tokens=400)
    assert est["fullVocabProjections"] == 0
    assert est["withinBudget"] is True


def test_topk_drives_the_compute_ceiling_and_lowering_k_does_not_help():
    """k selects from the result; it does not avoid the matmul."""
    big = ReadoutConfig(layers=[0, 1, 2], watchlist=[], topK=50,
                        topKLayers=[0, 1, 2])
    small = ReadoutConfig(layers=[0, 1, 2], watchlist=[], topK=5,
                          topKLayers=[0, 1, 2])
    a = preflight(big, generations=500, max_new_tokens=400)
    b = preflight(small, generations=500, max_new_tokens=400)
    assert a["fullVocabProjections"] == b["fullVocabProjections"]
    assert not a["withinBudget"]
    assert any("does not reduce it" in p for p in a["problems"])


def test_the_companion_doubles_projections_only_when_topk_is_armed():
    with_c = preflight(ReadoutConfig(layers=[0], topK=10, topKLayers=[0]),
                       generations=10, max_new_tokens=10)
    without = preflight(ReadoutConfig(layers=[0], topK=10, topKLayers=[0],
                                      logitLensCompanion=False),
                        generations=10, max_new_tokens=10)
    assert with_c["fullVocabProjections"] == 2 * without["fullVocabProjections"]


def test_the_bound_is_a_whole_study_quantity():
    """Computing it per shard would multiply the effective ceiling by the shard
    count — the exact failure the preflight exists to prevent."""
    whole = preflight(ReadoutConfig(layers=[0], watchlist=[1]),
                      generations=400, max_new_tokens=100)
    shard = preflight(ReadoutConfig(layers=[0], watchlist=[1]),
                      generations=100, max_new_tokens=100)
    assert whole["observations"] == 4 * shard["observations"]


def test_an_over_budget_configuration_is_refused_with_a_reason():
    est = preflight(ReadoutConfig(layers=list(range(20)), watchlist=[1]),
                    generations=10, max_new_tokens=10,
                    budget=Budget(maxArmedLayers=4))
    assert not est["withinBudget"]
    assert any("armed layers exceeds" in p for p in est["problems"])


def test_an_empty_configuration_refuses_rather_than_recording_nothing():
    with pytest.raises(JLensError):
        ReadoutConfig(layers=[0]).validate()
    with pytest.raises(JLensError):
        ReadoutConfig(layers=[], watchlist=[1]).validate()
