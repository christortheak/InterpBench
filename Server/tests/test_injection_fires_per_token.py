"""Injection must fire at the prompt end and on every decode step, and never on
a mid-prompt prefill chunk. Mirrors the Swift ``VectorInjector`` unit tests and
the smoke test's per-token fire assertion — the classic prefill-only bug.
"""

import torch

from steerlab_server.steering.injector import VectorInjector


def test_should_inject_single_chunk_prefill_then_decode():
    # Unknown prompt length: always inject (correct only for single-chunk prefill).
    assert VectorInjector.should_inject(offset=0, seq_len=10, prompt_token_count=None)

    pt = 10
    # Single-chunk prefill: last position is the prompt end → inject.
    assert VectorInjector.should_inject(offset=0, seq_len=10, prompt_token_count=pt)
    # Decode steps: each is one position past the prompt → inject.
    assert VectorInjector.should_inject(offset=10, seq_len=1, prompt_token_count=pt)
    assert VectorInjector.should_inject(offset=11, seq_len=1, prompt_token_count=pt)


def test_should_not_inject_on_mid_prompt_prefill_chunk():
    pt = 100
    # First prefill chunk ends at position 49 (< prompt end) → suppress.
    assert not VectorInjector.should_inject(offset=0, seq_len=50, prompt_token_count=pt)
    # Final prefill chunk reaches the prompt end → inject.
    assert VectorInjector.should_inject(offset=50, seq_len=50, prompt_token_count=pt)


def test_apply_adds_alpha_v_at_last_position_only():
    hidden = 4
    h = torch.zeros((1, 3, hidden), dtype=torch.float32)
    injector = VectorInjector.single(layer=2, vector=[1.0, 1.0, 1.0, 1.0], alpha=2.0)
    out = injector.apply(h, layer=2, offset=0)
    # Only the last position is modified, by alpha * v = 2.
    assert torch.allclose(out[0, 0], torch.zeros(hidden))
    assert torch.allclose(out[0, 1], torch.zeros(hidden))
    assert torch.allclose(out[0, 2], torch.full((hidden,), 2.0))


def test_apply_is_noop_for_other_layers():
    h = torch.ones((1, 2, 3), dtype=torch.float32)
    injector = VectorInjector.single(layer=5, vector=[1.0, 1.0, 1.0], alpha=1.0)
    out = injector.apply(h, layer=4, offset=0)
    assert out is h  # untouched layer returns the same tensor


def test_decode_step_single_position_is_injected():
    h = torch.zeros((1, 1, 3), dtype=torch.float32)
    injector = VectorInjector.single(layer=0, vector=[1.0, 2.0, 3.0], alpha=1.0,
                                     prompt_token_count=5)
    # offset past prompt → decode step → inject at the only position.
    out = injector.apply(h, layer=0, offset=5)
    assert out[0, 0].tolist() == [1.0, 2.0, 3.0]


def test_injection_fires_on_every_decode_step_of_a_sampled_generation():
    """Study-owned sampling rider (2026-07-21): the injector's fire rule is
    selection-agnostic — it gates on POSITION (prompt end + every decode
    step), never on how the next token is chosen, so a sampled saved-agent
    draw steers on prefill and on every one of its decode steps exactly
    like a greedy one."""
    pt = 12
    injector = VectorInjector.single(layer=0, vector=[1.0, 0.0], alpha=1.0,
                                     prompt_token_count=pt)
    # Prefill reaches the prompt end → fires.
    assert VectorInjector.should_inject(offset=0, seq_len=pt,
                                        prompt_token_count=pt)
    # Every decode step of a long sampled continuation fires, and the
    # addition lands on the (single) new position each time.
    for step in range(30):
        offset = pt + step
        assert VectorInjector.should_inject(offset=offset, seq_len=1,
                                            prompt_token_count=pt)
        h = torch.zeros((1, 1, 2), dtype=torch.float32)
        out = injector.apply(h, layer=0, offset=offset)
        assert out[0, 0].tolist() == [1.0, 0.0]


def test_composition_is_additive_across_injectors():
    h = torch.zeros((1, 1, 2), dtype=torch.float32)
    a = VectorInjector.single(layer=0, vector=[1.0, 0.0], alpha=1.0)
    b = VectorInjector.single(layer=0, vector=[0.0, 1.0], alpha=3.0)
    out = b.apply(a.apply(h, 0, 0), 0, 0)
    assert out[0, 0].tolist() == [1.0, 3.0]
