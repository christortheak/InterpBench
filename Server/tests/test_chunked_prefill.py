"""Chunked prefill: the MPS long-prompt path must be INVISIBLE to the science.

Three contracts, mirroring the Swift engine's chunked-prefill test suite
(where chunking has been native from day one, and where steering a chunk's
mid-prompt tail is the classic silent-near-null bug this guards against):

1. The device policy is a pure table — CUDA and CPU never chunk, so the
   science arm takes the exact single-pass path it always took.
2. Chunked and unchunked prefill produce the same model behavior: same
   next-token logits at the prompt end, same greedy continuation.
3. Injection gating: intermediate chunks are BIT-IDENTICAL to baseline
   (no steering mid-prompt), and the true prompt end still steers.

All on a tiny random in-memory Llama — no downloads, CPU, deterministic.
"""

import pytest
import torch
from types import SimpleNamespace

from steerlab_server.experiment import generate as generate_mod
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.injector import VectorInjector
from steerlab_server.steering.recorder import ActivationRecorder

HIDDEN = 32


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        max_position_embeddings=256)
    model = LlamaForCausalLM(config).eval()
    return SimpleNamespace(model=model, hooked=HookedModel(model),
                           device=torch.device("cpu"))


def _prompt(n=11):
    torch.manual_seed(11)
    return torch.randint(1, 127, (1, n))


# ------------------------------------------------------------ device policy


def test_chunking_is_a_per_device_table(monkeypatch):
    monkeypatch.delenv(generate_mod.PREFILL_CHUNK_ENV, raising=False)
    long, short = 30_000, 3_000
    assert generate_mod.prefill_chunk_size("mps", long) == 1024
    assert generate_mod.prefill_chunk_size("mps:0", long) == 1024
    # Below the threshold, MPS takes the exact path every measurement this
    # week characterized.
    assert generate_mod.prefill_chunk_size("mps", short) is None
    # CUDA chunks only past its own, higher threshold; below it, the exact
    # single-pass path. CPU never chunks.
    assert generate_mod.prefill_chunk_size("cuda", long) == 4096
    assert generate_mod.prefill_chunk_size("cuda:1", long) == 4096
    assert generate_mod.prefill_chunk_size("cuda", 8_192) is None
    assert generate_mod.prefill_chunk_size("cuda", short) is None
    assert generate_mod.prefill_chunk_size("cpu", long) is None

    monkeypatch.setenv(generate_mod.PREFILL_CHUNK_ENV, "512")
    assert generate_mod.prefill_chunk_size("mps", long) == 512
    monkeypatch.setenv(generate_mod.PREFILL_CHUNK_ENV, "0")
    assert generate_mod.prefill_chunk_size("mps", long) is None
    monkeypatch.setenv(generate_mod.PREFILL_CHUNK_ENV, "not-a-number")
    assert generate_mod.prefill_chunk_size("mps", long) == 1024


# ------------------------------------------------------------- equivalence


def test_chunked_prefill_matches_single_pass_logits_and_continuation():
    wrapper = _tiny_model()
    ids = _prompt()

    # Single-pass reference: full forward, logits at the last position.
    with torch.no_grad():
        reference = wrapper.model(input_ids=ids,
                                  attention_mask=torch.ones_like(ids))
    ref_logits = reference.logits[0, -1]

    # Chunked: cache over all but the last position, then the final token.
    past = generate_mod._chunked_prefill(wrapper, ids, chunk_size=3)
    with torch.no_grad():
        out = wrapper.model(input_ids=ids[:, -1:],
                            attention_mask=torch.ones_like(ids),
                            past_key_values=past)
    torch.testing.assert_close(out.logits[0, -1], ref_logits,
                               rtol=1e-4, atol=1e-5)

    # And the full generate() continuation agrees greedily.
    common = dict(attention_mask=torch.ones_like(ids), max_new_tokens=6,
                  do_sample=False, pad_token_id=0)
    with torch.no_grad():
        full = wrapper.model.generate(input_ids=ids, **common)
        chunked = wrapper.model.generate(
            input_ids=ids,
            past_key_values=generate_mod._chunked_prefill(wrapper, ids, 4),
            **common)
    assert full.tolist() == chunked.tolist()


def test_chunk_passes_advance_the_offsets_injection_gates_on():
    wrapper = _tiny_model()
    ids = _prompt(n=10)
    recorder = ActivationRecorder(layers=[0])
    with wrapper.hooked.session([recorder]):
        generate_mod._chunked_prefill(wrapper, ids, chunk_size=4)
    # 9 cached positions in chunks of 4: passes at offsets 0, 4, 8 — the
    # absolute positions VectorInjector.should_inject receives.
    assert [c.offset for c in recorder.captures] == [0, 4, 8]


# ---------------------------------------------------------------- steering


def test_intermediate_chunks_are_bit_identical_and_prompt_end_steers():
    wrapper = _tiny_model()
    ids = _prompt(n=10)
    n = ids.shape[1]
    injector = VectorInjector.single(
        layer=1, vector=[5.0] * HIDDEN, alpha=1.0, prompt_token_count=n)

    def cache_tensors(past):
        # transformers 5 DynamicCache: per-layer DynamicLayer with .keys/.values.
        return [t.clone() for layer in past.layers
                for t in (layer.keys, layer.values)]

    with wrapper.hooked.session([]):
        baseline = cache_tensors(
            generate_mod._chunked_prefill(wrapper, ids, chunk_size=3))
    with wrapper.hooked.session([injector]):
        past = generate_mod._chunked_prefill(wrapper, ids, chunk_size=3)
        steered_cache = cache_tensors(past)
        # The final prompt position runs INSIDE the session, as generate()'s
        # first forward does in production.
        with torch.no_grad():
            steered = wrapper.model(input_ids=ids[:, -1:],
                                    attention_mask=torch.ones_like(ids),
                                    past_key_values=past)

    # Every intermediate chunk's KV is BIT-identical to the uninjected run:
    # a chunk's last position is a mid-prompt token, and steering it is the
    # silent-near-null bug. torch.equal, not allclose — the gate either
    # fired or it did not.
    assert len(baseline) == len(steered_cache) and len(baseline) > 0
    for b, s in zip(baseline, steered_cache):
        assert torch.equal(b, s)

    with wrapper.hooked.session([]):
        past_clean = generate_mod._chunked_prefill(wrapper, ids, chunk_size=3)
        with torch.no_grad():
            clean = wrapper.model(input_ids=ids[:, -1:],
                                  attention_mask=torch.ones_like(ids),
                                  past_key_values=past_clean)

    # …and the true prompt end DOES steer: same cache, different logits.
    assert not torch.equal(steered.logits[0, -1], clean.logits[0, -1])


# ------------------------------------------------------------ hybrid model


def _tiny_hybrid_model():
    """A tiny random Qwen3.5-architecture causal LM: the layer pattern of
    a 27B hybrid (three gated-delta-rule linear-attention layers, then one
    full-attention layer with an output gate and partial rotary), small
    enough for CPU. Chunking must carry each linear layer's recurrent state
    and the last conv-kernel positions across chunk boundaries exactly."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        max_position_embeddings=1024)
    model = AutoModelForCausalLM.from_config(config).eval()
    return SimpleNamespace(model=model, hooked=HookedModel(model),
                           device=torch.device("cpu"))


@pytest.mark.parametrize("n,chunk", [(37, 8), (150, 40), (150, 64), (200, 97)])
def test_hybrid_chunked_prefill_matches_single_pass(n, chunk):
    wrapper = _tiny_hybrid_model()
    assert "linear_attention" in wrapper.model.config.layer_types
    assert "full_attention" in wrapper.model.config.layer_types
    ids = _prompt(n)

    with torch.no_grad():
        reference = wrapper.model(input_ids=ids,
                                  attention_mask=torch.ones_like(ids))
    ref_logits = reference.logits[0, -1]

    past = generate_mod._chunked_prefill(wrapper, ids, chunk_size=chunk)
    with torch.no_grad():
        out = wrapper.model(input_ids=ids[:, -1:],
                            attention_mask=torch.ones_like(ids),
                            past_key_values=past)
    torch.testing.assert_close(out.logits[0, -1], ref_logits,
                               rtol=1e-4, atol=1e-5)

    common = dict(attention_mask=torch.ones_like(ids), max_new_tokens=12,
                  do_sample=False, pad_token_id=0)
    with torch.no_grad():
        full = wrapper.model.generate(input_ids=ids, **common)
        chunked = wrapper.model.generate(
            input_ids=ids,
            past_key_values=generate_mod._chunked_prefill(wrapper, ids, chunk),
            **common)
    assert full.tolist() == chunked.tolist()
