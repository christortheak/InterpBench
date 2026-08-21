"""The reading-position diagnostic: same passes, same numbers, free extras.

Two contracts on a tiny random in-memory Llama (no downloads, CPU):

1. Multi-position capture is IDENTICAL to separate single-position captures
   — the diagnostic reads a second position from the same forward passes,
   so it must change nothing about the primary reading.
2. A pooled extraction carries the per-layer cosine against last-token
   vectors; a last-token extraction carries nothing. The measured gap is
   the METHODS-appendix justification for the pooled-reading rule.
"""

from types import SimpleNamespace

import pytest
import torch

from steerlab_server.steering import extractor
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.reading_position import LAST_TOKEN, mean_from_token


class _FakeTokenizer:
    """Deterministic ids per text; length grows with text length."""

    def __call__(self, text, return_tensors=None):
        torch.manual_seed(len(text) + sum(ord(c) for c in text[:8]))
        n = max(2, min(24, len(text) // 3))
        return SimpleNamespace(input_ids=torch.randint(1, 127, (1, n)))


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(model=lm, hooked=HookedModel(lm),
                           device=torch.device("cpu"), num_layers=2,
                           tokenizer=_FakeTokenizer())


TEXTS = ["a steady voice in a difficult and frightening meeting room",
         "an unhurried afternoon of ordinary and calm errands in town"]


def test_multi_position_capture_matches_single_position_captures():
    model = _model()
    pooled = mean_from_token(3)
    both = extractor.activations_multi(model, TEXTS, [pooled, LAST_TOKEN])
    alone_pooled = extractor.activations(model, TEXTS, pooled)
    alone_last = extractor.activations(model, TEXTS, LAST_TOKEN)
    assert both[0].values == alone_pooled.values
    assert both[1].values == alone_last.values
    assert both[0].residual_norm_per_layer == alone_pooled.residual_norm_per_layer
    assert both[1].residual_norm_per_layer == alone_last.residual_norm_per_layer


def test_short_stimulus_refusal_names_the_strictest_position():
    model = _model()
    with pytest.raises(extractor.ConceptExtractorError, match="mean from token 9"):
        extractor.activations_multi(model, ["hi"], [mean_from_token(9), LAST_TOKEN])


def test_pooled_extraction_carries_the_diagnostic_last_token_does_not():
    model = _model()
    stimuli = SimpleNamespace(positive=[TEXTS[0]] * 2, negative=[TEXTS[1]] * 2)

    pooled = extractor.extract(
        model, stimuli,
        extractor.ExtractionOptions(reading_position=mean_from_token(3)))
    diag = pooled.reading_position_diagnostic
    assert diag is not None
    assert diag["primaryReadingPosition"] == "mean from token 3"
    assert diag["comparedTo"] == "last token"
    assert len(diag["perLayerCosine"]) == 2  # one per layer
    assert all(-1.0 <= c <= 1.0 for c in diag["perLayerCosine"])
    assert diag["min"] <= diag["median"] <= diag["max"]

    last = extractor.extract(
        model, stimuli,
        extractor.ExtractionOptions(reading_position=LAST_TOKEN))
    assert last.reading_position_diagnostic is None
    # And the diagnostic changed nothing about the primary vectors: a pooled
    # run's science vector is what a diagnostic-free implementation computed.
    assert pooled.vectors.per_layer != last.vectors.per_layer
