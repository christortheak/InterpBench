"""Opt-in LIVE GPU smoke test — real model load + extract + inject + generate.

Structural tests can't catch numeric/runtime regressions; this does. It is
gated on ``STEERLAB_LIVE=1`` (so the default suite stays fast/offline) and
auto-skips if Qwen3-0.6B isn't in the HF cache. Run with:

    STEERLAB_LIVE=1 pytest tests/test_live_smoke.py -s
"""

import os

import pytest

if os.environ.get("STEERLAB_LIVE") != "1":
    pytest.skip("live GPU smoke test (set STEERLAB_LIVE=1 to run)", allow_module_level=True)

from steerlab_server.steering import model_loader, vector_math as vm
from steerlab_server.steering.extractor import ExtractionOptions, activations, extract
from steerlab_server.steering.stimulus_set import StimulusSet
from steerlab_server.experiment.generate import CellInjection, generate

MODEL = "Qwen/Qwen3-0.6B"

if MODEL not in model_loader.local_model_ids():
    pytest.skip(f"{MODEL} not in HF cache", allow_module_level=True)


@pytest.fixture(scope="module")
def model():
    return model_loader.load(MODEL)


def _french_concept(tmp_path):
    d = tmp_path / "fr"
    d.mkdir()
    pos = ["Bonjour, je m'appelle Marie.", "Le chat est sur la table.",
           "J'aime beaucoup le café.", "Nous allons au marché demain.",
           "Il fait très beau aujourd'hui."]
    neg = ["Hello, my name is Mary.", "The cat is on the table.",
           "I really like coffee.", "We are going to the market tomorrow.",
           "It is very nice today."]
    (d / "positive.jsonl").write_text("\n".join('{"text": %s}' % __import__("json").dumps(t) for t in pos))
    (d / "negative.jsonl").write_text("\n".join('{"text": %s}' % __import__("json").dumps(t) for t in neg))
    return str(d)


def test_extract_inject_generate_changes_output(model, tmp_path):
    stimuli = StimulusSet.from_directory(_french_concept(tmp_path))
    result = extract(model, stimuli, ExtractionOptions())
    assert result.vectors.layer_count == model.num_layers
    mid = model.num_layers // 2
    vec = result.vectors.per_layer[mid]
    assert vm.l2_norm(vec) > 0

    prompt = "Tell me about your morning."
    base = generate(model, prompt, max_tokens=24, temperature=0.0,
                    prompt_mode="rawCompletion")
    steered = generate(model, prompt, max_tokens=24, temperature=0.0,
                       prompt_mode="rawCompletion",
                       injections=[CellInjection(layer=mid, vector=vec, alpha=14.0)])
    # Strong injection must change greedy output...
    assert steered != base
    # ...and alpha 0 must reproduce the baseline exactly.
    alpha0 = generate(model, prompt, max_tokens=24, temperature=0.0,
                      prompt_mode="rawCompletion",
                      injections=[CellInjection(layer=mid, vector=vec, alpha=0.0)])
    assert alpha0 == base


def test_activations_shape(model, tmp_path):
    stimuli = StimulusSet.from_directory(_french_concept(tmp_path))
    acts = activations(model, stimuli.positive[:2])
    assert len(acts.values) == 2
    assert len(acts.values[0]) == model.num_layers
    assert len(acts.values[0][0]) == model.hidden_size
