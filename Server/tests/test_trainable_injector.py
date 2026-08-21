"""The training-time injector: sphere constraint, gradient isolation, position
gating, and — the load-bearing one — parity between the training forward path
and the deployed stepped-KV logprob instrument.

All CPU, all on a tiny in-memory Llama (the ``test_chunked_prefill`` fixture
convention). No downloads, no GPU.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import logprob
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.trainable_injector import (
    TrainableInjectorError, TrainableVectorInjector)

HIDDEN = 32


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        max_position_embeddings=256)
    return LlamaForCausalLM(config).eval()


# ------------------------------------------------------- constraint + grads


def test_constrained_vector_has_exactly_alpha_norm():
    injector = TrainableVectorInjector(
        layer=1, hidden_size=HIDDEN, alpha_absolute=2.5,
        u=torch.arange(1, HIDDEN + 1, dtype=torch.float32))
    assert math.isclose(float(injector.vector().detach().norm()), 2.5,
                        rel_tol=1e-6)
    exported = injector.constrained_vector()
    assert len(exported) == HIDDEN
    assert math.isclose(
        math.sqrt(sum(x * x for x in exported)), 2.5, rel_tol=1e-6)
    # The direction is u's direction; only the length is imposed.
    cosine = float(torch.nn.functional.cosine_similarity(
        injector.vector().detach(), injector.u.detach(), dim=0))
    assert math.isclose(cosine, 1.0, rel_tol=1e-6)


def test_alpha_scaling_does_not_change_direction():
    u = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(3))
    a = TrainableVectorInjector(layer=0, hidden_size=HIDDEN,
                                alpha_absolute=1.0, u=u).constrained_vector()
    b = TrainableVectorInjector(layer=0, hidden_size=HIDDEN,
                                alpha_absolute=7.0, u=u).constrained_vector()
    for x, y in zip(a, b):
        assert math.isclose(7.0 * x, y, rel_tol=1e-5, abs_tol=1e-6)


def test_gradient_flows_to_u_and_only_to_u():
    model = _tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hooked = HookedModel(model)
    injector = TrainableVectorInjector(layer=1, hidden_size=HIDDEN,
                                       alpha_absolute=1.0)
    ids = torch.randint(1, 127, (2, 6))
    mask = torch.ones_like(ids)
    injector.set_batch(answer_positions=torch.tensor([5, 5]),
                       attention_mask=mask)
    with hooked.session([injector]):
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    out.logits[:, -1].sum().backward()
    assert injector.u.grad is not None
    assert float(injector.u.grad.norm()) > 0
    assert all(p.grad is None for p in model.parameters())


def test_gradient_is_sphere_tangent():
    """The projection lives inside ``apply``, so the gradient the optimizer
    sees is already orthogonal to u — radial motion cannot change v."""
    injector = TrainableVectorInjector(layer=0, hidden_size=8,
                                       alpha_absolute=3.0)
    target = torch.randn(8, generator=torch.Generator().manual_seed(1))
    (injector.vector() @ target).backward()
    radial = float(torch.dot(injector.u.grad, injector.u.detach()))
    assert abs(radial) < 1e-4


def test_zero_and_mismatched_inits_refuse():
    with pytest.raises(TrainableInjectorError):
        TrainableVectorInjector(layer=0, hidden_size=4, alpha_absolute=1.0,
                                u=torch.zeros(4))
    with pytest.raises(TrainableInjectorError):
        TrainableVectorInjector(layer=0, hidden_size=4, alpha_absolute=1.0,
                                u=torch.ones(5))
    with pytest.raises(TrainableInjectorError):
        TrainableVectorInjector(layer=0, hidden_size=4, alpha_absolute=0.0)
    with pytest.raises(TrainableInjectorError):
        TrainableVectorInjector(layer=0, hidden_size=4, alpha_absolute=1.0,
                                position_mode="everywhere")


# ------------------------------------------------------------ position gating


def _batch_of_three():
    """Right-padded batch, lengths 6/4/2 — the padding convention the driver
    asserts."""
    lengths = [6, 4, 2]
    width = max(lengths)
    mask = torch.zeros((3, width), dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    positions = torch.tensor([n - 1 for n in lengths])
    return lengths, width, mask, positions


def test_from_response_injects_exactly_at_the_supplied_positions():
    lengths, width, mask, positions = _batch_of_three()
    injector = TrainableVectorInjector(
        layer=2, hidden_size=4, alpha_absolute=1.0,
        u=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    h = torch.zeros((3, width, 4))
    injector.set_batch(answer_positions=positions, attention_mask=mask)
    out = injector.apply(h, layer=2, offset=0).detach()
    for i, n in enumerate(lengths):
        for p in range(width):
            expected = 1.0 if p == n - 1 else 0.0
            assert float(out[i, p, 0]) == pytest.approx(expected)


def test_all_mode_injects_at_every_non_pad_position():
    lengths, width, mask, positions = _batch_of_three()
    injector = TrainableVectorInjector(
        layer=2, hidden_size=4, alpha_absolute=1.0, position_mode="all",
        u=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    h = torch.zeros((3, width, 4))
    injector.set_batch(answer_positions=positions, attention_mask=mask)
    out = injector.apply(h, layer=2, offset=0).detach()
    for i, n in enumerate(lengths):
        for p in range(width):
            expected = 1.0 if p < n else 0.0
            assert float(out[i, p, 0]) == pytest.approx(expected)


def test_other_layers_are_untouched():
    injector = TrainableVectorInjector(layer=2, hidden_size=4,
                                       alpha_absolute=1.0)
    h = torch.ones((2, 3, 4))
    injector.set_batch(answer_positions=torch.tensor([2, 1]))
    for layer in (0, 1, 3, 7):
        assert injector.apply(h, layer=layer, offset=0) is h


def test_unset_batch_refuses_rather_than_guessing():
    injector = TrainableVectorInjector(layer=0, hidden_size=4,
                                       alpha_absolute=1.0)
    with pytest.raises(TrainableInjectorError):
        injector.apply(torch.zeros((1, 2, 4)), layer=0, offset=0)
    injector.set_batch(answer_positions=torch.tensor([9]))
    with pytest.raises(TrainableInjectorError):
        injector.apply(torch.zeros((1, 2, 4)), layer=0, offset=0)


# -------------------------------------------------------------------- parity


class _FakeTokenizer:
    """Whitespace tokens hash to small ids (the test_logprob convention)."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(tok, len(self.vocab) + 1)
               for tok in text.split()]

        class R:
            input_ids = ids
        return R()


def test_exported_vector_under_the_deployed_injector_matches_training_path():
    """The equivalence argument, exercised end to end.

    The training path injects at the last prompt position of a single
    full-sequence pass; the deployed path injects at the true prompt end of a
    stepped KV-cache scoring pass. For a single-token option these must produce
    the same answer-position logprobs — otherwise every gradient is taken
    through a computation the instrument does not perform.
    """
    from steerlab_server.experiment.generate import CellInjection
    from steerlab_server.experiment.optvec_train import (PreparedItem,
                                                         answer_logit_rows,
                                                         choice_logits)

    lm = _tiny_model()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    tokenizer = _FakeTokenizer()
    model = SimpleNamespace(model=lm, tokenizer=tokenizer,
                            hooked=HookedModel(lm), model_id="test/tiny",
                            revision=None, device=torch.device("cpu"),
                            context_window=256, hidden_size=HIDDEN)

    prompt = "the holding of the court is"
    options = ["alpha", "beta"]
    injector = TrainableVectorInjector(
        layer=1, hidden_size=HIDDEN, alpha_absolute=0.5,
        generator=torch.Generator().manual_seed(5))
    vector = injector.constrained_vector()

    rendered_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    option_ids = tuple(tokenizer(o, add_special_tokens=False).input_ids[0]
                       for o in options)
    item = PreparedItem(id="i1", role="target", split="train", prompt=prompt,
                        prompt_text=prompt, input_ids=tuple(rendered_ids),
                        options=tuple(options), option_token_ids=option_ids,
                        target="alpha")

    rows = answer_logit_rows(model, [item], injector)
    training = torch.log_softmax(
        choice_logits(rows[0], item).detach().float(), dim=-1)

    deployed = logprob.score_options(
        model, prompt, options, prompt_mode=RAW_COMPLETION,
        injections=[CellInjection(layer=1, vector=vector, alpha=1.0)])
    by_name = {s.option: s.logprob for s in deployed.options}
    # Both are full-vocabulary log-softmax rows; compare the option GAP, which
    # is what every downstream quantity (margin, choice probability) uses.
    training_gap = float(training[0] - training[1])
    deployed_gap = by_name["alpha"] - by_name["beta"]
    assert training_gap == pytest.approx(deployed_gap, abs=1e-4)

    # And the unsteered pair agrees too, so the match is not an artifact of
    # both paths being equally wrong.
    plain_rows = answer_logit_rows(model, [item], None)
    plain = torch.log_softmax(choice_logits(plain_rows[0], item).float(), dim=-1)
    plain_deployed = logprob.score_options(model, prompt, options,
                                           prompt_mode=RAW_COMPLETION)
    plain_by_name = {s.option: s.logprob for s in plain_deployed.options}
    assert float(plain[0] - plain[1]) == pytest.approx(
        plain_by_name["alpha"] - plain_by_name["beta"], abs=1e-4)
    # Steering actually moved something (a null parity test proves nothing).
    assert abs(training_gap - float(plain[0] - plain[1])) > 1e-6
