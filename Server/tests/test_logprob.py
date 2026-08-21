"""Answer-token logprob instrument: pure scoring math + the stepped driver
against a fake causal LM (no GPU, no real model)."""

import math
from contextlib import contextmanager

import torch

from steerlab_server.experiment import logprob
from steerlab_server.experiment.prompt_render import RAW_COMPLETION


def test_option_scores_from_step_logits_multi_token_joint():
    vocab = 8
    row_a = torch.zeros(vocab)
    row_a[3] = 2.0
    row_b = torch.zeros(vocab)
    row_b[5] = 1.0
    scores = logprob.option_scores_from_step_logits(
        ["yes", "no way"],
        [[3], [4, 5]],
        [[row_a], [row_a, row_b]])
    lsm_a = torch.log_softmax(row_a, dim=-1)
    lsm_b = torch.log_softmax(row_b, dim=-1)
    assert math.isclose(scores[0].logprob, lsm_a[3].item(), rel_tol=1e-6)
    expected_joint = lsm_a[4].item() + lsm_b[5].item()
    assert math.isclose(scores[1].logprob, expected_joint, rel_tol=1e-6)
    assert math.isclose(scores[1].mean_token_logprob, expected_joint / 2, rel_tol=1e-6)


def test_choice_result_probability_margin_log_odds():
    a = logprob.OptionScore(option="A", token_ids=[1], token_logprobs=[-1.0])
    b = logprob.OptionScore(option="B", token_ids=[2], token_logprobs=[-2.0])
    result = logprob.ChoiceResult(options=[a, b])
    probabilities = result.probability
    assert math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-9)
    assert probabilities["A"] > probabilities["B"]
    assert result.selected == "A"
    assert math.isclose(result.margin, 1.0, rel_tol=1e-9)
    odds_a = result.log_odds["A"]
    expected = math.log(probabilities["A"] / (1 - probabilities["A"]))
    assert math.isclose(odds_a, expected, rel_tol=1e-9)
    fields = result.as_record_fields()
    assert fields["instrument"] == "answerTokenLogprob"
    assert fields["selected"] == "A"
    assert fields["optionTokenIDs"]["B"] == [2]


class _FakeOutput:
    def __init__(self, logits, past):
        self.logits = logits
        self.past_key_values = past


class _FakeCausalLM:
    """Constant next-token logits; counts forward passes and cache use."""

    def __init__(self, vocab=16):
        self.vocab = vocab
        self.calls = []

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None,
                 use_cache=True):
        seq = input_ids.shape[1]
        self.calls.append({
            "seq": seq, "cached": past_key_values is not None,
            "mask_len": None if attention_mask is None else attention_mask.shape[1],
            "mask_all_ones": (attention_mask is not None
                              and bool((attention_mask == 1).all())),
        })
        logits = torch.arange(self.vocab, dtype=torch.float32).repeat(1, seq, 1)
        return _FakeOutput(logits, past="cache")


class _FakeTokenizer:
    """Whitespace tokens hash to small ids; deterministic and offline."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(tok, len(self.vocab) + 1)
               for tok in text.split()]

        class R:
            input_ids = ids
        return R()


class _FakeHooked:
    def __init__(self):
        self.sessions = []
        self.resets = 0

    @contextmanager
    def session(self, interventions):
        self.sessions.append(list(interventions))
        yield

    def reset_offsets(self):
        self.resets += 1


class _FakeSteeredModel:
    model_id = "test-model"
    revision = "deadbeef"
    context_window = 0
    device = "cpu"

    def __init__(self):
        self.model = _FakeCausalLM()
        self.tokenizer = _FakeTokenizer()
        self.hooked = _FakeHooked()


def test_score_options_stepped_driver():
    model = _FakeSteeredModel()
    result = logprob.score_options(
        model, "the judgment is", ["affirmed", "reversed and remanded"],
        prompt_mode=RAW_COMPLETION)
    assert {s.option for s in result.options} == {"affirmed", "reversed and remanded"}
    by_name = {s.option: s for s in result.options}
    assert len(by_name["affirmed"].token_ids) == 1
    assert len(by_name["reversed and remanded"].token_ids) == 3
    # Constant logits: joint logprob of the 3-token option is 3 single-token
    # logprobs; ids differ but every row is the same ramp.
    assert math.isclose(sum(result.probability.values()), 1.0, rel_tol=1e-9)
    # Driver shape: one prefill per option (uncached) + len(option)-1 cached steps.
    uncached = [c for c in model.model.calls if not c["cached"]]
    cached = [c for c in model.model.calls if c["cached"]]
    assert len(uncached) == 2
    assert len(cached) == (1 - 1) + (3 - 1)
    assert all(c["seq"] == 1 for c in cached)
    # Every forward pass carries an explicit all-ones attention mask (batch of
    # one, no padding — same behavior, no HF pad==eos warning). Cached steps
    # mask the full sequence: prompt (3) + steps so far.
    assert all(c["mask_all_ones"] for c in model.model.calls)
    assert all(c["mask_len"] == 3 for c in uncached)
    assert [c["mask_len"] for c in cached] == [4, 5]
    # Offsets reset per option so injection gating sees a fresh pass each time.
    assert model.hooked.resets == 2
    assert len(model.hooked.sessions) == 1


def test_score_options_arms_injectors_with_prompt_token_count():
    model = _FakeSteeredModel()
    from steerlab_server.experiment.generate import CellInjection
    cell = CellInjection(layer=2, vector=[1.0, 0.0], alpha=3.0)
    logprob.score_options(model, "one two three", ["a", "b"],
                          injections=[cell], prompt_mode=RAW_COMPLETION)
    (injectors,) = model.hooked.sessions
    assert len(injectors) == 1
    assert injectors[0]._prompt_token_count == 3


def test_score_options_ablate_cell_builds_head_of_chain_ablator():
    """An ablate-mode cell must become a SubspaceAblator at the head of the
    chain — NOT a VectorInjector adding λ·v (the opposite intervention). This
    pins score_options to the shared builder (steering.plan.interventions),
    mirroring test_intervention_plan's coverage of the generation path."""
    from steerlab_server.experiment.generate import CellInjection
    from steerlab_server.steering import plan
    from steerlab_server.steering.ablator import SubspaceAblator
    from steerlab_server.steering.injector import VectorInjector

    model = _FakeSteeredModel()
    cells = [CellInjection(layer=1, vector=[0.0, 1.0], alpha=1.0,
                           mode="ablate", concept="fear"),
             CellInjection(layer=2, vector=[1.0, 0.0], alpha=3.0)]
    logprob.score_options(model, "one two three", ["a", "b"],
                          injections=cells, prompt_mode=RAW_COMPLETION)
    (chain,) = model.hooked.sessions
    assert sum(isinstance(i, SubspaceAblator) for i in chain) == 1
    assert isinstance(chain[0], SubspaceAblator)
    assert plan.satisfies_ordering_invariant(chain)
    # The add cell still rides behind the ablator, gated on the true prompt end.
    assert len(chain) == 2
    assert isinstance(chain[1], VectorInjector)
    assert chain[1]._prompt_token_count == 3


def test_score_options_rejects_degenerate_option_sets():
    model = _FakeSteeredModel()
    try:
        logprob.score_options(model, "prompt", ["only"], prompt_mode=RAW_COMPLETION)
        raise AssertionError("expected ValueError for a single option")
    except ValueError:
        pass


def test_option_length_guardrail_fields():
    a = logprob.OptionScore(option="A", token_ids=[1], token_logprobs=[-1.0])
    b = logprob.OptionScore(option="Nebraska law governs the damages",
                            token_ids=[2, 3, 4, 5], token_logprobs=[-0.5] * 4)
    fields = logprob.ChoiceResult(options=[a, b]).as_record_fields()
    assert fields["optionTokenCounts"] == {"A": 1,
                                           "Nebraska law governs the damages": 4}
    assert fields["optionLengthRatio"] == 4.0


def test_prompt_prefill_requests_sliced_logits():
    """The SD calibration cells are ~97K-token prompts through exactly this
    prefill: without logits_to_keep it materializes logits for EVERY prompt
    position to read one row — ~47 GiB at Gemma's 262K vocabulary, bigger
    than the model. The kwarg is passed when the forward supports it, in
    either spelling, and NEVER when it does not (test fakes, exotic models
    — the call must then be byte-identical to the pre-helper behavior)."""
    from steerlab_server.experiment.generate import logits_slice_kwargs

    class Modern:
        def forward(self, input_ids=None, attention_mask=None,
                    past_key_values=None, use_cache=True, logits_to_keep=0):
            pass

    class Legacy:
        def forward(self, input_ids=None, num_logits_to_keep=0):
            pass

    assert logits_slice_kwargs(Modern()) == {"logits_to_keep": 1}
    assert logits_slice_kwargs(Legacy()) == {"num_logits_to_keep": 1}
    assert logits_slice_kwargs(_FakeCausalLM()) == {}

    # And end to end: a fake that ACCEPTS the kwarg sees it on the prefill
    # (the long pass), while the per-option steps remain unconstrained.
    seen = []

    class SlicingLM(_FakeCausalLM):
        def forward(self, input_ids=None, attention_mask=None,
                    past_key_values=None, use_cache=True, logits_to_keep=0):
            seen.append((input_ids.shape[1], logits_to_keep))
            return self(input_ids=input_ids, attention_mask=attention_mask,
                        past_key_values=past_key_values, use_cache=use_cache)

    model = _FakeSteeredModel()
    lm = SlicingLM()
    model.model = lambda **kw: lm.forward(**kw)
    model.model.forward = lm.forward
    logprob.score_options(model, "the case turns on intent", ["Yes", "No"],
                          prompt_mode=RAW_COMPLETION)
    prefill = seen[0]
    assert prefill[0] > 1 and prefill[1] == 1, seen
