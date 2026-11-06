"""The answer-token readout reads a long prompt the way generation does (issue #8).

``logprob.score_options`` read its whole prompt in one pass, once per option.
Now a caller may pass a prompt reading (``reading``): when it chunks a prompt
of that length, the cache is filled by ``generate._chunked_prefill`` and the
last prompt position is read alone, as a chunked generation reads it. And
when every option is one token, the prompt is read once for the item.

All on CPU, on tiny random models (a dense Llama and a Qwen3.5-pattern
hybrid, whose linear-attention layers carry a running state across chunks),
with an explicit small rule: chunks of 4 tokens for prompts over 8. A chunked
reading sums in a different order, so its numbers are compared with the
single pass within float32 rounding (rtol 1e-4, atol 1e-5, as
``test_chunked_prefill.py`` compares them); where nothing about the
arithmetic changes, they are compared exactly.
"""
import pytest
import torch

from steerlab_server.experiment import logprob
from steerlab_server.experiment import prompt_reading as PR
from steerlab_server.experiment import prompt_render
from steerlab_server.experiment.generate import CellInjection
from steerlab_server.steering.intervention import LayerIntervention

WORDS = [f"w{i}" for i in range(40)]
MARKERS = ["turn", "user", "assistant", "system", "end"]
HIDDEN = 32
#: Chunks of 4 tokens for a prompt over 8: the tests' prompts are 30 tokens.
RULE = PR.PromptReading.chunked(4, 8, source="studyDeclaration")
PROMPT_WORDS = 30
PROMPT = " ".join(WORDS[(7 * i) % len(WORDS)] for i in range(PROMPT_WORDS))
SINGLE_TOKEN = ["w1", "w2"]
MULTI_TOKEN = ["w3 w4", "w5 w6"]
TRANSCRIPT = [
    {"role": "user", "content": " ".join(WORDS[:12])},
    {"role": "assistant", "content": " ".join(WORDS[12:24])},
    {"role": "user", "content": "w30 w31 w32"},
]
#: A chat template written with words the tokenizer knows, so a scripted
#: transcript renders to known token ids.
TEMPLATE = ("{% for m in messages %}turn {{ m['role'] }} {{ m['content'] }} end "
            "{% endfor %}{% if add_generation_prompt %}turn assistant {% endif %}")
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    vocab = {token: i for i, token in enumerate(["[PAD]", "[EOS]", "[UNK]",
                                                 *MARKERS, *WORDS])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = TEMPLATE
    return tokenizer, len(vocab)


def _llama(vocab):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=vocab,
        max_position_embeddings=256, pad_token_id=0, eos_token_id=None)).eval()


def _hybrid(vocab):
    """The layer pattern of a 27B hybrid: three gated-delta-rule
    linear-attention layers, then one full-attention layer (as
    ``test_chunked_prefill._tiny_hybrid_model``)."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.manual_seed(7)
    return AutoModelForCausalLM.from_config(Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        max_position_embeddings=256)).eval()


def _model(kind="llama"):
    from steerlab_server.steering.hooks import HookedModel
    from steerlab_server.steering.model_loader import SteeredModel
    tokenizer, vocab = _tokenizer()
    lm = _llama(vocab) if kind == "llama" else _hybrid(vocab)
    return SteeredModel(model=lm, tokenizer=tokenizer, hooked=HookedModel(lm),
                        model_id="example/model", revision="a" * 40)


def _passes(model):
    """The length of every forward pass the model makes, in order."""
    seen = []

    def hook(_module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        seen.append(int(ids.shape[1]))
    model.model.register_forward_pre_hook(hook, with_kwargs=True)
    return seen


def _steering():
    torch.manual_seed(5)
    return [CellInjection(layer=1, vector=torch.randn(HIDDEN).tolist(), alpha=2.0)]


CASES = {
    "no steering": dict(prompt_mode=prompt_render.RAW_COMPLETION),
    "steering": dict(prompt_mode=prompt_render.RAW_COMPLETION, steer=True),
    "scripted transcript": dict(prompt_mode=prompt_render.CHAT_ASSISTANT,
                                transcript=TRANSCRIPT, steer=True),
}


def _score(model, options, case, reading=None):
    kwargs = dict(CASES[case])
    steer = kwargs.pop("steer", False)
    return logprob.score_options(model, PROMPT, options,
                                 injections=_steering() if steer else None,
                                 reading=reading, **kwargs)


def _logprobs(result):
    return torch.tensor([lp for s in result.options for lp in s.token_logprobs],
                        dtype=torch.float64)


# --- chunked equals single pass ------------------------------------------------------

@pytest.mark.parametrize("kind", ["llama", "hybrid"])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("options", [SINGLE_TOKEN, MULTI_TOKEN],
                         ids=["single-token", "multi-token"])
def test_a_chunked_readout_equals_the_single_pass(kind, case, options):
    model = _model(kind)
    passes = _passes(model)
    single = _score(model, options, case)
    n = single.prompt_token_count
    assert n > 8
    reads = 1 if options is SINGLE_TOKEN else len(options)
    assert passes == ([n] if reads == 1 else [n, 1] * reads)
    passes.clear()
    chunked = _score(model, options, case, reading=RULE)
    assert chunked.prompt_token_count == n
    # Really chunked: chunks of 4 over every position but the last, then the
    # last alone, as a chunked generation reads it.
    read = [4] * ((n - 1) // 4) + ([(n - 1) % 4] if (n - 1) % 4 else []) + [1]
    assert passes == (read if reads == 1 else (read + [1]) * reads)
    torch.testing.assert_close(_logprobs(chunked), _logprobs(single), **TOLERANCE)
    assert chunked.selected == single.selected
    assert [s.token_ids for s in chunked.options] == [s.token_ids for s in single.options]


def test_the_hooks_are_told_where_the_prompt_ends_and_then_forget_it():
    """The memory diagnostic counts the last prompt position, read alone, as
    prompt rather than as a decode step; the readout says where the prompt
    ends while it reads, and leaves no trace afterwards."""
    model = _model()
    seen = []
    real = model.model.forward

    def forward(*args, **kwargs):
        seen.append(model.hooked.prompt_token_count)
        return real(*args, **kwargs)
    model.model.forward = forward
    result = _score(model, SINGLE_TOKEN, "no steering", reading=RULE)
    assert set(seen) == {result.prompt_token_count}
    assert model.hooked.prompt_token_count is None


# --- the steering gate ---------------------------------------------------------------

class _Spy(LayerIntervention):
    """Wraps one chain member; records the last absolute position of every
    pass on which it changed the residual stream."""

    def __init__(self, inner, hits):
        self.inner, self.hits = inner, hits

    def apply(self, h, layer, offset):
        out = self.inner.apply(h, layer, offset)
        if out is not h:
            self.hits.append((layer, offset + h.shape[1] - 1, h.shape[1]))
        return out

    def scope(self):
        return self.inner.scope()


def _hits(monkeypatch, model, options, case, reading=None):
    hits = []
    real = logprob._injectors
    monkeypatch.setattr(logprob, "_injectors", lambda *a, **k: [
        _Spy(member, hits) for member in real(*a, **k)])
    result = _score(model, options, case, reading=reading)
    monkeypatch.setattr(logprob, "_injectors", real)
    return hits, result.prompt_token_count


@pytest.mark.parametrize("case", ["steering", "scripted transcript"])
@pytest.mark.parametrize("options", [SINGLE_TOKEN, MULTI_TOKEN],
                         ids=["single-token", "multi-token"])
def test_the_injector_fires_at_the_prompt_end_and_each_option_step_only(
        monkeypatch, case, options):
    model = _model()
    single, n = _hits(monkeypatch, model, options, case)
    chunked, _ = _hits(monkeypatch, model, options, case, reading=RULE)
    # Hand-written: the prompt's last position (n - 1), then, for a two-token
    # option, its one cached step at position n; once per read.
    expected = ([(1, n - 1)] if options is SINGLE_TOKEN
                else [(1, n - 1), (1, n)] * len(options))
    assert [(layer, position) for layer, position, _ in single] == expected
    assert [(layer, position) for layer, position, _ in chunked] == expected
    # No chunk fired: every chunked hit is a one-token pass (the last prompt
    # position, read alone, or an option step), never a 4-token chunk.
    assert all(length == 1 for _, _, length in chunked)


# --- one read per item when every option is one token ---------------------------------

def _per_option_reference(model, options, case, reading=None):
    """The readout as it was before issue #8: one read of the prompt per
    option, each in a fresh pass, through the same chain."""
    kwargs = dict(CASES[case])
    steer = kwargs.pop("steer", False)
    transcript = kwargs.pop("transcript", None)
    rendered = (prompt_render.render_transcript(model.tokenizer, transcript,
                                                model_id=model.model_id, **kwargs)
                if transcript else
                prompt_render.render(model.tokenizer, PROMPT,
                                     model_id=model.model_id, **kwargs))
    ids = [list(model.tokenizer(o, add_special_tokens=False).input_ids)
           for o in options]
    chain = logprob._injectors(_steering() if steer else [],
                               rendered.prompt_token_count)
    chunk = reading.chunk_size(rendered.prompt_token_count) if reading else None
    with model.hooked.session(chain):
        rows = [logprob._continuation_logits(model, rendered.input_ids, option,
                                             chunk_size=chunk)
                for option in ids]
    return logprob.option_scores_from_step_logits(options, ids, rows)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("reading", [None, RULE], ids=["single pass", "chunked"])
def test_single_token_options_read_the_prompt_once_with_unchanged_scores(
        case, reading):
    model = _model()
    passes = _passes(model)
    result = _score(model, SINGLE_TOKEN + ["w7", "w8"], case, reading=reading)
    # One read of the prompt for four options, and nothing else.
    n = result.prompt_token_count
    assert passes == ([n] if reading is None else
                      [4] * ((n - 1) // 4) + [(n - 1) % 4] * bool((n - 1) % 4) + [1])
    reference = _per_option_reference(model, SINGLE_TOKEN + ["w7", "w8"], case,
                                      reading=reading)
    # Exactly the old per-option scores: the same arithmetic, read once.
    assert [s.token_logprobs for s in result.options] == \
        [s.token_logprobs for s in reference]


def test_multi_token_options_keep_one_read_each_and_score_correctly():
    model = _model()
    passes = _passes(model)
    options = MULTI_TOKEN + ["w9"]
    result = _score(model, options, "no steering")
    n = result.prompt_token_count
    # One read per option, plus one cached step for each two-token option.
    assert passes == [n, 1, n, 1, n]
    # Independently of the stepped driver: one unsteered pass over the
    # prompt and the option, read at the positions before each option token.
    prompt_ids = prompt_render.render(
        model.tokenizer, PROMPT, model_id=model.model_id,
        prompt_mode=prompt_render.RAW_COMPLETION).input_ids
    assert len(prompt_ids) == n
    for score in result.options:
        full = torch.tensor([prompt_ids + score.token_ids])
        with torch.no_grad():
            logits = model.model(input_ids=full).logits[0]
        rows = torch.log_softmax(logits.float(), dim=-1)
        expected = [rows[n - 1 + k, token].item()
                    for k, token in enumerate(score.token_ids)]
        assert score.token_logprobs == pytest.approx(expected, rel=1e-4, abs=1e-5)


# --- callers that pass no reading ------------------------------------------------------

def test_a_caller_that_passes_no_reading_reads_a_long_prompt_in_one_pass():
    """Even inside a run's reading scope, where generation would chunk this
    prompt: optvec, J-lens g0, a sweep's objective, and every other caller
    that passes no reading keep the single pass."""
    model = _model()
    passes = _passes(model)
    with PR.reading_scope(RULE):
        result = _score(model, MULTI_TOKEN, "steering")
    n = result.prompt_token_count
    assert n > RULE.threshold
    assert passes == [n, 1, n, 1]


def test_a_reading_that_does_not_chunk_this_prompt_is_bit_identical():
    """At or below the threshold the path is today's single pass, exactly."""
    model = _model()
    passes = _passes(model)
    plain = _score(model, MULTI_TOKEN, "steering")
    n = plain.prompt_token_count
    passes.clear()
    above = PR.PromptReading.chunked(4, n, source="studyDeclaration")
    same = _score(model, MULTI_TOKEN, "steering", reading=above)
    assert passes == [n, 1, n, 1]
    assert [s.token_logprobs for s in same.options] == \
        [s.token_logprobs for s in plain.options]
    passes.clear()
    once = _score(model, MULTI_TOKEN, "steering",
                  reading=PR.PromptReading.single_pass(source="studyDeclaration"))
    assert passes == [n, 1, n, 1]
    assert [s.token_logprobs for s in once.options] == \
        [s.token_logprobs for s in plain.options]
