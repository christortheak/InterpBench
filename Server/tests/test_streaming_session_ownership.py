"""Arming ownership: a suspended generator must never steer somebody else.

``_stream_rendered`` holds the hook session ACROSS its ``yield`` — it has to,
because generation runs on a background thread that keeps firing the residual
hooks while the consumer is suspended between chunks. The hazard that creates
(open-issues §15, hunt 1's by-catch) is what these tests pin:

* A consumer that breaks early leaves the generator suspended and still
  referenced, so the injectors stay armed on the shared ``HookedModel``.
* Under the old stack-restore ``session()``, the NEXT caller to open a session
  captured those stale injectors as its ``previous`` and RESTORED them on exit
  — permanently arming the model with a dead record's steering. Measured
  sequence: armed 0 → mid-generator 1 → inner session exits **1** → generator
  closed 0. Silent steering of everything that followed.

The fix is two rules in :class:`HookedModel`, both exercised below:

1. An arming is disarmed by its own handle, and a handle that is no longer the
   current arming disarms NOTHING (a dead scope can never restore itself).
2. A streaming arming is ``abandonable``: opening a session over one does not
   adopt it as the state to restore, so the next caller starts from bare metal
   and leaves the model disarmed.

Everything runs on a tiny in-memory fake — no model, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from steerlab_server.experiment.generate import CellInjection, stream_generate
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.injector import VectorInjector
from steerlab_server.steering.model_loader import SteeredModel

HIDDEN = 4


# --- fakes -------------------------------------------------------------------

class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=[2, 3, 4])

    def decode(self, ids, **kwargs):
        # Grows with the cache so TextIteratorStreamer emits one word per put
        # (a decode with constant output yields exactly one non-empty chunk).
        return "".join(f"tok{i} " for i in ids)


class _Block(nn.Module):
    def forward(self, x):
        return x


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])


class _StreamingLM(nn.Module):
    """Decoder blocks a real :class:`HookedModel` can hook, plus a ``generate``
    that fires one forward pass per step and streams a word per step.

    ``fail_at`` raises inside the generation thread (the error-propagation
    path); ``block`` is a barrier the thread waits on before its LAST step, so
    a test can hold a generation genuinely mid-flight.
    """

    def __init__(self, *, steps=6, fail_at=None):
        super().__init__()
        self.model = _Inner(2)
        self.config = SimpleNamespace(max_position_embeddings=4096,
                                      hidden_size=HIDDEN)
        self.weight = nn.Parameter(torch.zeros(1))
        self.steps = steps
        self.fail_at = fail_at
        self.passes = 0

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return h

    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        processor = kwargs["logits_processor"]
        streamer.put(kwargs["input_ids"])          # consumed by skip_prompt
        for step in range(self.steps):
            self.passes += 1
            self.forward(torch.zeros((1, 1, HIDDEN)))   # fires the hooks
            processor(kwargs["input_ids"], torch.zeros((1, 8)))
            if step == self.fail_at:
                raise RuntimeError("synthetic generation failure")
            streamer.put(torch.tensor([10 + step]))
        streamer.end()
        return torch.tensor([[2, 3, 4, 10]])


class _Counter:
    """Read-only observer: counts the hook fires it sees while armed."""

    def __init__(self):
        self.fires = 0

    def apply(self, h, layer: int, offset: int):
        self.fires += 1
        return h


def _model(**lm_kwargs):
    lm = _StreamingLM(**lm_kwargs)
    hooked = HookedModel(lm)
    model = SteeredModel(model=lm, tokenizer=_Tokenizer(), hooked=hooked,
                         model_id="org/fake", revision=None)
    return model, hooked


def _cell():
    return CellInjection(layer=0, vector=[1.0] * HIDDEN, alpha=2.0)


def _stream(model, **kwargs):
    return stream_generate(model, "hi", prompt_mode="rawCompletion",
                           injections=[_cell()], **kwargs)


# --- the pinned reproduction -------------------------------------------------

def test_abandoned_generator_never_steers_the_next_caller():
    """Hunt 1's reproduction, pinned.

    Old behaviour at the marked line: **1** — the inner session restored the
    abandoned stream's injector and every later generation on this model was
    silently steered by a dead record's vector.
    """
    model, hooked = _model(steps=6)
    stream = _stream(model)
    assert len(hooked.interventions) == 0          # armed 0

    first = next(stream)                            # suspended mid-generation
    assert first
    assert len(hooked.interventions) == 1          # armed 1 (a live generation)

    mine = VectorInjector.single(1, [1.0] * HIDDEN, 1.0)
    with hooked.session([mine]):
        # The nested caller sees ONLY what it asked for — never the stream's.
        assert hooked.interventions == [mine]
    assert hooked.interventions == []              # ← was 1 (stale restored)

    stream.close()                                  # the generator is collected
    assert hooked.interventions == []              # its exit re-arms nothing


def test_abandoned_generator_closing_last_still_disarms():
    """The other order: nobody supersedes the stream, so ITS OWN exit is what
    disarms the model — the ordinary abandonment path must still clean up."""
    model, hooked = _model(steps=6)
    stream = _stream(model)
    next(stream)
    assert len(hooked.interventions) == 1
    stream.close()
    assert hooked.interventions == []


def test_dropped_generator_is_disarmed_when_collected():
    """No explicit close: dropping the only reference runs the generator's
    ``finally`` (join + disarm) at collection time."""
    import gc

    model, hooked = _model(steps=6)
    stream = _stream(model)
    next(stream)
    assert len(hooked.interventions) == 1
    del stream
    gc.collect()
    assert hooked.interventions == []


# --- the correct-use paths must be untouched ---------------------------------

def test_fully_consumed_generator_arms_every_pass_then_disarms():
    """The study run loop's path: generate() exhausts the generator, so the
    injectors are armed for every forward pass and gone at the end."""
    model, hooked = _model(steps=6)
    counter = _Counter()
    chunks = list(_stream(model, observers=[counter]))

    assert "".join(chunks).split() == [f"tok{10 + i}" for i in range(6)]
    # 2 layers × (6 decode passes) — the observer was armed on every one.
    assert counter.fires == 2 * 6 == 2 * model.model.passes
    assert hooked.interventions == []

    # And the model is genuinely clean afterwards: a later session sees only
    # its own arm and restores to nothing.
    other = VectorInjector.single(1, [1.0] * HIDDEN, 1.0)
    with hooked.session([other]):
        assert hooked.interventions == [other]
    assert hooked.interventions == []


def test_two_streams_in_sequence_do_not_accumulate():
    model, hooked = _model(steps=3)
    for _ in range(2):
        list(_stream(model))
        assert hooked.interventions == []


def test_exception_mid_stream_leaves_the_model_disarmed():
    """A generation-thread failure propagates AND releases the arming — the
    traceback keeps the generator frame alive, which is one of the two ways
    hunt 1's hazard was reachable in production."""
    model, hooked = _model(steps=6, fail_at=2)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        list(_stream(model))
    assert hooked.interventions == []


def test_consumer_exception_mid_stream_leaves_the_model_disarmed():
    model, hooked = _model(steps=6)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        for _ in _stream(model):
            raise _Boom
    import gc
    gc.collect()
    assert hooked.interventions == []


# --- the arming primitive ----------------------------------------------------

def _bare_hooked():
    return HookedModel(_StreamingLM(steps=1))


def test_ordinary_nested_sessions_still_stack_and_restore():
    """Deliberate nesting is NOT abandonment: the inner exit must hand the
    outer scope its interventions back, exactly as it always did."""
    hooked = _bare_hooked()
    outer = VectorInjector.single(0, [1.0] * HIDDEN, 1.0)
    inner = VectorInjector.single(1, [1.0] * HIDDEN, 1.0)
    with hooked.session([outer]):
        assert hooked.interventions == [outer]
        with hooked.session([inner]):
            assert hooked.interventions == [inner]
        assert hooked.interventions == [outer]
    assert hooked.interventions == []


def test_disarming_a_superseded_handle_is_a_no_op():
    """Rule 1 in isolation: a handle that is no longer current restores
    nothing, so a deferred exit can never resurrect a dead scope."""
    hooked = _bare_hooked()
    stale = VectorInjector.single(0, [1.0] * HIDDEN, 1.0)
    live = VectorInjector.single(1, [1.0] * HIDDEN, 1.0)

    abandoned = hooked.arm([stale], abandonable=True)
    current = hooked.arm([live])
    assert hooked.interventions == [live]

    assert hooked.disarm(abandoned) is False       # superseded → no-op
    assert hooked.interventions == [live]          # the live scope is untouched
    assert hooked.disarm(current) is True
    assert hooked.interventions == []
    assert hooked.disarm(current) is False         # and it cannot fire twice


def test_arming_over_an_abandonable_scope_restores_to_bare_metal():
    hooked = _bare_hooked()
    stream_arm = hooked.arm([VectorInjector.single(0, [1.0] * HIDDEN, 1.0)],
                            abandonable=True)
    mine = VectorInjector.single(1, [1.0] * HIDDEN, 1.0)
    handle = hooked.arm([mine])
    assert hooked.interventions == [mine]
    hooked.disarm(handle)
    assert hooked.interventions == []
    hooked.disarm(stream_arm)
    assert hooked.interventions == []
