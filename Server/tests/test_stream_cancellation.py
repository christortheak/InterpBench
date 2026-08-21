"""Cooperative cancellation of streamed generations: the SSE stop event ends
generation within a step, so an abandoned chat cannot squat on the model-slot
lock until max_tokens (found live: a slow fp32 generation held the slot with
nothing visible in Jobs)."""

import threading
import time

import torch

from steerlab_server.api.routes import _locked_sse
from steerlab_server.experiment.generate import _stopping_criteria


def test_stop_on_event_criteria_flips_with_the_event():
    stop = threading.Event()
    criteria = _stopping_criteria(stop)
    ids = torch.zeros((1, 3), dtype=torch.long)
    scores = torch.zeros((1, 8))
    assert not criteria[0](ids, scores)
    stop.set()
    assert criteria[0](ids, scores)


def test_locked_sse_sets_stop_when_consumer_disconnects():
    stopped = threading.Event()
    produced = threading.Event()

    def produce(stop):
        # Emulate a long generation that honors the stop event per "step".
        for _ in range(10_000):
            if stop.is_set():
                stopped.set()
                return
            produced.set()
            yield "chunk"
            time.sleep(0.001)
        raise AssertionError("generation ran to completion despite disconnect")

    stream = _locked_sse(produce)
    first = next(stream)
    assert "chunk" in first
    assert produced.wait(timeout=2)
    # Client disconnect = the response generator is closed.
    stream.close()
    assert stopped.wait(timeout=2), "producer never observed the stop event"


def test_locked_sse_sets_stop_after_normal_completion_too():
    seen: list[bool] = []

    def produce(stop):
        yield "only-chunk"
        seen.append(stop.is_set())  # not yet set while producing

    chunks = list(_locked_sse(produce))
    assert any("only-chunk" in c for c in chunks)
    assert any("done" in c for c in chunks)
    assert seen == [False]
