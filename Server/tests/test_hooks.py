"""Hook-manager dispatch + KV-offset tracking, on a tiny fake model (CPU only).

This is the engine-level analogue of the Swift smoke test's per-token hook-fire
assertion: it proves the forward hooks fire on every pass (prefill + each decode
step) with correct ``offset`` labelling, without loading a real model.
"""

import torch
import torch.nn as nn

from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.injector import VectorInjector
from steerlab_server.steering.recorder import ActivationRecorder


class _Block(nn.Module):
    def forward(self, x):  # identity block; returns a bare tensor
        return x


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])


class _FakeModel(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.model = _Inner(n_layers)
        self.config = type("cfg", (), {"hidden_size": hidden})()

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_offsets_advance_across_prefill_and_decode():
    hidden = 3
    fake = _FakeModel(n_layers=4, hidden=hidden)
    hooked = HookedModel(fake)
    recorder = ActivationRecorder(layers=range(4))  # last-token capture
    with hooked.session([recorder]):
        fake(torch.zeros((1, 5, hidden)))   # prefill, seq 5 → offset 0
        fake(torch.zeros((1, 1, hidden)))   # decode 1   → offset 5
        fake(torch.zeros((1, 1, hidden)))   # decode 2   → offset 6
    offsets_per_layer0 = [c.offset for c in recorder.captures if c.layer == 0]
    assert offsets_per_layer0 == [0, 5, 6]
    # Every layer captured on every pass: 4 layers × 3 passes.
    assert len(recorder.captures) == 12


def test_injector_through_hooks_modifies_last_position():
    hidden = 3
    fake = _FakeModel(n_layers=2, hidden=hidden)
    hooked = HookedModel(fake)
    injector = VectorInjector.single(layer=1, vector=[1.0, 1.0, 1.0], alpha=2.0)
    x = torch.zeros((1, 4, hidden))
    with hooked.session([injector]):
        out = fake(x)
    assert torch.allclose(out[0, :3], torch.zeros((3, hidden)))
    assert torch.allclose(out[0, 3], torch.full((hidden,), 2.0))


def test_interventions_cleared_after_session():
    fake = _FakeModel(n_layers=2, hidden=2)
    hooked = HookedModel(fake)
    with hooked.session([VectorInjector.single(0, [1.0, 1.0], 1.0)]):
        pass
    assert hooked.interventions == []
    # With no interventions armed, output is unchanged.
    out = fake(torch.ones((1, 2, 2)))
    assert torch.allclose(out, torch.ones((1, 2, 2)))
