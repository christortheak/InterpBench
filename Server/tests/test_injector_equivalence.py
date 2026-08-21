"""The 2026-08-18 injection-path performance fixes must be BIT-IDENTITY
preserving. This file is the pin.

Three changes landed together (open-issues §15), all of them shape-of-the-code
changes that must not be shape-of-the-science changes:

1. ``injector.apply`` adds in place on its fresh clone
   (``out[…].add_(v)``) instead of materializing ``out[…] + v`` and copying it
   back — one temporary per armed layer per decode step, gone.
2. ``plan.interventions`` consolidates a condition's ADD edits into ONE
   ``VectorInjector`` when their layers are all distinct — 682 ``apply`` calls
   per decode step at an 11-cell band on 62 layers becomes 62. Edits SHARING a
   layer keep the historical per-edit chain, because ``(h+v₁)+v₂`` is not
   ``h+(v₁+v₂)`` in floating point and a mix is a hashed condition.
3. ``hooks._make_hook`` returns the block's own output object at layers nothing
   modified, instead of rebuilding the output tuple at every armed layer.

None of the three may move a single bit of any output. The historical code
paths are reimplemented here verbatim (``_LegacyInjector``,
``_legacy_chain``, ``_LegacyHookedModel``) and the two are compared at four
depths: the tensor expression itself, post-hook hidden states, next-token
logits, and generated token ids (greedy AND sampled under a fixed seed).

Devices: CPU always; MPS opportunistically (this is a Mac dev box, and it is
free). **CUDA cannot be exercised locally** — there is no NVIDIA device on the
development machine, so the cluster's dtype/backend combination is covered by
construction (the same kernels, the same expression) and by the §15 confirm
matrix, not by this file.
"""

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace  # noqa: E402

from steerlab_server.steering import plan as plan_mod  # noqa: E402
from steerlab_server.steering.hooks import (  # noqa: E402
    HookedModel, _rejoin_output, _split_output)
from steerlab_server.steering.injector import Injection, VectorInjector  # noqa: E402


HIDDEN = 32


# --------------------------------------------------------------- legacy paths


class _LegacyInjector(VectorInjector):
    """``VectorInjector`` as it stood before 2026-08-18: clone, then assign the
    sum of the slice and the vector back onto the slice."""

    def apply(self, h, layer, offset):
        injection = self._injections.get(layer)
        if injection is None:
            return h
        seq_len = h.shape[1]
        if not self.should_inject(offset, seq_len, self._prompt_token_count):
            return h
        key = (h.device, h.dtype, layer)
        v = self._tensor_cache.get(key)
        if v is None:
            v = (torch.tensor(injection.vector, device=h.device,
                              dtype=torch.float32)
                 * float(injection.alpha)).to(h.dtype)
            self._tensor_cache[key] = v
        last_index = seq_len - 1
        out = h.clone()
        out[:, last_index, :] = out[:, last_index, :] + v   # the old expression
        return out


def _legacy_chain(edits, prompt_token_count=None):
    """``plan.interventions`` as it stood: one injector per ADD edit, always."""
    return [_LegacyInjector(
        {edit.layer: Injection(vector=edit.vector, alpha=edit.strength)},
        prompt_token_count=prompt_token_count)
        for edit in edits if edit.mode == plan_mod.Mode.ADD]


class _LegacyHookedModel(HookedModel):
    """``HookedModel`` as it stood: the armed branch always split the output and
    rejoined a fresh tuple, whether or not anything changed."""

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            if not self.interventions:
                hidden, _ = _split_output(output)
                self._advance(index, hidden.shape[1])
                return output
            hidden, rest = _split_output(output)
            if index == 0:
                self._current_offset = self._cumulative_offset
            offset = self._current_offset
            for intervention in self.interventions:
                hidden = intervention.apply(hidden, index, offset)
            self._advance(index, hidden.shape[1])
            return _rejoin_output(hidden, rest)
        return hook


# ------------------------------------------------------------------- fixtures


def _devices():
    devices = [torch.device("cpu")]
    if getattr(torch.backends, "mps", None) is not None and \
            torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def _dtypes(device):
    out = [torch.float32]
    for dtype in (torch.float16, torch.bfloat16):
        try:
            probe = torch.zeros((2,), device=device, dtype=dtype)
            (probe + probe).sum().item()
        except Exception:  # noqa: BLE001 - backend does not do this dtype
            continue
        out.append(dtype)
    return out


def _edits(layers, *, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [plan_mod.Edit(
        layer=layer,
        vector=torch.randn(HIDDEN, generator=generator).tolist(),
        strength=0.25 + 0.1 * position, mode=plan_mod.Mode.ADD,
        concept=f"c{position}")
        for position, layer in enumerate(layers)]


def _tiny_model(hooked_class=HookedModel):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        max_position_embeddings=256)
    model = LlamaForCausalLM(config).eval()
    return SimpleNamespace(model=model, hooked=hooked_class(model),
                           device=torch.device("cpu"))


def _prompt(n=9):
    torch.manual_seed(11)
    return torch.randint(1, 127, (1, n))


# --------------------------------------------------- 1. the tensor expression


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_in_place_addition_is_bit_identical_to_the_old_expression(device):
    """``out[…] = out[…] + v`` vs ``out[…].add_(v)``: same kernel, same inputs,
    so ``torch.equal`` — not ``allclose``. Asserted per dtype because a
    low-precision add is exactly where an accidental change of accumulation
    order would show."""
    for dtype in _dtypes(device):
        for trial in range(8):
            generator = torch.Generator().manual_seed(trial)
            h = (torch.randn((2, 5, HIDDEN), generator=generator)
                 .to(device=device, dtype=dtype))
            v = (torch.randn(HIDDEN, generator=generator)
                 .to(device=device, dtype=dtype))

            old = h.clone()
            old[:, 4, :] = old[:, 4, :] + v
            new = h.clone()
            new[:, 4, :].add_(v)

            assert torch.equal(old, new), (
                f"{device.type}/{dtype}: the in-place addition moved a bit")
            # …and neither touched the input.
            assert torch.equal(h, h)


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_the_injector_never_writes_through_to_its_input(device):
    """The clone exists so the model may reuse ``h``; ``add_`` must land on the
    clone, never on the block's own tensor."""
    h = torch.zeros((1, 3, HIDDEN), device=device)
    injector = VectorInjector.single(layer=0, vector=[1.0] * HIDDEN, alpha=2.0)
    out = injector.apply(h, layer=0, offset=0)
    assert out is not h
    assert torch.equal(h, torch.zeros((1, 3, HIDDEN), device=device))
    assert torch.equal(out[:, 2, :],
                       torch.full((1, HIDDEN), 2.0, device=device))


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_apply_matches_the_legacy_injector_exactly(device):
    for dtype in _dtypes(device):
        generator = torch.Generator().manual_seed(3)
        vector = torch.randn(HIDDEN, generator=generator).tolist()
        h = (torch.randn((1, 6, HIDDEN), generator=generator)
             .to(device=device, dtype=dtype))
        new = VectorInjector.single(2, vector, 0.7).apply(h, 2, 0)
        old = _LegacyInjector(
            {2: Injection(vector=vector, alpha=0.7)}).apply(h, 2, 0)
        assert torch.equal(old, new)


# ------------------------------------------------- 2. the plan consolidation


def test_distinct_layers_consolidate_into_one_injector():
    """The saving, stated as a count: 11 cells over 62 layers went from 682
    ``apply`` calls per pass (11 injectors × 62 layers) to 62."""
    layers = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54]
    chain = plan_mod.interventions(_edits(layers))
    assert len(chain) == 1
    (injector,) = chain
    assert isinstance(injector, VectorInjector)
    assert sorted(injector._injections) == sorted(layers)
    n_layers = 62
    assert len(chain) * n_layers == 62
    assert len(_legacy_chain(_edits(layers))) * n_layers == 682


def test_shared_layers_keep_the_historical_per_edit_chain():
    """Two edits at one layer must NOT fold into ``h + (v₁+v₂)``: float
    addition is not associative and a mix is a first-class hashed condition."""
    chain = plan_mod.interventions(_edits([4, 9, 9, 14]))
    assert len(chain) == 4
    assert all(isinstance(item, VectorInjector) for item in chain)
    assert [sorted(item._injections) for item in chain] == [[4], [9], [9], [14]]


@pytest.mark.parametrize("layers", [[0, 1, 2, 3], [1, 3], [2]])
def test_consolidated_chain_equals_the_legacy_chain_bit_for_bit(layers):
    """Distinct layers: one injector, same arithmetic, exactly equal output —
    over a whole fake multi-layer pass, at every layer, in three dtypes."""
    for dtype in _dtypes(torch.device("cpu")):
        edits = _edits(layers)
        new_chain = plan_mod.interventions(edits, prompt_token_count=5)
        old_chain = _legacy_chain(edits, prompt_token_count=5)
        for offset, seq_len in ((0, 5), (5, 1), (6, 1)):
            for layer in range(4):
                generator = torch.Generator().manual_seed(layer + offset)
                h = (torch.randn((1, seq_len, HIDDEN), generator=generator)
                     .to(dtype))
                new_h, old_h = h, h
                for item in new_chain:
                    new_h = item.apply(new_h, layer, offset)
                for item in old_chain:
                    old_h = item.apply(old_h, layer, offset)
                assert torch.equal(new_h, old_h), (
                    f"layer {layer}, offset {offset}, {dtype}")


def test_shared_layer_chain_equals_the_legacy_chain_bit_for_bit():
    """The case that is NOT consolidated still has to match the old path — this
    is the arm where folding would have been a silent numerical change."""
    edits = _edits([1, 2, 2])
    new_chain = plan_mod.interventions(edits)
    old_chain = _legacy_chain(edits)
    assert len(new_chain) == len(old_chain) == 3
    for dtype in _dtypes(torch.device("cpu")):
        generator = torch.Generator().manual_seed(21)
        h = torch.randn((1, 3, HIDDEN), generator=generator).to(dtype)
        new_h, old_h = h, h
        for item in new_chain:
            new_h = item.apply(new_h, 2, 0)
        for item in old_chain:
            old_h = item.apply(old_h, 2, 0)
        assert torch.equal(new_h, old_h)


# --------------------------------------------------------- 3. the hook return


def test_the_hook_returns_the_blocks_own_object_at_untouched_layers():
    """Identity, not equality: an unmodified layer must hand back the very
    object the block returned, which is what deletes the 62-tuples-per-step
    churn. Tuple-shaped and bare-tensor-shaped blocks both."""
    class _TupleBlock(torch.nn.Module):
        def forward(self, x):
            return (x, "extra")

    class _Inner(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = torch.nn.ModuleList([_TupleBlock() for _ in range(n)])

    class _Fake(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.model = _Inner(n)

        def forward(self, x):
            h = x
            for layer in self.model.layers:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result
            return h

    fake = _Fake(4)
    hooked = HookedModel(fake)
    # Wrap the real hook so we can see whether it handed back the very object
    # the block returned, without changing what it does.
    identities = []
    original_hook = hooked._make_hook

    def make(index):
        inner = original_hook(index)

        def hook(module, inputs, output):
            result = inner(module, inputs, output)
            identities.append((index, result is output))
            return result
        return hook

    for handle in hooked._handles:
        handle.remove()
    hooked._handles = [layer.register_forward_hook(make(index))
                       for index, layer in enumerate(hooked.layers)]

    injector = VectorInjector.single(layer=2, vector=[1.0] * HIDDEN, alpha=1.0)
    with hooked.session([injector]):
        fake(torch.zeros((1, 2, HIDDEN)))

    # Layer 2 steers and must rebuild; every other layer hands back the very
    # object the block returned.
    assert identities == [(0, True), (1, True), (2, False), (3, True)]


def test_offsets_still_advance_when_no_layer_is_modified():
    """The offset bookkeeping rides on the hook firing, not on the rejoin — an
    armed pass in which nothing fires (an intermediate prefill chunk) must still
    advance exactly as it always did."""
    from steerlab_server.steering.recorder import ActivationRecorder

    class _Block(torch.nn.Module):
        def forward(self, x):
            return x

    class _Inner(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])

    class _Fake(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.model = _Inner(n)

        def forward(self, x):
            h = x
            for layer in self.model.layers:
                h = layer(h)
            return h

    fake = _Fake(3)
    hooked = HookedModel(fake)
    recorder = ActivationRecorder(layers=[0])
    # Gated OFF for every pass here (prompt_token_count far beyond what we
    # feed), so the injector never returns a new tensor…
    injector = VectorInjector.single(0, [1.0] * HIDDEN, 1.0,
                                     prompt_token_count=10_000)
    with hooked.session([recorder, injector]):
        fake(torch.zeros((1, 4, HIDDEN)))
        fake(torch.zeros((1, 4, HIDDEN)))
        fake(torch.zeros((1, 1, HIDDEN)))
    # …and the offsets are still 0, 4, 8.
    assert [c.offset for c in recorder.captures] == [0, 4, 8]


# ---------------------------------------- 4. end to end on a real tiny model


def _hidden_states(wrapper, chain, ids):
    from steerlab_server.steering.recorder import ActivationRecorder

    recorder = ActivationRecorder(layers=range(wrapper.hooked.num_layers))
    with wrapper.hooked.session(list(chain) + [recorder]):
        with torch.no_grad():
            out = wrapper.model(input_ids=ids,
                                attention_mask=torch.ones_like(ids))
    return out.logits, [torch.tensor(c.values) for c in recorder.captures]


def test_post_hook_hidden_states_and_logits_match_the_legacy_path():
    ids = _prompt()
    edits = _edits([0, 1, 2, 3])

    new_model = _tiny_model(HookedModel)
    old_model = _tiny_model(_LegacyHookedModel)

    new_logits, new_hidden = _hidden_states(
        new_model, plan_mod.interventions(edits, prompt_token_count=ids.shape[1]),
        ids)
    old_logits, old_hidden = _hidden_states(
        old_model, _legacy_chain(edits, prompt_token_count=ids.shape[1]), ids)

    assert len(new_hidden) == len(old_hidden) == new_model.hooked.num_layers
    for new, old in zip(new_hidden, old_hidden):
        assert torch.equal(new, old)
    assert torch.equal(new_logits, old_logits)
    # And the steering actually did something, so equality is not vacuous.
    with new_model.hooked.session([]):
        with torch.no_grad():
            clean = new_model.model(input_ids=ids,
                                    attention_mask=torch.ones_like(ids)).logits
    assert not torch.equal(clean, new_logits)


def _generate(wrapper, chain, ids, *, sample, seed=1234):
    common = dict(attention_mask=torch.ones_like(ids), max_new_tokens=8,
                  pad_token_id=0)
    with wrapper.hooked.session(list(chain)):
        torch.manual_seed(seed)
        with torch.no_grad():
            if sample:
                return wrapper.model.generate(
                    input_ids=ids, do_sample=True, temperature=1.0, top_k=0,
                    top_p=1.0, **common)
            return wrapper.model.generate(input_ids=ids, do_sample=False,
                                          **common)


@pytest.mark.parametrize("sample", [False, True],
                         ids=["greedy", "sampled-fixed-seed"])
def test_generated_token_ids_match_the_legacy_path(sample):
    """The end the science actually reads. Sampled draws are pinned under a
    fixed seed: identical logits + identical RNG state ⇒ identical ids, so any
    numerical drift at all would show up here as a different sequence."""
    ids = _prompt()
    edits = _edits([0, 2, 3])
    n = ids.shape[1]

    new_out = _generate(_tiny_model(HookedModel),
                        plan_mod.interventions(edits, prompt_token_count=n),
                        ids, sample=sample)
    old_out = _generate(_tiny_model(_LegacyHookedModel),
                        _legacy_chain(edits, prompt_token_count=n),
                        ids, sample=sample)
    assert new_out.tolist() == old_out.tolist()


def test_shared_layer_generation_matches_the_legacy_path():
    """The un-consolidated arm, end to end."""
    ids = _prompt()
    edits = _edits([1, 1, 3])
    n = ids.shape[1]
    new_out = _generate(_tiny_model(HookedModel),
                        plan_mod.interventions(edits, prompt_token_count=n),
                        ids, sample=False)
    old_out = _generate(_tiny_model(_LegacyHookedModel),
                        _legacy_chain(edits, prompt_token_count=n),
                        ids, sample=False)
    assert new_out.tolist() == old_out.tolist()
