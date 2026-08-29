"""True SAE latent intervention: exact h' arithmetic, JumpReLU gate behaviour,
firing-gate parity with ``VectorInjector``, chain composition, and the manifest
contract.

Everything here is OFFLINE and synthetic: the SAE weights are hand-built so the
expected residual state can be written down in closed form, and the loader seam
is substituted. No model loads, no HuggingFace, no ``sae_lens``.
"""

import hashlib
import json
import math
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import sae_latent as condition_mod
from steerlab_server.steering import plan
from steerlab_server.steering.injector import VectorInjector
from steerlab_server.steering.sae_latent import (
    ADD, CLAMP, SAELatentEdit, SAELatentFeature, SAELatentIntervention,
    group_edits)


# --------------------------------------------------------------------------
# A synthetic one-feature SAE whose arithmetic is checkable by hand
# --------------------------------------------------------------------------

def feature(*, encoder=(1.0, 0.0, 0.0, 0.0), decoder=(0.0, 1.0, 0.0, 0.0),
            bias=0.0, threshold=1.0) -> SAELatentFeature:
    """Encoder reads component 0, decoder writes component 1 — so the induced
    delta is visible in a coordinate the encoder does not read, which keeps the
    expected value independent of the edit itself."""
    return SAELatentFeature(encoder_row=encoder, decoder_row=decoder,
                            encoder_bias=bias, threshold=threshold)


def state(component0: float, hidden: int = 4, seq: int = 1,
          dtype=torch.float32) -> torch.Tensor:
    h = torch.zeros((1, seq, hidden), dtype=dtype)
    h[0, seq - 1, 0] = component0
    return h


def edit_at(layer=0, *, mode=ADD, beta=1.0, **kwargs) -> SAELatentEdit:
    return SAELatentEdit(layer=layer, feature=feature(**kwargs), mode=mode,
                         beta=beta, feature_id=7, label="probe")


# --------------------------------------------------------------------------
# JumpReLU gate
# --------------------------------------------------------------------------

def test_jumprelu_gate_is_strictly_above_threshold():
    """act(pre) = pre if pre > θ else 0 — the Gemma Scope form. Exactly AT the
    threshold the feature is off (the published rule is a strict inequality)."""
    intervention = SAELatentIntervention.single(edit_at(beta=0.0))
    for pre, expected in ((0.5, 0.0), (1.0, 0.0), (1.5, 1.5)):
        readout = intervention.latent_readout(state(pre), layer=0)
        assert readout["latent"][0][0] == pytest.approx(expected)


def test_encoder_bias_shifts_the_pre_activation():
    """The bias is part of the encode, including the folded b_dec correction —
    a feature whose raw projection is below θ can still be active."""
    intervention = SAELatentIntervention.single(
        edit_at(beta=0.0, bias=2.0, threshold=1.0))
    readout = intervention.latent_readout(state(0.5), layer=0)
    assert readout["preActivation"][0][0] == pytest.approx(2.5)
    assert readout["latent"][0][0] == pytest.approx(2.5)


# --------------------------------------------------------------------------
# add: edits the PRE-activation, gate re-evaluated
# --------------------------------------------------------------------------

def test_add_on_an_active_feature_decodes_exactly_beta():
    """pre=3 > θ=1, and pre+β=5 > θ, so Δz = β and h' = h + β·W_dec."""
    intervention = SAELatentIntervention.single(edit_at(mode=ADD, beta=2.0))
    out = intervention.apply(state(3.0), layer=0, offset=0)
    assert out[0, 0].tolist() == [3.0, 2.0, 0.0, 0.0]


def test_add_activates_a_dormant_feature_only_when_beta_clears_the_gate():
    """A dormant feature (pre ≤ θ) is activated only if pre+β > θ, and the
    induced delta is then the WHOLE (pre+β), not β."""
    intervention = SAELatentIntervention.single(edit_at(mode=ADD, beta=1.5))
    # pre = 0.2, θ = 1.0 → z = 0; pre + β = 1.7 > θ → z' = 1.7; Δ = 1.7.
    out = intervention.apply(state(0.2), layer=0, offset=0)
    assert out[0, 0, 1].item() == pytest.approx(1.7)


def test_add_is_a_noop_on_a_feature_dormant_by_more_than_beta():
    """The mode that distinguishes a latent edit from decoder-direction
    addition: below the gate by more than β, nothing happens, and the residual
    stream is bit-identical to baseline."""
    intervention = SAELatentIntervention.single(edit_at(mode=ADD, beta=0.5))
    h = state(0.2)  # pre + β = 0.7 ≤ θ = 1.0
    out = intervention.apply(h, layer=0, offset=0)
    assert torch.equal(out, h)


def test_add_with_a_negative_beta_can_close_the_gate_and_remove_the_feature():
    """pre=1.5 (active, z=1.5); β=−1.0 → pre+β=0.5 ≤ θ → z'=0, Δ = −1.5."""
    intervention = SAELatentIntervention.single(edit_at(mode=ADD, beta=-1.0))
    out = intervention.apply(state(1.5), layer=0, offset=0)
    assert out[0, 0, 1].item() == pytest.approx(-1.5)


def test_add_is_not_decoder_direction_addition_in_general():
    """The whole reason 'add' edits the pre-activation: post-activation
    addition would be identically h + β·W_dec for every state, which is the
    decoder-direction mechanism this condition type must never be conflated
    with. Here two states with the same β give different deltas."""
    intervention = SAELatentIntervention.single(edit_at(mode=ADD, beta=1.5))
    active = intervention.apply(state(3.0), layer=0, offset=0)[0, 0, 1].item()
    dormant = intervention.apply(state(0.2), layer=0, offset=0)[0, 0, 1].item()
    assert active == pytest.approx(1.5)
    assert dormant == pytest.approx(1.7)
    assert active != pytest.approx(dormant)


# --------------------------------------------------------------------------
# clamp: sets the POST-activation latent outright
# --------------------------------------------------------------------------

def test_clamp_activates_a_dormant_feature():
    """z_f = 0 → Δ = β·W_dec: clamp CAN activate a dormant feature, whatever
    the pre-activation, which is the mode's whole point."""
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=4.0))
    out = intervention.apply(state(-10.0), layer=0, offset=0)
    assert out[0, 0].tolist() == [-10.0, 4.0, 0.0, 0.0]


def test_clamp_on_an_active_feature_decodes_the_difference():
    """pre=3 > θ → z=3; β=4 → Δ = (4 − 3) = 1."""
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=4.0))
    out = intervention.apply(state(3.0), layer=0, offset=0)
    assert out[0, 0, 1].item() == pytest.approx(1.0)


def test_clamp_to_zero_removes_exactly_this_features_contribution():
    """Per-feature ablation: Δ = −z_f·W_dec where the feature is active, and
    exactly nothing where it is not."""
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=0.0))
    assert intervention.apply(state(3.0), layer=0, offset=0)[0, 0, 1].item() \
        == pytest.approx(-3.0)
    dormant = state(0.5)
    assert torch.equal(intervention.apply(dormant, layer=0, offset=0), dormant)


def test_clamp_refuses_a_negative_beta_at_construction():
    """A JumpReLU latent is non-negative by construction, so a negative clamp
    target names a state the SAE cannot represent."""
    with pytest.raises(ValueError, match="non-negative"):
        edit_at(mode=CLAMP, beta=-1.0)


def test_unknown_mode_and_non_finite_beta_refuse_at_construction():
    with pytest.raises(ValueError, match="unknown SAE latent mode"):
        SAELatentEdit(layer=0, feature=feature(), mode="scale", beta=1.0)
    with pytest.raises(ValueError, match="finite"):
        SAELatentEdit(layer=0, feature=feature(), mode=ADD, beta=math.inf)


# --------------------------------------------------------------------------
# Only the delta is decoded — reconstruction error never enters the stream
# --------------------------------------------------------------------------

def test_a_null_edit_leaves_the_residual_stream_bit_identical():
    """The reconstruction-error argument, made testable: a full encode→decode
    round trip would replace h with the SAE's (lossy) reconstruction even at
    β=0. Decoding ONLY the induced delta means β=0 is exactly baseline."""
    intervention = SAELatentIntervention.single(edit_at(mode=ADD, beta=0.0))
    h = torch.randn((1, 3, 4), dtype=torch.float32)
    assert torch.equal(intervention.apply(h, layer=0, offset=0), h)


def test_the_edit_lands_only_along_the_decoder_row():
    """h' − h is a multiple of W_dec[f] and of nothing else — no b_dec, no
    residual of the dictionary's other features."""
    decoder = (0.3, -0.5, 0.8, 0.1)
    intervention = SAELatentIntervention.single(
        edit_at(mode=CLAMP, beta=2.0, decoder=decoder))
    h = torch.randn((1, 2, 4), dtype=torch.float32)
    delta = (intervention.apply(h, layer=0, offset=0) - h)[0, 1]
    scale = delta[0].item() / decoder[0]
    assert delta.tolist() == pytest.approx(
        [scale * value for value in decoder], abs=1e-6)


# --------------------------------------------------------------------------
# Position / firing rule — parity with VectorInjector
# --------------------------------------------------------------------------

def test_applies_at_the_last_position_only():
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=2.0))
    h = torch.zeros((1, 3, 4), dtype=torch.float32)
    h[0, :, 0] = 5.0  # every position would activate the feature
    out = intervention.apply(h, layer=0, offset=0)
    assert out[0, 0, 1].item() == 0.0
    assert out[0, 1, 1].item() == 0.0
    assert out[0, 2, 1].item() != 0.0


def test_untouched_layer_returns_the_same_tensor():
    intervention = SAELatentIntervention.single(edit_at(layer=5))
    h = torch.ones((1, 2, 4), dtype=torch.float32)
    assert intervention.apply(h, layer=4, offset=0) is h


def test_never_mutates_the_input_tensor():
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=3.0))
    h = state(2.0)
    before = h.clone()
    intervention.apply(h, layer=0, offset=0)
    assert torch.equal(h, before)


@pytest.mark.parametrize(
    "offset,seq_len,prompt_token_count",
    [(0, 10, None), (0, 10, 10), (10, 1, 10), (11, 1, 10),
     (0, 50, 100), (50, 50, 100), (0, 1, 1), (0, 99, 100), (99, 1, 100)])
def test_gate_is_literally_the_injectors_gate(offset, seq_len, prompt_token_count):
    """Parity by delegation, asserted across the chunked-prefill scenarios the
    injector's own tests cover: two copies of this arithmetic would be two
    chances for the prefill-only bug to come back on one of them."""
    assert (SAELatentIntervention.should_inject(offset, seq_len, prompt_token_count)
            is VectorInjector.should_inject(offset, seq_len, prompt_token_count))


def test_suppressed_on_a_mid_prompt_prefill_chunk_and_fired_at_the_prompt_end():
    intervention = SAELatentIntervention.single(
        edit_at(mode=CLAMP, beta=2.0), prompt_token_count=100)
    chunk = torch.zeros((1, 50, 4), dtype=torch.float32)
    chunk[0, :, 0] = 5.0
    # First prefill chunk ends at position 49, mid-prompt → bit-identical.
    assert torch.equal(intervention.apply(chunk, layer=0, offset=0), chunk)
    # Final prefill chunk reaches the prompt end → the edit lands.
    assert intervention.apply(chunk, layer=0, offset=50)[0, 49, 1].item() != 0.0


def test_fires_on_every_decode_step():
    """The standing hard requirement: steering only during prefill silently
    produces near-null results."""
    intervention = SAELatentIntervention.single(
        edit_at(mode=CLAMP, beta=2.0), prompt_token_count=12)
    for step in range(30):
        h = state(0.0, seq=1)
        out = intervention.apply(h, layer=0, offset=12 + step)
        assert out[0, 0, 1].item() == pytest.approx(2.0)


# --------------------------------------------------------------------------
# dtype / device cache
# --------------------------------------------------------------------------

def test_math_runs_in_float32_and_the_delta_is_cast_back_to_h_dtype():
    """The residual stream is float (bf16/fp16) even for quantized models; the
    encode/edit/decode arithmetic runs in float32 and only the delta is cast."""
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=2.0))
    h = state(3.0, dtype=torch.bfloat16)
    out = intervention.apply(h, layer=0, offset=0)
    assert out.dtype == torch.bfloat16
    assert float(out[0, 0, 1]) == pytest.approx(-1.0, abs=1e-2)


def test_tensors_are_cached_per_device_dtype_layer():
    intervention = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=1.0))
    assert intervention._tensor_cache == {}
    first = state(1.0)
    intervention.apply(first, layer=0, offset=0)
    keys = list(intervention._tensor_cache)
    assert keys == [(first.device, torch.float32, 0)]
    cached = intervention._tensor_cache[keys[0]]
    # A second pass in the same (device, dtype, layer) reuses the tensors...
    intervention.apply(state(2.0), layer=0, offset=0)
    assert intervention._tensor_cache[keys[0]] is cached
    # ...and a different dtype is a different key, never a silent cache hit.
    intervention.apply(state(2.0, dtype=torch.float16), layer=0, offset=0)
    assert len(intervention._tensor_cache) == 2
    assert all(tensor.dtype == torch.float32
               for tensor in intervention._tensor_cache[keys[0]][:5])


def test_dimension_mismatch_refuses_rather_than_broadcasting():
    intervention = SAELatentIntervention.single(edit_at())
    with pytest.raises(ValueError, match="wrong SAE for this model"):
        intervention.apply(torch.zeros((1, 1, 8)), layer=0, offset=0)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

def _delta(intervention, h, layer=0, offset=0):
    return intervention.apply(h, layer, offset) - h


def test_latent_plus_vector_injection_is_the_sum_of_their_deltas():
    """Both are additive residual edits at the last position, and
    ``plan.interventions`` orders the (state-dependent) latent edit BEFORE the
    (pure-offset) injectors, so the composed result is exactly the sum of what
    each does alone — for an arbitrary injected vector."""
    h = state(3.0)
    latent = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=5.0))
    injector = VectorInjector.single(layer=0, vector=[0.7, -0.2, 1.1, 0.4],
                                     alpha=2.0)
    expected = h + _delta(latent, h) + _delta(injector, h)

    chain = plan.interventions(
        [plan.Edit(layer=0, vector=[0.7, -0.2, 1.1, 0.4], strength=2.0,
                   mode=plan.Mode.ADD, concept="c")],
        latent_edits=[edit_at(mode=CLAMP, beta=5.0)])
    composed = h
    for intervention in chain:
        composed = intervention.apply(composed, 0, 0)
    assert torch.allclose(composed, expected, atol=1e-6)


def test_the_reverse_order_would_not_be_additive():
    """Why the ordering in ``plan.interventions`` is a contract: an injected
    offset that is NOT orthogonal to the encoder direction moves the SAE's
    pre-activation, so running the injector first changes what the latent edit
    computes."""
    h = state(3.0)
    latent = SAELatentIntervention.single(edit_at(mode=CLAMP, beta=5.0))
    injector = VectorInjector.single(layer=0, vector=[1.0, 0.0, 0.0, 0.0],
                                     alpha=4.0)
    additive = injector.apply(latent.apply(h, 0, 0), 0, 0)
    reversed_order = latent.apply(injector.apply(h, 0, 0), 0, 0)
    assert not torch.allclose(additive, reversed_order)


def test_plan_places_the_latent_intervention_after_ablation_before_injection():
    chain = plan.interventions(
        [plan.Edit(layer=0, vector=[1.0, 0.0, 0.0, 0.0], strength=1.0,
                   mode=plan.Mode.ABLATE, concept="a"),
         plan.Edit(layer=0, vector=[0.0, 1.0, 0.0, 0.0], strength=1.0,
                   mode=plan.Mode.ADD, concept="b")],
        latent_edits=[edit_at()])
    assert plan.satisfies_ordering_invariant(chain)
    assert plan.satisfies_latent_ordering_invariant(chain)
    assert isinstance(chain[1], SAELatentIntervention)
    assert len(chain) == 3


def test_plan_without_latent_edits_is_unchanged():
    chain = plan.interventions(
        [plan.Edit(layer=0, vector=[1.0, 0.0], strength=1.0,
                   mode=plan.Mode.ADD, concept="b")])
    assert [type(item) for item in chain] == [VectorInjector]
    assert plan.satisfies_latent_ordering_invariant(chain)


def test_two_features_at_a_layer_are_encoded_from_one_read_and_summed():
    """The ablator's 'one subspace' rule, applied to latents: feature g's
    pre-activation must be computed against the unedited state, or a
    multi-feature condition would be order-dependent."""
    # f reads component 0 and writes component 0 — so a cascade WOULD move g.
    f = SAELatentEdit(layer=0, mode=CLAMP, beta=6.0, feature_id=1, label="f",
                      feature=feature(encoder=(1.0, 0.0, 0.0, 0.0),
                                      decoder=(1.0, 0.0, 0.0, 0.0)))
    g = SAELatentEdit(layer=0, mode=ADD, beta=1.0, feature_id=2, label="g",
                      feature=feature(encoder=(1.0, 0.0, 0.0, 0.0),
                                      decoder=(0.0, 1.0, 0.0, 0.0)))
    together = SAELatentIntervention({0: [f, g]})
    h = state(2.0)
    out = together.apply(h, layer=0, offset=0)
    # f: z=2 → z'=6 → Δ = 4 along component 0.
    # g: pre=2 (from the UNEDITED state), z=2 → z'=3 → Δ = 1 along component 1.
    # A cascade would have given g pre = 6 and a delta of 1 anyway, but from a
    # state the condition never declared; the readout pins which one ran.
    assert out[0, 0].tolist() == pytest.approx([6.0, 1.0, 0.0, 0.0])
    readout = together.latent_readout(h, layer=0)
    assert readout["preActivation"][0] == pytest.approx([2.0, 2.0])


def test_group_edits_orders_by_feature_id_and_label():
    a = edit_at()
    b = SAELatentEdit(layer=0, feature=feature(), mode=ADD, beta=1.0,
                      feature_id=2, label="b")
    grouped = group_edits([a, b])
    assert [e.feature_id for e in grouped[0]] == [2, 7]


# --------------------------------------------------------------------------
# Manifest contract
# --------------------------------------------------------------------------

def valid_entry(**overrides) -> dict:
    entry = {"name": "sympathy-clamp-10", "interventionType": "saeLatent",
             "serverOnly": True, "release": "gemma-scope-2-27b-it-res",
             "saeID": "layer_31_width_16k_l0_medium", "feature": 12345,
             "mode": "clamp", "beta": 10.0, "constructLabel": "sympathy"}
    entry.update(overrides)
    return {k: v for k, v in entry.items() if v is not _ABSENT}


_ABSENT = object()


def test_a_valid_latent_condition_parses():
    specs = condition_mod.parse({"saeLatentConditions": [valid_entry()]})
    assert len(specs) == 1
    assert specs[0].mode == "clamp" and specs[0].beta == 10.0
    assert specs[0].feature == 12345


def test_absent_key_is_not_a_violation():
    assert condition_mod.condition_violations({"conditions": []}) == []
    assert condition_mod.parse({}) == []


@pytest.mark.parametrize("overrides,expected", [
    ({"mode": "scale"}, "unknown mode"),
    ({"mode": _ABSENT}, "missing required key"),
    ({"beta": _ABSENT}, "missing required key"),
    ({"beta": "10"}, "finite number"),
    ({"beta": -1.0}, "clamp beta must be >= 0"),
    ({"serverOnly": _ABSENT}, "missing required key"),
    ({"serverOnly": False}, "serverOnly"),
    ({"interventionType": "vector"}, "interventionType"),
    ({"feature": -1}, "non-negative integer"),
    ({"feature": "12345"}, "non-negative integer"),
    ({"release": ""}, "release must be a non-empty string"),
    ({"name": ""}, "non-empty name"),
    ({"layer": "31"}, "layer must be a non-negative integer"),
    ({"constructLabel": "a:b"}, "may not contain ':'"),
    ({"betaa": 3.0}, "unknown key"),
])
def test_malformed_latent_conditions_refuse(overrides, expected):
    raw = {"saeLatentConditions": [valid_entry(**overrides)]}
    violations = condition_mod.condition_violations(raw)
    assert any(expected in v for v in violations), violations
    with pytest.raises(ValueError):
        condition_mod.parse(raw)


def test_a_latent_condition_misfiled_in_conditions_is_a_violation():
    """Left inside ``conditions`` it has no slots, so it would run as a
    baseline arm under a steered condition's name — the failure the separate
    key exists to prevent."""
    raw = {"conditions": [{"name": "clamp-10", "interventionType": "saeLatent"}]}
    violations = condition_mod.condition_violations(raw)
    assert any("saeLatentConditions" in v and "slotless" in v
               for v in violations), violations


def test_an_unknown_intervention_type_on_an_ordinary_condition_refuses():
    raw = {"conditions": [{"name": "c", "interventionType": "sparseMixture"}]}
    assert any("unknown interventionType" in v
               for v in condition_mod.condition_violations(raw))


def test_an_ordinary_condition_without_an_intervention_type_is_untouched():
    raw = {"conditions": [{"name": "c", "slots": [
        {"concept": "x", "layer": 3, "alpha": 1.0}]}]}
    assert condition_mod.condition_violations(raw) == []


def test_name_collisions_across_condition_surfaces_refuse():
    raw = {"conditions": [{"name": "arm"}],
           "saeLatentConditions": [valid_entry(name="arm")]}
    assert any("must be unique" in v
               for v in condition_mod.condition_violations(raw))


# --------------------------------------------------------------------------
# Materialization through the injectable loader seam (offline)
# --------------------------------------------------------------------------

class FakeLoaded:
    def __init__(self, **kwargs):
        self.encoder_row = kwargs.get("encoder_row", [1.0, 0.0, 0.0, 0.0])
        self.decoder_row = kwargs.get("decoder_row", [0.0, 1.0, 0.0, 0.0])
        self.encoder_bias = kwargs.get("encoder_bias", -0.25)
        self.threshold = kwargs.get("threshold", 1.5)
        self.repo_id = kwargs.get("repo_id", "google/gemma-scope-2-27b-it")
        self.repo_revision = kwargs.get("repo_revision", "a" * 40)
        self.config = kwargs.get("config", {"hook_layer": 31, "d_sae": 16384})
        self.sparsity = kwargs.get("sparsity", 0.001)
        self.b_dec_folded = kwargs.get("b_dec_folded", True)
        self.activation = kwargs.get("activation", "jumprelu")


def fake_loader(**kwargs):
    calls = []

    def load(release, sae_id, feature):
        calls.append((release, sae_id, feature))
        return FakeLoaded(**kwargs)

    load.calls = calls
    return load


def test_materialize_builds_the_edit_and_stamps_the_pinned_commit():
    spec = condition_mod.parse({"saeLatentConditions": [valid_entry()]})[0]
    loader = fake_loader()
    edit, provenance = condition_mod.materialize(spec, loader=loader)

    assert loader.calls == [("gemma-scope-2-27b-it-res",
                             "layer_31_width_16k_l0_medium", 12345)]
    assert edit.layer == 31 and edit.mode == "clamp" and edit.beta == 10.0
    assert edit.feature.threshold == 1.5
    assert edit.feature.encoder_bias == -0.25

    assert provenance["repositoryRevision"] == "a" * 40
    assert provenance["interventionType"] == "saeLatent"
    assert provenance["activation"] == "jumprelu"
    assert provenance["bDecFoldedIntoBias"] is True
    assert provenance["saeConfigHash"].startswith("sha256:")
    assert provenance["encoderRowHash"] != provenance["decoderRowHash"]
    assert provenance["width"] == "16k"


def test_beta_is_stamped_as_a_latent_unit_never_an_alpha():
    """β is in the feature's own activation scale. The provenance says so in
    two forms so neither a human nor a viewer can read it as a dose in
    residual-norm units, and records that no dose calibration was run."""
    spec = condition_mod.parse({"saeLatentConditions": [valid_entry()]})[0]
    _edit, provenance = condition_mod.materialize(spec, loader=fake_loader())
    assert provenance["betaUnits"] == "latent"
    assert provenance["latentUnits"] is True
    assert provenance["doseCalibrated"] is False
    assert "alpha" not in provenance and "alphaInNormUnits" not in provenance


def test_materialize_refuses_an_unpinned_source():
    spec = condition_mod.parse({"saeLatentConditions": [valid_entry()]})[0]
    with pytest.raises(ValueError, match="repository id/revision"):
        condition_mod.materialize(spec, loader=fake_loader(repo_revision=""))


def test_materialize_refuses_a_declared_layer_the_sae_contradicts():
    spec = condition_mod.parse(
        {"saeLatentConditions": [valid_entry(layer=17)]})[0]
    with pytest.raises(ValueError, match="disagrees with the SAE's own layer"):
        condition_mod.materialize(spec, loader=fake_loader())


def test_materialize_derives_the_layer_from_the_sae_id_when_config_is_silent():
    spec = condition_mod.parse({"saeLatentConditions": [valid_entry()]})[0]
    edit, provenance = condition_mod.materialize(
        spec, loader=fake_loader(config={}))
    assert edit.layer == 31 and provenance["layer"] == 31


def test_materialize_refuses_when_the_layer_is_undeterminable():
    spec = condition_mod.parse(
        {"saeLatentConditions": [valid_entry(saeID="custom-dictionary")]})[0]
    with pytest.raises(ValueError, match="cannot determine the layer"):
        condition_mod.materialize(spec, loader=fake_loader(config={}))


# --------------------------------------------------------------------------
# Encoder/decoder ORIENTATION — verified from the artifact, never guessed
# --------------------------------------------------------------------------

class FakeTensor:
    """The 2-D / 1-D duck type the orientation helpers read: ``.detach()``,
    ``.float()``, ``.cpu()``, ``.shape``, ``.ndim`` and indexing. Built on
    torch so the arithmetic is real, without loading an SAE."""

    def __init__(self, tensor):
        self._t = tensor

    def detach(self):
        return self._t

    @property
    def shape(self):
        return self._t.shape

    @property
    def ndim(self):
        return self._t.ndim


class FakeSAE:
    def __init__(self, *, d_in=4, d_sae=8, transpose_encoder=False):
        torch.manual_seed(0)
        w_enc = torch.randn(d_in, d_sae)
        self.W_enc = FakeTensor(w_enc.T.contiguous() if transpose_encoder else w_enc)
        self.W_dec = FakeTensor(torch.randn(d_sae, d_in))
        self.b_enc = FakeTensor(torch.randn(d_sae))
        self.b_dec = FakeTensor(torch.randn(d_in))
        self.threshold = FakeTensor(torch.rand(d_sae))


def test_dimensions_come_from_the_biases_which_pin_the_orientation():
    """``b_enc`` has length d_sae and ``b_dec`` has length d_in — that pins the
    orientation without guessing, and transposing W_enc would silently
    substitute a decoder row for an encoder column."""
    from steerlab_server.experiment import gemma_scope

    assert gemma_scope._sae_dimensions(FakeSAE(), {}) == (4, 8)
    # The config must AGREE with the biases; an inconsistent SAE refuses.
    with pytest.raises(RuntimeError, match="d_sae=99"):
        gemma_scope._sae_dimensions(FakeSAE(), {"d_sae": 99})


def test_dimensions_fall_back_to_the_decoder_shape_and_refuse_the_square_case():
    from steerlab_server.experiment import gemma_scope

    bare = FakeSAE()
    del bare.b_enc, bare.b_dec
    assert gemma_scope._sae_dimensions(bare, {}) == (4, 8)
    square = FakeSAE(d_in=6, d_sae=6)
    del square.b_enc, square.b_dec
    with pytest.raises(RuntimeError, match="genuinely ambiguous"):
        gemma_scope._sae_dimensions(square, {})


@pytest.mark.parametrize("transposed", [False, True])
def test_the_encoder_column_is_read_the_same_way_from_either_storage(transposed):
    """Both storage orders yield the SAME encoder column for a feature; an
    unrecognised shape refuses rather than being reinterpreted."""
    from steerlab_server.experiment import gemma_scope

    canonical = FakeSAE(transpose_encoder=False)
    other = FakeSAE(transpose_encoder=transposed)
    expected = gemma_scope._oriented_row(canonical.W_enc, feature=3, d_in=4,
                                         d_sae=8, which="encoder")
    assert gemma_scope._oriented_row(other.W_enc, feature=3, d_in=4, d_sae=8,
                                     which="encoder") == expected
    assert len(expected) == 4


def test_an_unrecognised_weight_shape_refuses():
    from steerlab_server.experiment import gemma_scope

    with pytest.raises(RuntimeError, match="matches neither"):
        gemma_scope._oriented_row(FakeTensor(torch.randn(5, 7)), feature=0,
                                  d_in=4, d_sae=8, which="encoder")


def test_intervention_state_nulls_the_vector_only_fields():
    """Records keep ONE parseable shape across mechanisms: a latent condition
    has no slots, no band and no α, and says so explicitly."""
    spec = condition_mod.parse({"saeLatentConditions": [valid_entry()]})[0]
    _edit, provenance = condition_mod.materialize(spec, loader=fake_loader())
    state_block = condition_mod.intervention_state(spec, provenance)
    assert state_block["slots"] == []
    assert state_block["alphaInNormUnits"] is None
    assert state_block["bandWidth"] is None
    assert state_block["interventionType"] == "saeLatent"
    assert state_block["saeLatent"]["beta"] == 10.0


# --------------------------------------------------------------------------
# Run-start preflight: validated, and refused where no loop can arm it
# --------------------------------------------------------------------------

def manifest_with(**overrides):
    from steerlab_server.experiment.manifest import Manifest

    raw = {"name": "study", "modelID": "google/gemma-3-27b-it"}
    raw.update(overrides)
    return Manifest.from_dict(raw)


def test_the_manifest_exposes_declared_latent_conditions():
    manifest = manifest_with(saeLatentConditions=[valid_entry()])
    assert [e["name"] for e in manifest.sae_latent_conditions] \
        == ["sympathy-clamp-10"]
    assert manifest_with().sae_latent_conditions == []


def test_an_invalid_latent_declaration_refuses_before_the_model_loads():
    from steerlab_server.experiment import tasks

    manifest = manifest_with(
        saeLatentConditions=[valid_entry(mode="scale")])
    with pytest.raises(RuntimeError, match="invalid SAE latent conditions"):
        tasks._sae_latent_preflight(manifest, lambda *_: None)


def test_a_multi_agent_study_refuses_a_latent_condition():
    """The panel loop runs configured/baseline per seat and has nowhere to arm
    a study-level residual-stream edit, so the arm would never execute while
    the run completed normally."""
    from steerlab_server.experiment import tasks

    manifest = manifest_with(studyKind="multiAgent",
                             saeLatentConditions=[valid_entry()])
    with pytest.raises(RuntimeError, match="multi-agent study"):
        tasks._sae_latent_preflight(manifest, lambda *_: None)


def test_a_manifest_without_latent_conditions_is_untouched():
    from steerlab_server.experiment import tasks

    assert tasks._sae_latent_preflight(manifest_with(), lambda *_: None) == []


def test_a_sweep_says_out_loud_that_it_does_not_cover_latent_conditions():
    """Advisory, not refusal: a sweep's matrix comes from CONCEPTS and its own
    layer×alpha grid, so it drops nothing declared — but a researcher must not
    assume the ladder they just swept covered a latent β."""
    from steerlab_server.experiment import tasks

    lines: list[str] = []
    tasks._advise_sweep_ignores_sae_latent(
        manifest_with(saeLatentConditions=[valid_entry()]), lines.append)
    assert any("NOT part of this sweep" in line for line in lines)
    lines.clear()
    tasks._advise_sweep_ignores_sae_latent(manifest_with(), lines.append)
    assert lines == []


def test_analyze_gives_latent_conditions_their_own_modality():
    """Not "injection": the edit is state-dependent and dosed in latent units,
    so pooling it with vector additions would invite comparing doses that are
    not comparable."""
    from steerlab_server.experiment import tasks

    manifest = manifest_with(
        conditions=[{"name": "fear-a4", "slots": [
            {"concept": "fear", "layer": 3, "alpha": 4.0}]}],
        saeLatentConditions=[valid_entry()])
    modalities = tasks._condition_modalities(manifest, None)
    assert modalities["baseline"] == "none"
    assert modalities["fear-a4"] == "injection"
    assert modalities["sympathy-clamp-10"] == "saeLatent"


def test_verify_surfaces_latent_condition_violations(tmp_path):
    """The declaration is checked by the firewall, before any behavior is
    measured — not first at run start."""
    manifest = manifest_with(
        concepts=[{"name": "c", "stimulusSetHash": "sha256:0"}],
        saeLatentConditions=[valid_entry(mode="scale")])
    violations = manifest.verify(root=str(tmp_path))
    assert any("unknown mode" in v for v in violations), violations


# --------------------------------------------------------------------------
# End-to-end run with a latent condition (fake model, offline)
# --------------------------------------------------------------------------

def _write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def latent_study_fixture(root, name, *, entries=None):
    """A four-item categorical study: implicit baseline + one concept
    condition + one SAE latent condition, answer-token + sampled instruments."""
    from steerlab_server.experiment import experiment_store as es

    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write_file(os.path.join(concept_dir, "positive.jsonl"),
                '{"text": "I feel dread"}\n')
    _write_file(os.path.join(concept_dir, "negative.jsonl"),
                '{"text": "calm morning"}\n')
    es.create(name, model_id="google/gemma-3-27b-it", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    raw["conditions"] = [{"name": "fear-a4", "slots": [
        {"concept": "fear", "layer": 2, "alpha": 4.0}]}]
    raw["saeLatentConditions"] = (entries if entries is not None
                                  else [valid_entry()])
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    lines = [json.dumps({"id": f"p{i}", "prompt": f"Decide case {i}.",
                         "options": ["affirm", "reverse"], "target": "reverse"})
             for i in range(4)]
    _write_file(prompts_path, "\n".join(lines) + "\n")
    return prompts_path


#: The two capability-battery probes the fake back-ends answer correctly. The
#: first is scored by the choice reader, the second by generated text — both
#: paths must be armed under a latent arm, so both are exercised.
BATTERY_CAPITAL = "What is the capital of France?"
BATTERY_SUM = "What is 17 + 26?"

BATTERY_V2 = "\n".join([
    json.dumps({"batteryFormat": 2, "scoring": "choiceProbability"}),
    json.dumps({"id": "cap-fr", "prompt": BATTERY_CAPITAL, "answer": "Paris",
                "options": ["Paris", "Rome"]}),
    json.dumps({"id": "sum", "prompt": BATTERY_SUM, "answer": "43",
                "scoring": "generatedText", "grading": "exact_number"}),
]) + "\n"

BATTERY_LEGACY = "\n".join([
    json.dumps({"prompt": BATTERY_CAPITAL, "answer": "Paris"}),
    json.dumps({"prompt": BATTERY_SUM, "answer": "43"}),
]) + "\n"


def _pin_battery(root, name, lines=BATTERY_V2):
    """Pin a capability battery into an existing draft study."""
    from steerlab_server.experiment import experiment_store as es

    rel = f"prompts/batteries/{name}.jsonl"
    _write_file(os.path.join(root, rel), lines)
    raw = es.load_raw(name, root)
    raw["capabilityBatteryFile"] = rel
    raw["capabilityBatteryHash"] = hashlib.sha256(
        lines.encode("utf-8")).hexdigest()
    es.save_raw(raw, root)


def _fake_bundle():
    from steerlab_server.experiment import tasks
    from steerlab_server.steering.vector_store import ConceptVectors

    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model_provider(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _patch_run(monkeypatch, log, counts=None):
    """Fake extraction, generation, scoring and the SAE LOADER SEAM — so the
    real materialize path (validation, layer derivation, provenance stamping)
    runs, with no network and no model.

    Both back-ends answer the capability-battery items correctly, so a battery
    reading is 1.0 under every arm: this rig proves WHAT WAS ARMED on each
    scoring pass, never whether a fake model is capable. ``counts`` is an
    optional ``[generations, scorings]`` pair the resume tests read to assert
    which forward passes a resumed run actually re-ran."""
    from steerlab_server.experiment import gemma_scope
    from steerlab_server.experiment import logprob as logprob_mod
    from steerlab_server.experiment import tasks

    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(gemma_scope, "load_sae_latent_feature",
                        fake_loader())

    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, latent_edits=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False,
                 token_ids_out=None):
        if counts is not None:
            counts[0] += 1
        log.append({"kind": "generate", "prompt": prompt,
                    "injections": list(injections or []),
                    "latentEdits": list(latent_edits or [])})
        return "43" if BATTERY_SUM in prompt else "I choose reverse."

    def score_options(model, prompt, options, *, model_id=None, injections=None,
                      latent_edits=None, prompt_mode=None, system_prompt=None,
                      qwen_thinking_enabled=False):
        from steerlab_server.experiment import logprob as lp
        if counts is not None:
            counts[1] += 1
        log.append({"kind": "instrument", "prompt": prompt,
                    "injections": list(injections or []),
                    "latentEdits": list(latent_edits or [])})
        target = "Paris" if "Paris" in options else "reverse"
        scores = [lp.OptionScore(option=o, token_ids=[7],
                                 token_logprobs=[-0.5 if o == target else -2.0])
                  for o in options]
        return lp.ChoiceResult(options=scores, prompt_token_count=5,
                               prompt_text=prompt)

    monkeypatch.setattr(tasks, "generate", generate)
    monkeypatch.setattr(logprob_mod, "score_options", score_options)


def _run_records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def test_a_latent_condition_runs_as_its_own_arm_with_stamped_provenance(
        tmp_path, monkeypatch):
    """The acceptance case: a declared latent condition executes through the
    SAME per-item executor as every other arm, producing a distinct condition
    in the matrix whose records say which mechanism ran."""
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "latent1")
    log: list[dict] = []
    _patch_run(monkeypatch, log)
    run_dir = tasks.run("latent1", prompts, root,
                        model_provider=_fake_model_provider,
                        log=lambda *_: None)
    records = _run_records(run_dir)
    conditions = {r.get("condition") for r in records}
    assert conditions == {"baseline", "fear-a4", "sympathy-clamp-10"}

    latent_records = [r for r in records
                      if r.get("condition") == "sympathy-clamp-10"]
    # Same measurement pipeline: 4 sampled + 4 instrument records, exactly as
    # baseline gets.
    assert len(latent_records) == 8
    baseline = [r for r in records if r.get("condition") == "baseline"]
    assert len(baseline) == len(latent_records)

    state = latent_records[0]["interventionState"]
    assert state["interventionType"] == "saeLatent"
    assert state["slots"] == [] and state["alphaInNormUnits"] is None
    assert state["saeLatent"]["mode"] == "clamp"
    assert state["saeLatent"]["beta"] == 10.0
    assert state["saeLatent"]["latentUnits"] is True
    assert state["saeLatent"]["feature"] == 12345
    assert state["saeLatent"]["layer"] == 31

    # A latent record carries the ordinary field set — it is not a poorer arm.
    sampled_latent = [r for r in latent_records if "instrument" not in r]
    sampled_base = [r for r in baseline if "instrument" not in r]
    assert set(sampled_latent[0]) == set(sampled_base[0])
    # ...and the ordinary arms are NOT polluted with latent provenance.
    assert "saeLatent" not in sampled_base[0]["interventionState"]


def test_the_run_stamps_the_pinned_sae_repository_commit_in_config_notes(
        tmp_path, monkeypatch):
    """config.json's top-level key set is CLOSED, so the provenance rides
    `notes`. A run that steers on a published dictionary must record which
    published bytes it read, or the arm cannot be reproduced from its record."""
    from steerlab_server.experiment import run_config, tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "latent2")
    _patch_run(monkeypatch, [])
    run_dir = tasks.run("latent2", prompts, root,
                        model_provider=_fake_model_provider,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    # The closed key set is untouched — the extras are inside notes.
    assert tuple(sorted(config)) == run_config.RUN_CONFIG_KEYS
    (stamp,) = config["notes"]["saeLatentConditions"]
    assert stamp["repository"] == "google/gemma-scope-2-27b-it"
    assert stamp["repositoryRevision"] == "a" * 40
    assert stamp["release"] == "gemma-scope-2-27b-it-res"
    assert stamp["saeID"] == "layer_31_width_16k_l0_medium"
    assert stamp["betaUnits"] == "latent"
    assert stamp["decoderRowHash"].startswith("sha256:")


def test_the_latent_edit_reaches_generation_and_the_answer_token_instrument(
        tmp_path, monkeypatch):
    """Both measurement paths of a latent condition are armed, and no other
    condition is: the mechanism travels with the condition, not the run."""
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "latent3")
    log: list[dict] = []
    _patch_run(monkeypatch, log)
    tasks.run("latent3", prompts, root, model_provider=_fake_model_provider,
              log=lambda *_: None)
    armed = [entry for entry in log if entry["latentEdits"]]
    # 4 items x (1 instrument + 1 sampled) for the one latent condition.
    assert len(armed) == 8
    assert {entry["kind"] for entry in armed} == {"generate", "instrument"}
    for entry in armed:
        # A latent condition injects no vectors: the mechanisms never mix here.
        assert entry["injections"] == []
        (edit,) = entry["latentEdits"]
        assert edit.mode == "clamp" and edit.beta == 10.0 and edit.layer == 31
    # Every OTHER call is unarmed, including the concept condition's.
    assert all(not entry["latentEdits"] for entry in log if entry not in armed)


def test_a_latent_condition_takes_its_place_in_the_shard_key_order(
        tmp_path, monkeypatch):
    """Sharding slices a contiguous range of the run's expected record keys, so
    the plan's condition order must match the executor's emission order."""
    from steerlab_server.experiment import sharding, tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "latent4")
    log: list[dict] = []
    _patch_run(monkeypatch, log)
    spec = sharding.ShardSpec(index=1, count=2)
    run_dir = tasks.run("latent4", prompts, root,
                        model_provider=_fake_model_provider,
                        log=lambda *_: None, shard=spec)
    records = _run_records(run_dir)
    # The shard emitted a contiguous slice; the latent arm is enumerated in it
    # (it is last in the matrix), and every emitted record belongs to a
    # condition this shard owns.
    assert records, "shard emitted no records"
    emitted = {r.get("condition") for r in records}
    assert emitted <= {"baseline", "fear-a4", "sympathy-clamp-10"}
    assert "sympathy-clamp-10" in emitted


def test_two_latent_conditions_are_two_arms(tmp_path, monkeypatch):
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "latent5", entries=[
        valid_entry(name="clamp-10", beta=10.0),
        valid_entry(name="clamp-2", beta=2.0)])
    _patch_run(monkeypatch, [])
    run_dir = tasks.run("latent5", prompts, root,
                        model_provider=_fake_model_provider,
                        log=lambda *_: None)
    records = _run_records(run_dir)
    betas = {r["condition"]: r["interventionState"]["saeLatent"]["beta"]
             for r in records if r.get("condition", "").startswith("clamp-")}
    assert betas == {"clamp-10": 10.0, "clamp-2": 2.0}


# --------------------------------------------------------------------------
# The capability battery covers the latent arm — under its intervention
# --------------------------------------------------------------------------

def _battery_rows(run_dir):
    with open(os.path.join(run_dir, "battery.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def _battery_lines(run_dir):
    with open(os.path.join(run_dir, "battery.jsonl"), encoding="utf-8") as h:
        return [line for line in h if line.strip()]


def _battery_accuracies(run_dir):
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as h:
        report = json.load(h)
    return {name: (block.get("capabilityBattery") or {}).get("accuracy")
            for name, block in report["conditions"].items()}


def _battery_calls(log):
    return [entry for entry in log
            if entry["prompt"] in (BATTERY_CAPITAL, BATTERY_SUM)]


def test_the_capability_battery_scores_the_latent_arm_under_its_intervention(
        tmp_path, monkeypatch):
    """Review finding 2. The battery is the study's capability GUARDRAIL, and
    a latent arm used to escape it entirely: the loop ran ordinary + variant
    conditions only, so the one mechanism with no sweep, no α ladder and no
    qualification was also the one arm with no capability control. It is now
    scored like every other arm — and armed, on BOTH scoring back-ends, with
    its own latent edit rather than with nothing."""
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "lbat1")
    _pin_battery(root, "lbat1")
    log: list[dict] = []
    _patch_run(monkeypatch, log)
    run_dir = tasks.run("lbat1", prompts, root,
                        model_provider=_fake_model_provider,
                        log=lambda *_: None)

    rows = _battery_rows(run_dir)
    assert {r["condition"] for r in rows} == {"baseline", "fear-a4",
                                              "sympathy-clamp-10"}
    latent_rows = [r for r in rows if r["condition"] == "sympathy-clamp-10"]
    assert {r["promptID"] for r in latent_rows} == {"cap-fr", "sum"}
    # Both scoring paths of the format-2 battery ran for the latent arm.
    assert {r["scoring"] for r in latent_rows} == {"choiceProbability",
                                                   "generatedText"}
    # ...and the report carries a real per-condition reading, not a null.
    assert _battery_accuracies(run_dir) == {"baseline": 1.0, "fear-a4": 1.0,
                                            "sympathy-clamp-10": 1.0}

    # The load-bearing assertion: the EDIT FIRED on the battery's own scoring
    # passes. Three conditions × two items = six battery calls; exactly the
    # latent arm's two are armed, one per back-end.
    calls = _battery_calls(log)
    assert len(calls) == 6
    armed = [entry for entry in calls if entry["latentEdits"]]
    assert len(armed) == 2
    assert {entry["kind"] for entry in armed} == {"instrument", "generate"}
    assert {entry["prompt"] for entry in armed} == {BATTERY_CAPITAL,
                                                    BATTERY_SUM}
    for entry in armed:
        # A latent arm injects no vectors: the mechanisms never mix.
        assert entry["injections"] == []
        (edit,) = entry["latentEdits"]
        assert (edit.mode, edit.beta, edit.layer) == ("clamp", 10.0, 31)


def test_the_latent_battery_row_carries_the_intervention_provenance(
        tmp_path, monkeypatch):
    """A battery.jsonl row must be self-describing about what was armed —
    otherwise "the latent arm passed the guardrail" is a claim that can only
    be checked by trusting the code that wrote it. The row carries the SAME
    interventionState block generations.jsonl carries, so one reader parses
    both files."""
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "lbat2")
    _pin_battery(root, "lbat2")
    _patch_run(monkeypatch, [])
    run_dir = tasks.run("lbat2", prompts, root,
                        model_provider=_fake_model_provider,
                        log=lambda *_: None)

    latent_rows = [r for r in _battery_rows(run_dir)
                   if r["condition"] == "sympathy-clamp-10"]
    generation = next(r for r in _run_records(run_dir)
                      if r.get("condition") == "sympathy-clamp-10")
    assert latent_rows
    for row in latent_rows:
        assert row["interventionState"] == generation["interventionState"]
        stamp = row["interventionState"]["saeLatent"]
        assert stamp["mode"] == "clamp" and stamp["beta"] == 10.0
        assert stamp["betaUnits"] == "latent" and stamp["latentUnits"] is True
        assert stamp["feature"] == 12345 and stamp["layer"] == 31
        assert stamp["repositoryRevision"] == "a" * 40


@pytest.mark.parametrize("lines", [BATTERY_V2, BATTERY_LEGACY])
def test_non_latent_battery_rows_are_unchanged_by_the_latent_plumbing(
        tmp_path, monkeypatch, lines):
    """Additive by construction: baseline and ordinary injection conditions
    produce byte-identical battery rows whether or not a latent arm is
    declared, in BOTH battery formats — so a legacy pinned hash keeps meaning
    exactly what it meant, and only the latent rows carry the new key."""
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    with_latent = latent_study_fixture(root, "lbat3")
    _pin_battery(root, "lbat3", lines)
    without = latent_study_fixture(root, "lbat4", entries=[])
    _pin_battery(root, "lbat4", lines)
    _patch_run(monkeypatch, [])
    a = tasks.run("lbat3", with_latent, root, model_provider=_fake_model_provider,
                  log=lambda *_: None)
    b = tasks.run("lbat4", without, root, model_provider=_fake_model_provider,
                  log=lambda *_: None)

    def _non_latent(run_dir):
        return [line for line in _battery_lines(run_dir)
                if json.loads(line)["condition"] != "sympathy-clamp-10"]

    assert _non_latent(a) == _non_latent(b)
    assert all("interventionState" not in json.loads(line)
               for line in _non_latent(a))
    # The run WITHOUT a latent arm is byte-identical to a pre-change run; the
    # run WITH one adds latent rows and nothing else.
    assert len(_battery_lines(a)) == len(_battery_lines(b)) + 2
    if lines is BATTERY_LEGACY:
        # The historical legacy key set, plus exactly the provenance key on
        # the (new) latent rows.
        legacy = {"condition", "promptIndex", "promptID", "sampleIndex",
                  "prompt", "answer", "output", "batteryHash", "correct"}
        rows = _battery_rows(a)
        for row in rows:
            expected = (legacy | {"interventionState"}
                        if row["condition"] == "sympathy-clamp-10" else legacy)
            assert set(row) == expected


def test_a_cancel_inside_the_latent_battery_resumes_without_rescoring(
        tmp_path, monkeypatch):
    """Resume/skip discipline holds for latent battery rows exactly as for
    every other condition's: the completed item's forward pass is SKIPPED, not
    merely deduplicated, and the resumed run ends with one row per
    (condition, item)."""
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    prompts = latent_study_fixture(root, "lbat5")
    _pin_battery(root, "lbat5")

    # 3 conditions × 4 items × (instrument + sampled) = 24 study calls, then
    # baseline's 2 battery items, then fear-a4's 2, then the latent arm's
    # first: stop there, mid-latent-battery.
    counts = [0, 0]
    _patch_run(monkeypatch, [], counts)
    cancelled_dir = tasks.run(
        "lbat5", prompts, root, model_provider=_fake_model_provider,
        log=lambda *_: None,
        should_cancel=lambda: counts[0] + counts[1] >= 29)
    assert counts[0] + counts[1] == 29
    rows = _battery_rows(cancelled_dir)
    assert [(r["condition"], r["promptID"]) for r in rows] == [
        ("baseline", "cap-fr"), ("baseline", "sum"),
        ("fear-a4", "cap-fr"), ("fear-a4", "sum"),
        ("sympathy-clamp-10", "cap-fr")]
    assert not os.path.exists(os.path.join(cancelled_dir, "report.json"))

    resumed = [0, 0]
    _patch_run(monkeypatch, [], resumed)
    resumed_dir = tasks.run("lbat5", prompts, root,
                            model_provider=_fake_model_provider,
                            log=lambda *_: None, run_directory=cancelled_dir)
    assert resumed_dir == cancelled_dir
    # Exactly the one missing latent battery item — the generatedText probe.
    assert tuple(resumed) == (1, 0)
    keys = [(r["condition"], r["promptID"]) for r in _battery_rows(resumed_dir)]
    assert len(keys) == len(set(keys)) == 6
    assert _battery_accuracies(resumed_dir)["sympathy-clamp-10"] == 1.0


# --------------------------------------------------------------------------
# The answer-token instrument really fires the edit on its scoring passes
# --------------------------------------------------------------------------

class ProbeHooked:
    """A hook manager with the real offset semantics of
    ``steering.hooks.HookedModel``, over a single decoder layer, so a test can
    watch WHERE the armed interventions fire during stepped scoring."""

    def __init__(self, hidden: int, layer: int):
        self.hidden = hidden
        self.layer = layer
        self.interventions: list = []
        self.fires: list[dict] = []
        self._offset = 0

    @contextmanager
    def session(self, interventions):
        self.interventions = list(interventions)
        self.reset_offsets()
        try:
            yield self
        finally:
            self.interventions = []

    def reset_offsets(self):
        self._offset = 0

    def forward(self, seq_len: int, activation: float):
        """Simulate one block: build the residual state this pass would carry,
        run every armed intervention, and record the induced delta."""
        h = torch.zeros((1, seq_len, self.hidden), dtype=torch.float32)
        h[0, :, 0] = activation
        offset = self._offset
        out = h
        for intervention in self.interventions:
            out = intervention.apply(out, self.layer, offset)
        self._offset += seq_len
        self.fires.append({"offset": offset, "seqLen": seq_len,
                           "delta": float((out - h).abs().sum().item())})


class ProbeCausalLM:
    """Drives :class:`ProbeHooked` once per forward, so the interventions see
    the same pass sequence the stepped scorer produces."""

    def __init__(self, hooked, vocab=16, activation=3.0):
        self.hooked = hooked
        self.vocab = vocab
        self.activation = activation

    def __call__(self, input_ids=None, attention_mask=None,
                 past_key_values=None, use_cache=True):
        seq = input_ids.shape[1]
        self.hooked.forward(seq, self.activation)
        logits = torch.arange(self.vocab, dtype=torch.float32).repeat(1, seq, 1)
        return SimpleNamespace(logits=logits, past_key_values="cache")


class ProbeTokenizer:
    def __init__(self):
        self.vocab: dict = {}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(tok, len(self.vocab) + 1)
               for tok in text.split()]
        return SimpleNamespace(input_ids=ids)


class ProbeSteeredModel:
    model_id = "google/gemma-3-27b-it"
    revision = "abc"
    context_window = 0
    device = "cpu"

    def __init__(self, *, hidden=4, layer=0, activation=3.0):
        self.hooked = ProbeHooked(hidden, layer)
        self.model = ProbeCausalLM(self.hooked, activation=activation)
        self.tokenizer = ProbeTokenizer()


def _score_with_latent(model, edit, **kwargs):
    from steerlab_server.experiment import logprob
    from steerlab_server.experiment.prompt_render import RAW_COMPLETION

    return logprob.score_options(
        model, "one two three", ["a", "b"], latent_edits=[edit],
        prompt_mode=RAW_COMPLETION, **kwargs)


def test_the_latent_edit_fires_on_the_prompt_end_of_every_scored_option():
    """Decode-identical semantics: the edit lands at the true prompt end, which
    is exactly the position injection steers before the first scored token.

    Two single-token options → two uncached passes over the 3-token prompt (the
    stepped driver re-prefills per option and resets the offset each time), and
    the edit fires on both."""
    model = ProbeSteeredModel()
    _score_with_latent(model, edit_at(mode=CLAMP, beta=5.0))
    fires = model.hooked.fires
    assert [f["seqLen"] for f in fires] == [3, 3]
    assert [f["offset"] for f in fires] == [0, 0]
    assert all(f["delta"] > 0 for f in fires), fires


def test_a_multi_token_option_steers_on_every_one_of_its_steps():
    model = ProbeSteeredModel()
    from steerlab_server.experiment import logprob
    from steerlab_server.experiment.prompt_render import RAW_COMPLETION

    logprob.score_options(model, "one two three", ["a", "x y z"],
                          latent_edits=[edit_at(mode=CLAMP, beta=5.0)],
                          prompt_mode=RAW_COMPLETION)
    fires = model.hooked.fires
    cached = [f for f in fires if f["seqLen"] == 1]
    assert len(cached) == 2, fires  # the 3-token option's 2 continuation steps
    assert all(f["delta"] > 0 for f in fires), fires


def test_the_instrument_honours_the_jumprelu_gate_rather_than_always_editing():
    """Proof the armed thing is the real mechanism and not a constant offset:
    an ADD below the gate leaves every scoring pass bit-identical."""
    model = ProbeSteeredModel(activation=0.2)
    _score_with_latent(model, edit_at(mode=ADD, beta=0.5))  # 0.2+0.5 <= θ=1.0
    assert model.hooked.fires
    assert all(f["delta"] == 0.0 for f in model.hooked.fires)
    # The same dormant state under CLAMP does fire — the difference is the mode.
    clamped = ProbeSteeredModel(activation=0.2)
    _score_with_latent(clamped, edit_at(mode=CLAMP, beta=5.0))
    assert all(f["delta"] > 0 for f in clamped.hooked.fires)


def test_the_latent_intervention_leads_the_instruments_chain():
    """Same ordering contract as generation: the state-dependent edit reads the
    residual the model would carry, before any pure offset."""
    from steerlab_server.experiment import logprob
    from steerlab_server.experiment.generate import CellInjection
    from steerlab_server.experiment.prompt_render import RAW_COMPLETION
    from steerlab_server.steering.injector import VectorInjector
    from steerlab_server.steering.sae_latent import SAELatentIntervention

    seen: list = []
    model = ProbeSteeredModel()
    original = model.hooked.session

    @contextmanager
    def capture(interventions):
        seen.extend(interventions)
        with original(interventions):
            yield model.hooked

    model.hooked.session = capture
    logprob.score_options(
        model, "one two three", ["a", "b"],
        injections=[CellInjection(layer=0, vector=[1.0, 0.0, 0.0, 0.0],
                                  alpha=2.0)],
        latent_edits=[edit_at(mode=CLAMP, beta=5.0)],
        prompt_mode=RAW_COMPLETION)
    assert [type(item) for item in seen] == [SAELatentIntervention,
                                             VectorInjector]
    assert seen[0]._prompt_token_count == 3
    assert seen[1]._prompt_token_count == 3
