"""The final-norm gain convention is OBSERVED, and a lens may be imported for
any model that has one (2026-09-05).

Two facts this file pins, both read off the installed transformers source
rather than a family name: Gemma-3 and Qwen3.5 fold ``1 + weight`` (offset),
Llama and Qwen3 fold ``weight`` (direct), Gemma-3n is direct despite the
name, and GPT-2's LayerNorm cannot be folded at all. Reading ``1 + w`` off a
direct model would shift every gain entry by one — a REORDERING of the
vocabulary, the same defect shape as the dropped gain the review found — so
every path that folds the gain observes the convention first and stamps it.

The second half covers the tier plumbing that makes an uncurated lens usable:
declared at import, carried by the record, honoured by freeze.
"""

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import experiment_store
from steerlab_server.jlens import (backend, importer, lens_store,
                                   norm_convention, qualification, schemas)
from steerlab_server.jlens.readout import LensReadout, ReadoutConfig


# ----------------------------------------------------------------- fixtures

class _RMSNorm(torch.nn.Module):
    """A final norm in either parameterization, so a test can say which fold
    the module applies and check that the observer names it."""

    def __init__(self, weight, *, offset: bool, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.variance_epsilon = eps
        self._offset = offset

    def forward(self, x):
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        gain = (1.0 + self.weight.float()) if self._offset else self.weight.float()
        return (normed * gain).type_as(x)


class _LayerNormWithBias(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(d), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), self.weight,
                                              self.bias, 1e-5)


class _CenteringNorm(torch.nn.Module):
    """Gain-only, but mean-centering: an RMSNorm fold cannot reproduce it."""

    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d) * 1.5,
                                         requires_grad=False)

    def forward(self, x):
        centered = x - x.mean(-1, keepdim=True)
        return centered * torch.rsqrt(centered.pow(2).mean(-1, keepdim=True)
                                      + 1e-6) * self.weight


def _weights(d=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(d, generator=generator) * 4.0 + 0.25


# --------------------------------------------------- observe(): live modules

def test_the_observer_names_the_fold_a_module_actually_computes():
    w = _weights()
    offset = norm_convention.observe(_RMSNorm(w, offset=True))
    direct = norm_convention.observe(_RMSNorm(w, offset=False))
    assert offset["convention"] == "offset"
    assert direct["convention"] == "direct"
    assert offset["eps"] == direct["eps"] == 1e-6
    assert offset["agreement"] < 1e-5 and direct["agreement"] < 1e-5
    assert offset["className"] == "_RMSNorm"


def test_the_observer_reads_a_bf16_module_through_a_float32_copy():
    """A bf16 runtime's rounding must not blur the comparison, and the live
    module is never touched."""
    module = _RMSNorm(_weights(), offset=True).to(torch.bfloat16)
    observed = norm_convention.observe(module)
    assert observed["convention"] == "offset"
    assert observed["agreement"] < 1e-5
    assert module.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("module, reason", [
    (_LayerNormWithBias(8), "carries a bias"),
    (_CenteringNorm(8), "matches neither"),
])
def test_a_norm_the_fold_cannot_reproduce_is_refused_by_name(module, reason):
    with pytest.raises(schemas.JLensError, match=reason):
        norm_convention.observe(module)


def test_a_bare_object_with_only_a_weight_cannot_be_observed():
    with pytest.raises(schemas.JLensError, match="not a torch module"):
        norm_convention.observe(SimpleNamespace(weight=torch.ones(8)))


def test_gain_from_weight_follows_the_convention():
    w = torch.tensor([0.0, 1.0, -0.5])
    assert torch.equal(norm_convention.gain_from_weight(w, "offset"),
                       torch.tensor([1.0, 2.0, 0.5]))
    assert torch.equal(norm_convention.gain_from_weight(w, "direct"), w)
    with pytest.raises(schemas.JLensError, match="unknown"):
        norm_convention.gain_from_weight(w, "gemma")


# ------------------------------------------ from_config(): checkpoint paths

def _tiny_config(model_type, d=8, layers=1, vocab=16):
    return {"model_type": model_type, "hidden_size": d,
            "num_hidden_layers": layers, "vocab_size": vocab,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": d, "intermediate_size": 2 * d}


def _snapshot_with(tmp_path, config: dict, name="snap"):
    snap = tmp_path / name
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(json.dumps(config))
    return str(snap)


@pytest.mark.parametrize("model_type, expected, cls", [
    ("gemma3_text", "offset", "Gemma3RMSNorm"),
    ("llama", "direct", "LlamaRMSNorm"),
    ("qwen3", "direct", "Qwen3RMSNorm"),
])
def test_the_convention_is_observed_from_the_architecture_without_weights(
        tmp_path, model_type, expected, cls):
    """The installed library's own norm class is instantiated at the model's
    width and run; no weight file is read, and no name rule is consulted."""
    observed = norm_convention.from_config(
        _snapshot_with(tmp_path, _tiny_config(model_type)))
    assert observed["convention"] == expected
    assert observed["className"] == cls
    assert observed["modelType"] == model_type


def test_a_layernorm_architecture_is_refused_on_the_checkpoint_path(tmp_path):
    config = {"model_type": "gpt2", "n_embd": 8, "n_layer": 1, "vocab_size": 16,
              "n_head": 1, "n_positions": 16}
    with pytest.raises(schemas.JLensError, match="could not locate a final norm"):
        norm_convention.from_config(_snapshot_with(tmp_path, config))


def test_a_snapshot_without_a_config_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(schemas.JLensError, match="no config.json"):
        norm_convention.from_config(str(tmp_path / "empty"))


def test_derive_reads_the_direct_gain_off_a_llama_snapshot_and_stamps_it(
        tmp_path, monkeypatch):
    """The checkpoint-only path: ``g = w`` on a direct architecture, ``1 + w``
    on an offset one, and the direction says which it folded."""
    from safetensors.torch import save_file

    from steerlab_server.jlens import derive

    d, vocab = 8, 16
    embed = torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d) / 100
    norm = torch.arange(d, dtype=torch.float32) / 10 + 0.5
    for name, model_type in (("llama", "llama"), ("gemma", "gemma3_text")):
        snap = tmp_path / name
        snap.mkdir()
        save_file({"model.embed_tokens.weight": embed,
                   "model.norm.weight": norm}, str(snap / "model.safetensors"))
        (snap / "config.json").write_text(json.dumps(_tiny_config(model_type)))

    monkeypatch.setattr(derive, "_snapshot_dir",
                        lambda model_id, revision=None: str(tmp_path / model_id))
    u_llama, g_llama, conv_llama = derive.read_token_row_gain_and_convention(
        "llama", 3)
    u_gemma, g_gemma, conv_gemma = derive.read_token_row_gain_and_convention(
        "gemma", 3)
    assert torch.equal(u_llama, embed[3]) and torch.equal(u_gemma, embed[3])
    assert torch.allclose(g_llama, norm)
    assert torch.allclose(g_gemma, 1.0 + norm)
    assert conv_llama["convention"] == "direct"
    assert conv_gemma["convention"] == "offset"


# ------------------------------------------- LensReadout.build(): observed

class _Trunk(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm


class _Host(torch.nn.Module):
    def __init__(self, *, head_weight, norm):
        super().__init__()
        vocab, d = head_weight.shape
        self.model = _Trunk(norm)
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)
        self.lm_head.weight.requires_grad_(False)
        with torch.no_grad():
            self.lm_head.weight.copy_(head_weight)
        self.config = SimpleNamespace(final_logit_softcapping=None,
                                      vocab_size=vocab)

    @property
    def device(self):
        return torch.device("cpu")


class _FixedSource:
    def __init__(self, jacobians):
        self._j = {int(k): v.to(torch.float32) for k, v in jacobians.items()}
        self.d_model = next(iter(self._j.values())).shape[0]
        self.n_prompts = 3

    @property
    def source_layers(self):
        return sorted(self._j)

    def jacobian(self, layer):
        return self._j[layer]


def _lens(tmp_path, d=8, lens_id="conv-lens"):
    from steerlab_server.experiment import paths

    root = str(tmp_path / "ws")
    source = _FixedSource({0: torch.eye(d)})
    directory = paths.jlens_lens_directory(lens_id, root)
    os.makedirs(directory, exist_ok=True)
    converted = importer.convert_to_per_layer(source, directory)
    record = schemas.JLensRecord(
        lensID=lens_id,
        source=schemas.SourceRef(repo="t", folder="f", tensorFile="t.pt",
                                 configFile="c.yaml", tensorSHA256="dead"),
        fit=schemas.FitProvenance(modelID="test/tiny", dtype="float32"),
        sourceLayers=[0], dModel=d, targetLayer=1, nPrompts=3,
        converted=converted)
    lens_store.save(record, root)
    return record, root


def _build(tmp_path, norm, *, head_weight):
    record, root = _lens(tmp_path, d=head_weight.shape[1])
    inner = _Host(head_weight=head_weight, norm=norm).eval()
    model = SimpleNamespace(model=inner, tokenizer=None, model_id="test/tiny")
    config = ReadoutConfig(layers=[0], watchlist=[1, 4], topK=2)
    return LensReadout.build(record=record, config=config, model=model,
                             root=root)


def test_the_readout_folds_whichever_gain_the_norm_is_observed_to_apply(
        tmp_path):
    """Same weight, two parameterizations, two different gains — and both
    vocabulary paths agree with the model's own norm-then-head arithmetic."""
    torch.manual_seed(7)
    d, vocab = 8, 16
    head = torch.randn(vocab, d) * 0.5
    w = _weights(d)
    h = torch.randn(d) * 3.0
    for offset in (True, False):
        norm = _RMSNorm(w, offset=offset)
        readout = _build(tmp_path / ("o" if offset else "d"), norm,
                         head_weight=head)
        assert readout.gain_convention == ("offset" if offset else "direct")
        expected_gain = (1.0 + w) if offset else w
        assert torch.allclose(readout.gain, expected_gain)
        # The model's own arithmetic is the ground truth for both paths.
        truth = head @ norm(h)
        watched = readout.watched_scores(h, 0)
        full = readout.logits(h, 0)
        assert torch.allclose(watched, truth[[1, 4]], atol=1e-4)
        assert torch.allclose(full, truth, atol=1e-4)


def test_the_readout_refuses_a_norm_it_cannot_fold(tmp_path):
    head = torch.randn(16, 8)
    with pytest.raises(schemas.JLensError, match="carries a bias"):
        _build(tmp_path, _LayerNormWithBias(8), head_weight=head)


def test_the_readout_uses_the_norms_own_epsilon(tmp_path):
    head = torch.randn(16, 8)
    readout = _build(tmp_path, _RMSNorm(_weights(), offset=False, eps=1e-3),
                     head_weight=head)
    assert readout.eps == 1e-3


# ---------------------------------------------------- tiers: declared lenses

def _published_lens(tmp_path, model_id="Qwen/Qwen3-14B", *, tier,
                    folder="qwen3-14b/jlens/Salesforce-wikitext",
                    tensor="Qwen3-14B_jacobian_lens.pt"):
    snap = tmp_path / "snap"
    dest = snap / folder
    dest.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=[0, 1, 2]).save_checkpoint(
        str(dest / tensor))
    (dest / "config.yaml").write_text(f"hf_model_name: {model_id}\n")
    root = str(tmp_path / "ws")
    record = importer.import_lens(model_id, root=root, snapshot=str(snap),
                                  tier=tier)
    return record, root


REV = "005ad3404e59d6023443cb575daa05336842228a"


def _qualified(record, root, *, model_id, passed=True):
    record.qualifications.append(schemas.Qualification(
        qualificationID="q1", modelID=model_id, revision=REV, dtype="bfloat16",
        tier=record.tier or "unknown", passed=passed,
        lensSHA256=record.source.tensorSHA256,
        convertedSHA256=record.converted.sha256, layers=[0, 1, 2]))
    lens_store.save(record, root)
    return record


def _manifest(record, model_id):
    return {"name": "s", "modelID": model_id, "modelRevision": REV,
            "dtype": "bfloat16", "concepts": [],
            "jlensReadout": {"lensID": record.lensID,
                             "lensSHA256": record.source.tensorSHA256,
                             "layers": [0, 1], "watchlist": [5], "topK": 0,
                             "configHash": "c" * 64, "qualificationID": "q1",
                             "tokenizerHash": "t" * 64}}


def test_a_declared_evidence_lens_on_an_uncurated_model_can_be_frozen(tmp_path):
    record, root = _published_lens(tmp_path, tier="evidence")
    _qualified(record, root, model_id="Qwen/Qwen3-14B")
    experiment_store._check_jlens_readout("s", _manifest(record, "Qwen/Qwen3-14B"),
                                          root)


def test_a_declared_testing_lens_is_refused_at_freeze_and_says_so(tmp_path):
    record, root = _published_lens(tmp_path, tier="testing")
    _qualified(record, root, model_id="Qwen/Qwen3-14B")
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="testing-tier.*declared"):
        experiment_store._check_jlens_readout(
            "s", _manifest(record, "Qwen/Qwen3-14B"), root)


def test_a_lens_with_no_tier_at_all_is_refused_at_freeze_with_the_repair(tmp_path):
    """A record from before declarations existed, on a model the table has no
    row for: nothing says what this study treats it as, so freeze refuses and
    names the re-import that would say it."""
    record, root = _published_lens(tmp_path, tier="evidence")
    record.tier, record.tierSource = None, None
    _qualified(record, root, model_id="Qwen/Qwen3-14B")
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no evidence tier.*--tier"):
        experiment_store._check_jlens_readout(
            "s", _manifest(record, "Qwen/Qwen3-14B"), root)


def test_qualify_stamps_the_declared_tier_and_a_curated_row_still_wins(tmp_path):
    record, _ = _published_lens(tmp_path, tier="evidence")
    assert qualification.tier_for("Qwen/Qwen3-14B", record) == "evidence"
    assert qualification.tier_for("Qwen/Qwen3-14B") == "unknown"
    assert qualification.tier_for("google/gemma-3-4b-it", record) == "testing"
