"""The RMSNorm gain, on BOTH vocabulary paths of the canonical J-lens readout.

The canonical readout is ``softcap(U · (g ⊙ RMSNorm(J_l h)))`` with Gemma's
``g = 1 + norm.weight``. It has two implementations inside
:class:`~steerlab_server.jlens.readout.LensReadout` — a watchlist slice that
folds ``g`` into the token rows once at build time, and a full-vocabulary
projection through the model's own ``lm_head`` — and an external review
(2026-09-05) found the second one dropping ``g`` entirely. Because ``g`` varies
per coordinate that is not a scale error: it reorders tokens, so every top-k
table, emergent-token table and full-vocabulary rank derived from it read a
distribution the model does not compute.

These tests hold the two paths to ONE number. They use the production
``LensReadout`` built through the production store and importer — a fake that
re-derived the formula would only agree with whichever version of it the fake
copied. The last test drops the formula altogether and asks the model's own
final-norm module and output head instead.
"""

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import paths
from steerlab_server.jlens import importer, lens_store, schemas
from steerlab_server.jlens.readout import LensReadout, ReadoutConfig


# ------------------------------------------------------------------ fixtures

class _FixedSource:
    """A lens whose ``J_l`` matrices are handed in, so a test can say exactly
    what transport it wants (identity for the arithmetic fixture, dense for the
    claims that are about transport doing work)."""

    def __init__(self, jacobians: dict, *, n_prompts: int = 3):
        self._j = {int(k): v.to(torch.float32) for k, v in jacobians.items()}
        self._d_model = next(iter(self._j.values())).shape[0]

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_prompts(self) -> int:
        return self._n_prompts if hasattr(self, "_n_prompts") else 3

    @property
    def source_layers(self) -> list[int]:
        return sorted(self._j)

    def jacobian(self, layer: int):
        return self._j[layer]


def _write_lens(tmp_path, jacobians, *, lens_id="gain-lens",
                model_id="test/tiny"):
    """A record in a real workspace lens library, through the real converter."""
    root = str(tmp_path / "ws")
    source = _FixedSource(jacobians)
    directory = paths.jlens_lens_directory(lens_id, root)
    os.makedirs(directory, exist_ok=True)
    converted = importer.convert_to_per_layer(source, directory)
    record = schemas.JLensRecord(
        lensID=lens_id,
        source=schemas.SourceRef(repo="test/lens", folder="f",
                                 tensorFile="t.pt", configFile="c.yaml",
                                 tensorSHA256="deadbeef"),
        fit=schemas.FitProvenance(modelID=model_id, dtype="float32"),
        sourceLayers=source.source_layers, dModel=source.d_model,
        targetLayer=source.source_layers[-1] + 1, nPrompts=source.n_prompts,
        converted=converted)
    lens_store.save(record, root)
    return record, root


class _Trunk(torch.nn.Module):
    """Stands where the decoder stack sits: ``LensReadout`` reaches the final
    norm through ``model.model.norm``, which is where HF puts it."""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm


class _GainNorm(torch.nn.Module):
    """Gemma's final norm: ``x·rsqrt(mean(x²)+eps) · (1 + weight)``. The WEIGHT
    is stored, the GAIN is ``1 + weight`` — the off-by-one this readout folds."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = 1e-6

    def forward(self, x):
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class _Host(torch.nn.Module):
    """The minimum an inner model must present to be read out from."""

    def __init__(self, *, head_weight, norm_weight, softcap=None):
        super().__init__()
        vocab, d_model = head_weight.shape
        self.model = _Trunk(_GainNorm(norm_weight))
        self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight.requires_grad_(False)
        with torch.no_grad():
            self.lm_head.weight.copy_(head_weight)
        self.config = SimpleNamespace(final_logit_softcapping=softcap,
                                      vocab_size=vocab)

    @property
    def device(self):
        return torch.device("cpu")


def _build(tmp_path, *, jacobians, head_weight, norm_weight, watchlist,
           softcap=None, lens_id="gain-lens"):
    """``(readout, built)`` where ``built`` carries the record, root and host —
    the qualification tests need all three, the readout tests only the first."""
    record, root = _write_lens(tmp_path, jacobians, lens_id=lens_id)
    inner = _Host(head_weight=head_weight, norm_weight=norm_weight,
                  softcap=softcap).eval()
    model = SimpleNamespace(model=inner, tokenizer=None, model_id="test/tiny")
    config = ReadoutConfig(layers=sorted(jacobians), watchlist=list(watchlist),
                           topK=min(len(watchlist), head_weight.shape[0]))
    readout = LensReadout.build(record=record, config=config, model=model,
                                root=root)
    return readout, SimpleNamespace(record=record, root=root, host=model,
                                    inner=inner)


# --------------------------------------------- (a) the reviewer's fixture

def test_the_full_vocabulary_path_applies_the_norm_gain(tmp_path):
    """The regression, in the smallest arithmetic that shows it.

    ``J = I``, an identity two-token head, ``g = [10, 1]``, ``h = [1, 2]``.
    ``RMSNorm(h) = [0.632455, 1.264911]``, so the canonical readout is
    ``[6.324554, 1.264911]`` and token 0 wins. A full-vocabulary path that
    skips ``g`` returns ``[0.632455, 1.264911]`` and token 1 wins: the defect
    is a REORDERING, which is why a scale-tolerant comparison would miss it.
    """
    readout, _built = _build(
        tmp_path,
        jacobians={0: torch.eye(2)},
        head_weight=torch.eye(2),
        # gain = 1 + weight, so weight [9, 0] is the reviewer's g = [10, 1].
        norm_weight=torch.tensor([9.0, 0.0]),
        watchlist=[0, 1])
    h = torch.tensor([1.0, 2.0])

    watched = readout.watched_scores(h, 0)
    full = readout.logits(h, 0)

    canonical = torch.tensor([6.324554, 1.264911])
    assert torch.allclose(watched, canonical, atol=1e-5), watched
    assert torch.allclose(full, canonical, atol=1e-5), full
    assert int(full.argmax()) == 0
    assert int(watched.argmax()) == int(full.argmax())


# ------------------------------------- (b) the two paths, over the matrix

@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("use_jacobian", [True, False])
def test_watchlist_and_full_vocabulary_agree(tmp_path, softcap, use_jacobian):
    """A dense (non-diagonal) Jacobian, a nonuniform gain, softcap on and off,
    and the logit-lens companion (``use_jacobian=False``).

    Tolerance: both sides are float32 throughout and compute the same triple
    product ``z_d · g_d · U_td`` with the multiplication ASSOCIATED
    differently — the watchlist folds ``g`` into ``U`` at build time, the full
    path scales ``z`` — so they may differ only by float32 reassociation.
    1e-4 absolute on logits of order 10 is ~1e-5 relative, far below that and
    far above the ~1e-6 the reassociation can produce.
    """
    torch.manual_seed(20260905)
    d_model, vocab = 8, 24
    generator = torch.Generator().manual_seed(4242)
    jacobians = {1: torch.randn(d_model, d_model, generator=generator) * 0.3}
    head = torch.randn(vocab, d_model, generator=generator) * 0.5
    # Nonuniform on purpose: a constant gain would make the defect a pure
    # rescale and every ranking claim here vacuous.
    norm_weight = torch.randn(d_model, generator=generator) * 0.8
    watchlist = [0, 3, 7, 11, 19, 23]

    readout, _built = _build(tmp_path, jacobians=jacobians, head_weight=head,
                             norm_weight=norm_weight, watchlist=watchlist,
                             softcap=softcap)
    hidden = torch.randn(d_model, generator=generator) * 12.0

    watched = readout.watched_scores(hidden, 1, use_jacobian=use_jacobian)
    full = readout.logits(hidden, 1, use_jacobian=use_jacobian)

    assert torch.allclose(watched, full[watchlist], atol=1e-4), (
        watched, full[watchlist])


def test_full_vocabulary_ranks_match_the_watchlist_ordering(tmp_path):
    """``ranks_of``/``topk`` are the consumers that the defect actually
    corrupts, so the ordering — not only the values — is asserted."""
    generator = torch.Generator().manual_seed(99)
    d_model, vocab = 8, 24
    jacobians = {0: torch.randn(d_model, d_model, generator=generator) * 0.3}
    head = torch.randn(vocab, d_model, generator=generator) * 0.5
    norm_weight = torch.randn(d_model, generator=generator) * 0.8
    watchlist = list(range(vocab))

    readout, _built = _build(tmp_path, jacobians=jacobians, head_weight=head,
                             norm_weight=norm_weight, watchlist=watchlist)
    hidden = torch.randn(d_model, generator=generator) * 12.0

    watched = readout.watched_scores(hidden, 0)
    full = readout.logits(hidden, 0)
    assert torch.equal(torch.argsort(watched, descending=True),
                       torch.argsort(full, descending=True))
    ids, _values = readout.topk(hidden, 0, 5)
    assert ids.tolist() == torch.argsort(watched,
                                         descending=True)[:5].tolist()


# ----------------------------------------------- (c) the independent oracle

@pytest.mark.parametrize("family", ["gemma", "gemma2"])
def test_both_paths_match_the_models_own_norm_and_head(tmp_path, family):
    """The oracle is the MODEL, not the formula.

    A tiny randomly-initialized Gemma/Gemma2 is built from a config (no
    download), and the readout is compared against that model's actual final
    ``norm`` module followed by its actual ``lm_head`` on the same transported
    residual. Nothing in the expectation restates the readout's arithmetic, so
    a shared misreading of the formula cannot make this pass.

    The norm weight is filled with a seeded draw rather than left at the
    ``zeros`` a fresh init gives it: at zero weight the gain is exactly 1 and
    the defect under test is invisible.
    """
    transformers = pytest.importorskip("transformers")
    if family == "gemma":
        config = transformers.GemmaConfig(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, intermediate_size=32, vocab_size=32,
            max_position_embeddings=64, head_dim=8)
        lm = transformers.GemmaForCausalLM(config).eval()
    else:
        config = transformers.Gemma2Config(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, intermediate_size=32, vocab_size=32,
            max_position_embeddings=64, head_dim=8,
            final_logit_softcapping=20.0)
        lm = transformers.Gemma2ForCausalLM(config).eval()
    torch.manual_seed(20260905)
    with torch.no_grad():
        lm.model.norm.weight.copy_(torch.randn(16) * 0.7)

    generator = torch.Generator().manual_seed(11)
    jacobians = {0: torch.randn(16, 16, generator=generator) * 0.3}
    watchlist = [1, 5, 17, 31]
    record, root = _write_lens(tmp_path, jacobians, lens_id=f"oracle-{family}")
    model = SimpleNamespace(model=lm, tokenizer=None, model_id="test/tiny")
    config_obj = ReadoutConfig(layers=[0], watchlist=watchlist, topK=4)
    readout = LensReadout.build(record=record, config=config_obj, model=model,
                                root=root)

    hidden = torch.randn(16, generator=generator) * 12.0
    with torch.no_grad():
        z = hidden @ jacobians[0].T                     # the transport only
        oracle = lm.lm_head(lm.model.norm(z)).to(torch.float32)
        softcap = getattr(lm.config, "final_logit_softcapping", None)
        if softcap:
            oracle = softcap * torch.tanh(oracle / softcap)

    full = readout.logits(hidden, 0)
    watched = readout.watched_scores(hidden, 0)
    assert torch.allclose(full, oracle, atol=1e-3), (full, oracle)
    assert torch.allclose(watched, oracle[watchlist], atol=1e-3), (
        watched, oracle[watchlist])


# --------------------- the gate: qualification must SEE either path drop g

REF_D_MODEL = 8
REF_VOCAB = 16
REF_WATCHLIST = [1, 4, 9, 15]


def _rms(z):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)


class _ReferenceModel:
    """The reference's wrapper as ``_check_reference_agreement`` uses it: an
    output head plus the final-norm gain, exposed through ``unembed``. Held in
    float32 here so the only thing a deviation can be about is the FORMULA —
    the pinned reference's dtype-cast asymmetry has its own tests in
    ``test_jlens_qualification.py`` and is not the subject."""

    def __init__(self, head, gain):
        self._lm_head = torch.nn.Linear(head.shape[1], head.shape[0],
                                        bias=False)
        self._lm_head.weight.requires_grad_(False)
        with torch.no_grad():
            self._lm_head.weight.copy_(head)
        self._gain = gain

    def unembed(self, z):
        return self._lm_head(_rms(z) * self._gain).to(torch.float32)


class _ReferenceLens:
    def __init__(self, *, jacobians, n_prompts, d_model):
        self._j = jacobians

    def transport(self, h, layer):
        return h @ self._j[layer].T.to(h.dtype)


def _reference_fixture(tmp_path, monkeypatch, *, lens_id):
    """A REAL ``LensReadout`` with the full-vocabulary path armed, plus a fake
    ``jlens`` standing in for the absent extra and holding the same head and
    gain. Returns ``(readout, built)``."""
    import sys
    import types

    from steerlab_server.jlens import backend as backend_mod

    generator = torch.Generator().manual_seed(20260905)
    jacobians = {0: torch.randn(REF_D_MODEL, REF_D_MODEL,
                                generator=generator) * 0.3}
    head = torch.randn(REF_VOCAB, REF_D_MODEL, generator=generator) * 0.5
    norm_weight = torch.randn(REF_D_MODEL, generator=generator) * 0.8

    readout, built = _build(tmp_path, jacobians=jacobians, head_weight=head,
                            norm_weight=norm_weight, watchlist=REF_WATCHLIST,
                            lens_id=lens_id)
    ref_model = _ReferenceModel(head, 1.0 + norm_weight)
    module = types.ModuleType("jlens")
    module.from_hf = lambda inner, tokenizer, force_bos=False: ref_model
    module.JacobianLens = _ReferenceLens
    monkeypatch.setitem(sys.modules, "jlens", module)
    monkeypatch.setattr(backend_mod, "require_reference", lambda: module)
    return readout, built


def _run_check(readout, built):
    from steerlab_server.jlens import qualification

    return qualification._check_reference_agreement(
        built.record, built.host, readout, [0], root=built.root)


def test_reference_agreement_covers_the_full_vocabulary_path_too(tmp_path,
                                                                 monkeypatch):
    """Baseline: with top-k armed, BOTH implementations are held to the
    reference and the record says how many comparisons each contributed."""
    readout, built = _reference_fixture(tmp_path, monkeypatch, lens_id="ref-ok")
    result = _run_check(readout, built)

    assert result.passed, result.detail
    assert result.measured["fullVocabArmed"] is True
    assert result.measured["fullVocabComparisons"] > 0
    assert result.measured["watchlistComparisons"] == \
        result.measured["fullVocabComparisons"]
    assert {r["path"] for r in result.measured["perComparison"]} == \
        {"watchlist", "fullVocab"}
    assert "both vocabulary paths" in result.detail


def test_a_full_vocabulary_path_that_drops_the_gain_is_FLAGGED(tmp_path,
                                                               monkeypatch):
    """The defect this whole file exists for, put back deliberately.

    Neutering ``readout.gain`` leaves the watchlist path untouched — it folded
    ``g`` into its rows at build time — and removes it from the full path only.
    Before the check grew its second leg this ran green, which is how the
    defect survived: the study's evidence said "agrees with the reference"
    about a path the reference had never been shown.
    """
    readout, built = _reference_fixture(tmp_path, monkeypatch,
                                        lens_id="ref-nogain-full")
    assert _run_check(readout, built).passed          # it agreed a moment ago
    readout.gain = torch.ones_like(readout.gain)

    result = _run_check(readout, built)

    assert not result.passed
    assert "exceeds the pinned tolerance" in result.detail
    worst = result.measured["worstComparison"]
    assert worst["path"] == "fullVocab"
    # And the watchlist leg is still clean, so the record names the right one.
    watchlist_worst = max(r["absDeviation"]
                          for r in result.measured["perComparison"]
                          if r["path"] == "watchlist")
    from steerlab_server.jlens import qualification
    assert watchlist_worst < qualification.REFERENCE_TOLERANCE


def test_a_watchlist_path_that_drops_the_gain_is_STILL_flagged(tmp_path,
                                                               monkeypatch):
    """The other direction, so the new leg is an addition and not a swap."""
    from steerlab_server.jlens import qualification

    readout, built = _reference_fixture(tmp_path, monkeypatch,
                                        lens_id="ref-nogain-watch")
    readout.watched_rows = readout.watched_rows / readout.gain

    result = _run_check(readout, built)

    assert not result.passed
    assert result.measured["worstComparison"]["path"] == "watchlist"
    full_worst = max(r["absDeviation"]
                     for r in result.measured["perComparison"]
                     if r["path"] == "fullVocab")
    assert full_worst < qualification.REFERENCE_TOLERANCE


def test_a_watchlist_only_readout_says_the_full_path_was_NOT_checked(
        tmp_path, monkeypatch):
    """"Not armed" and "checked and agreed" must never read the same. A
    watchlist-only study never calls ``logits()``, so there is nothing to
    check — but the record has to say that rather than imply coverage."""
    readout, built = _reference_fixture(tmp_path, monkeypatch,
                                        lens_id="ref-watch-only")
    readout.lm_head = None                    # what a topK=0 build produces

    result = _run_check(readout, built)

    assert result.passed
    assert result.measured["fullVocabArmed"] is False
    assert result.measured["fullVocabComparisons"] == 0
    assert "not armed" in result.detail
