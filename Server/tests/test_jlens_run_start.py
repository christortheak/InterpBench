"""A declared jlensReadout is armed at run start, or the run refuses.

The gap this closes: _execute_condition accepted a session and nothing built
one, so a manifest could declare a readout, pass every freeze gate, run, and
produce no trace — silently. Nothing downstream could tell that apart from a
study that never asked for one.
"""

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.jlens import backend, importer, lens_store

MODEL = "google/gemma-3-4b-it"


class _FakeNorm:
    def __init__(self, d):
        import torch as t
        self.weight = t.zeros(d)


class _FakeHead:
    def __init__(self, d, vocab=16):
        import torch as t
        self.weight = t.zeros(vocab, d)

    def __call__(self, x):
        return x @ self.weight.T


class _FakeInner:
    def __init__(self, d):
        self.norm = _FakeNorm(d)


class _FakeHF:
    def __init__(self, d=8):
        import torch as t
        self.device = t.device("cpu")
        self.model = _FakeInner(d)
        self.lm_head = _FakeHead(d)
        self.config = type("C", (), {"text_config": type(
            "T", (), {"final_logit_softcapping": None})()})()


class _FakeModel:
    def __init__(self, d=8):
        self.model = _FakeHF(d)
        self.revision = "abc123"
        self.dtype = "bfloat16"


def _lens(tmp_path, *, layers=(0, 1, 2)):
    entry = importer.SUPPORTED[MODEL]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=list(layers)).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {MODEL}\n")
    return importer.import_lens(MODEL, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


def _manifest(block):
    return Manifest.from_dict({"name": "s", "modelID": MODEL,
                               "jlensReadout": block})


def _open(tmp_path, block, run_dir=None):
    run_dir = run_dir or (tmp_path / "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    return tasks._open_jlens_trace(
        _manifest(block), _FakeModel(), str(tmp_path / "ws"),
        run_directory=str(run_dir), checkpoint=None, resuming=False,
        log=lambda _m: None)


def test_no_declaration_means_no_session_and_no_cost(tmp_path):
    assert tasks._open_jlens_trace(
        _manifest(None), _FakeModel(), str(tmp_path),
        run_directory=str(tmp_path), checkpoint=None, resuming=False,
        log=lambda _m: None) is None


def test_a_declared_readout_is_armed(tmp_path):
    record = _lens(tmp_path)
    session = _open(tmp_path, {"lensID": record.lensID, "layers": [0, 1],
                               "watchlist": [3]})
    assert session is not None
    assert session.config.layers == [0, 1]
    assert len(session.configHash) == 64


def test_a_missing_lens_id_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="without a lensID"):
        _open(tmp_path, {"layers": [0], "watchlist": [3]})


def test_an_unimported_lens_refuses_rather_than_skipping(tmp_path):
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="cannot be armed"):
        _open(tmp_path, {"lensID": "never-imported", "layers": [0],
                         "watchlist": [3]})


def test_a_layer_the_lens_never_fitted_refuses(tmp_path):
    record = _lens(tmp_path, layers=(0, 1, 2))
    with pytest.raises(RuntimeError, match="cannot be armed"):
        _open(tmp_path, {"lensID": record.lensID, "layers": [0, 99],
                         "watchlist": [3]})


def test_a_readout_that_would_record_nothing_refuses(tmp_path):
    record = _lens(tmp_path)
    with pytest.raises(RuntimeError, match="cannot be armed"):
        _open(tmp_path, {"lensID": record.lensID, "layers": [0],
                         "watchlist": [], "topK": 0})


def test_config_drift_from_the_pinned_hash_refuses(tmp_path):
    """The pin is the researcher's declared choices. A mismatch means the block
    was edited after freeze — a firewall violation, not something to run
    through."""
    record = _lens(tmp_path)
    with pytest.raises(RuntimeError, match="drifted from its pinned hash"):
        _open(tmp_path, {"lensID": record.lensID, "layers": [0],
                         "watchlist": [3], "configHash": "f" * 64})


def test_a_matching_pinned_hash_is_accepted(tmp_path):
    record = _lens(tmp_path)
    block = {"lensID": record.lensID, "layers": [0], "watchlist": [3]}
    first = _open(tmp_path, block, run_dir=tmp_path / "r1")
    block["configHash"] = first.configHash
    assert _open(tmp_path, block, run_dir=tmp_path / "r2") is not None


def test_the_session_reaches_the_executor(tmp_path):
    """The whole point: the run loop must PASS what it opened."""
    import inspect

    src = inspect.getsource(tasks._run_impl)
    assert "_open_jlens_trace(" in src
    assert "jlens_trace=jlens_trace" in src


def test_the_session_is_closed_with_the_writer(tmp_path):
    import inspect

    src = inspect.getsource(tasks._run_impl)
    assert "jlens_trace.close(" in src


EVIDENCE_MODEL = "google/gemma-3-27b-it"
REV = "005ad3404e59d6023443cb575daa05336842228a"


def _import(tmp_path, model_id=EVIDENCE_MODEL, *, layers=(0, 1, 2)):
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=list(layers)).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {model_id}\n")
    return importer.import_lens(model_id, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


# --- run start DERIVES tier and qualification (external review 2026-08-16) ---
#
# Behavioural, not source-inspection (review round 2): the earlier versions
# asserted that `_open_jlens_trace` CONTAINED certain substrings, which checks
# that the code looks right rather than that it behaves right.

def _runtime(revision=REV, dtype="bfloat16"):
    """The file's existing full fake, at a named revision — the run-start path
    resolves the tier and the qualification from what the RUNTIME presents."""
    model = _FakeModel()
    model.revision = revision
    model.dtype = dtype
    return model


def _armed(record, root, block_overrides=None, *, revision=REV,
           model_id=EVIDENCE_MODEL, log=None):
    """Call the real run-start helper and return its refusal or its session."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    block = {"lensID": record.lensID, "lensSHA256": record.source.tensorSHA256,
             "layers": [0, 1], "watchlist": [11], "topK": 0}
    block.update(block_overrides or {})
    manifest = Manifest.from_dict(
        {"name": "s", "modelID": model_id, "modelRevision": revision,
         "dtype": "bfloat16", "concepts": [], "maxTokens": 8,
         "jlensReadout": block})
    return tasks._open_jlens_trace(
        manifest, _runtime(revision), root,
        run_directory=str(root), checkpoint=None, resuming=False,
        log=(log or (lambda _m: None)), expected_generations=1)


def test_a_contradicted_tier_claim_refuses(tmp_path):
    """The tier is a property of the model. Every trace row stamps it, so a
    self-declared claim would travel into the artifact as though checked."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    with pytest.raises(RuntimeError, match="claims evidenceTier"):
        _armed(record, root, {"evidenceTier": "testing"})


def test_a_pin_that_does_not_resolve_refuses_at_run_start(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    with pytest.raises(RuntimeError, match="does not resolve"):
        _armed(record, root, {"qualificationID": "q-ghost"})


def test_an_unqualified_runtime_warns_and_stamps_nothing(tmp_path):
    """Exploratory reading stays possible; what it must not do is stamp a
    qualification it does not hold."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    lines = []
    session = _armed(record, root, log=lines.append)
    assert session.qualification_id is None
    assert any("NO passing qualification" in line for line in lines)


def test_the_budget_refuses_an_over_ceiling_study(tmp_path):
    """It was constructed, logged, and never consulted — a declared ceiling
    that stopped nothing."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    with pytest.raises(RuntimeError, match="over its declared budget"):
        _armed(record, root,
               {"topK": 10, "budget": {"maxFullVocabProjections": 1}})


def test_the_budget_is_part_of_the_pinned_identity(tmp_path):
    """An edit to the ceiling after freeze must be a verify violation like any
    other measurement-side change, so it belongs in configHash."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    loose = _armed(record, root, {"budget": {"maxArmedLayers": 8}}).configHash
    tight = _armed(record, root, {"budget": {"maxArmedLayers": 7}}).configHash
    assert loose != tight


def test_the_budget_prices_seeds_not_just_samplesPerItem():
    """With samplesPerItem == 1 the run loop emits one generation per declared
    SEED, so a three-seed study is three times the size `max(1,
    samplesPerItem)` priced it at (external review round 2). Sharding has
    always used this rule; now there is one resolver."""
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.experiment.tasks import effective_sample_count

    def m(**kw):
        return Manifest.from_dict({"name": "s", "modelID": MODEL,
                                   "concepts": [], **kw})

    assert effective_sample_count(m(seeds=[1, 2, 3], temperature=0.7)) == 3
    assert effective_sample_count(m(seeds=[1], temperature=0.0)) == 1
    # samplesPerItem wins only where the run loop actually uses it.
    assert effective_sample_count(
        m(seeds=[1, 2, 3], samplesPerItem=5, temperature=0.7)) == 5
    assert effective_sample_count(
        m(seeds=[1, 2, 3], samplesPerItem=5, temperature=0.0)) == 3
