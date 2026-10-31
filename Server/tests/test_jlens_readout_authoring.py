"""``experiment set-jlens-readout`` on the cross-platform client.

The engine reads, prices, and enforces a study's ``jlensReadout`` block; this
verb is the first thing that WRITES one. What this file asserts:

* the pins come from the workspace's imported lens record and the model
  cache, never from a caller (the client has no ``--lens-sha256``);
* the configuration hash is the one run start computes — checked against the
  engine's own ``TraceSession``, not against this module's arithmetic;
* the block passes the engine's verify and, once qualified, its freeze gate;
* the budget is priced with the engine's ``preflight`` at the token cap, and a
  declaration that cannot cover one response is refused;
* every refusal is typed and carries a repair, and a frozen study refuses with
  the duplicate-to-iterate repair before the lens is read.

The Swift twin is ``JLensReadoutAuthoringTests``; the cross-client parity of
the written block and of the design identity is asserted there, against this
client run in a subprocess.
"""

import json
import os
import subprocess
import sys

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import experiment_store as store
from steerlab_server.experiment import jlens_readout_authoring as readout
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.jlens import lens_store, schemas
from steerlab_server.jlens.readout import Budget, ReadoutConfig
from steerlab_server.jlens.trace import TraceSession

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "google/gemma-3-27b-it"
REVISION = "005ad3404e59d6023443cb575daa05336842228a"
LENS_SHA = "a" * 64
CONVERTED_SHA = "b" * 64


def _record(lens_id="lens-a", *, model=MODEL, layers=range(0, 61),
            revision=None, converted=True):
    return schemas.JLensRecord(
        lensID=lens_id,
        source=schemas.SourceRef(repo="org/lenses", folder="f",
                                 tensorFile="lens.pt", configFile="c.yaml",
                                 tensorSHA256=LENS_SHA),
        fit=schemas.FitProvenance(modelID=model, revision=revision,
                                  revisionKnown=revision is not None),
        sourceLayers=list(layers), dModel=8, targetLayer=61, nPrompts=1,
        converted=(schemas.ConvertedRef(
            path=f"runs/jlens-lenses/{lens_id}/jacobians.safetensors",
            dtype="float16", sha256=CONVERTED_SHA) if converted else None))


def _qualify(record, root, *, qid="q1", layers=(20, 21, 22, 26)):
    record.qualifications.append(schemas.Qualification(
        qualificationID=qid, modelID=MODEL, revision=REVISION,
        dtype="bfloat16", tier="evidence", passed=True,
        lensSHA256=record.source.tensorSHA256,
        convertedSHA256=record.converted.sha256, layers=list(layers)))
    lens_store.save(record, root)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A draft on an evidence-tier model, one imported lens, and a model cache
    holding the study revision's tokenizer."""
    root = str(tmp_path / "ws")
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    store.create("demo", model_id=MODEL, revision=REVISION, root=root)
    document = store.load_raw("demo", root)
    document["dtype"] = "bfloat16"
    document["maxTokens"] = 2048
    store.save_raw(document, root)
    lens_store.save(_record(), root)
    snapshot = hub / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text('{"model": "test"}')
    (snapshot / "tokenizer_config.json").write_text('{"x": 1}')
    return root


def _client(root, *args):
    return client_cli.main([client_cli.ROOT_FLAG, str(root), "experiment",
                            readout.VERB, *args, client_cli.JSON_FLAG])


def _document(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _declare(root, capsys, *args):
    code = _client(root, "demo", *args)
    return code, _document(capsys)


# =============================================================================
# The block and its pins
# =============================================================================


def test_the_block_pins_the_lens_record_and_the_tokenizer(workspace, capsys):
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20,26",
                              "--top-k", "10", "--watchlist", "5,9")
    assert code == 0, document
    assert document["state"] == "ready" and document["changed"] is True
    block = store.load_raw("demo", workspace)["jlensReadout"]
    assert block == document["result"]["jlensReadout"]
    assert block["lensID"] == "lens-a"
    assert block["lensSHA256"] == LENS_SHA
    assert block["layers"] == [20, 26]
    assert block["topK"] == 10 and block["watchlist"] == [5, 9]
    assert block["logitLensCompanion"] is True
    assert block["tokenizerHash"] == readout.tokenizer_hash(MODEL, REVISION)
    assert len(block["configHash"]) == 64
    # Absent settings are absent KEYS: the design identity hashes this block
    # opaquely, so a null would be data.
    assert None not in block.values()
    assert not {"topKLayers", "budget", "qualificationID"} & set(block)
    result = document["result"]
    assert result["pins"] == {
        "lensID": "lens-a", "lensSHA256": LENS_SHA, "lensModel": MODEL,
        "fittedLayers": "0-60", "tokenizerHash": block["tokenizerHash"],
        "qualificationID": None}
    assert result["pinsComplete"] is True and result["written"] is True


def test_the_configuration_hash_is_the_one_run_start_computes(workspace,
                                                              capsys):
    """Built the way ``run_readouts.open_jlens_trace`` builds it, through the
    engine's own ``TraceSession`` — not through this module's helper."""
    _qualify(lens_store.resolve("lens-a", workspace), workspace)
    code, document = _declare(
        workspace, capsys, "lens-a", "--layers", "20-22", "--top-k", "5",
        "--top-k-layers", "21", "--watchlist", "7", "--no-companion",
        "--max-trace-bytes", "4294967296")
    assert code == 0, document
    block = store.load_raw("demo", workspace)["jlensReadout"]
    assert block["qualificationID"] == "q1"
    config = ReadoutConfig(
        layers=[int(x) for x in block["layers"]],
        watchlist=[int(x) for x in block["watchlist"]],
        topK=int(block["topK"]),
        topKLayers=[int(x) for x in block["topKLayers"]],
        logitLensCompanion=bool(block["logitLensCompanion"]))
    budget = Budget(**{k: v for k, v in block["budget"].items()
                       if k in Budget.__dataclass_fields__})
    session = TraceSession(
        record=lens_store.resolve("lens-a", workspace), config=config,
        readout=None, writer=None, run_id="run",
        qualification_id=block["qualificationID"],
        tokenizer_hash=block["tokenizerHash"], budget=budget)
    assert session.configHash == block["configHash"]


def test_the_written_block_verifies_and_passes_the_freeze_gate(workspace,
                                                               capsys):
    _qualify(lens_store.resolve("lens-a", workspace), workspace)
    code, _ = _declare(workspace, capsys, "lens-a", "--layers", "20,26",
                       "--top-k", "10", "--record-token-ids")
    assert code == 0
    document = store.load_raw("demo", workspace)
    assert document["recordTokenIDs"] is True
    manifest = Manifest.from_dict(dict(document))
    assert manifest._verify_jlens_readout(workspace) == []
    # The gate the readout belongs to (``measurementPins``), unchanged.
    store._check_jlens_readout("demo", dict(document), workspace)


def test_the_newest_passing_qualification_is_pinned_and_a_named_one_kept(
        workspace, capsys):
    record = lens_store.resolve("lens-a", workspace)
    _qualify(record, workspace, qid="q1")
    _qualify(lens_store.resolve("lens-a", workspace), workspace, qid="q2")
    _declare(workspace, capsys, "lens-a", "--layers", "20", "--top-k", "3")
    assert store.load_raw("demo", workspace)["jlensReadout"][
        "qualificationID"] == "q2"
    _declare(workspace, capsys, "lens-a", "--layers", "20", "--top-k", "3",
             "--qualification", "q1")
    assert store.load_raw("demo", workspace)["jlensReadout"][
        "qualificationID"] == "q1"


def test_without_a_cached_tokenizer_the_hash_pins_are_left_out(
        workspace, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20",
                              "--top-k", "3")
    assert code == 0
    block = document["result"]["jlensReadout"]
    # run start hashes the tokenizer it finds, so a configHash without one
    # could never match: both stay out, and the note says what to do.
    assert "tokenizerHash" not in block and "configHash" not in block
    assert document["result"]["pinsComplete"] is False
    assert any("not in this machine's model cache" in note
               for note in document["result"]["notes"])
    # …and the hash printed on the machine that runs the study fills them.
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20",
                              "--top-k", "3", "--tokenizer-hash", "c" * 64)
    assert code == 0
    assert document["result"]["jlensReadout"]["tokenizerHash"] == "c" * 64
    assert document["result"]["pinsComplete"] is True


def test_a_tokenizer_hash_that_disagrees_with_the_cache_is_refused(
        workspace, capsys):
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20",
                              "--top-k", "3", "--tokenizer-hash", "c" * 64)
    assert code == 64
    assert document["error"]["code"] == "usage"
    assert "disagrees with the tokenizer" in document["error"]["reason"]
    assert "jlensReadout" not in store.load_raw("demo", workspace)


def test_the_tokenizer_hash_twins_the_engine_one(workspace):
    pytest.importorskip("torch")
    from steerlab_server.jlens import derive

    assert readout.TOKENIZER_FILES == derive._TOKENIZER_FILES
    assert readout.tokenizer_hash(MODEL, REVISION) == \
        derive.tokenizer_identity_hash(MODEL, REVISION)
    assert readout.tokenizer_hash(MODEL) == derive.tokenizer_identity_hash(MODEL)


# =============================================================================
# The budget
# =============================================================================


def test_the_worked_example_is_priced_before_anything_runs(workspace, capsys):
    """At a 2,048-token cap, top-50 at 8 layers with the companion costs 32,768
    full-vocabulary projections and about 41 MB of trace per response, so the
    default 200,000-projection ceiling covers 6 responses."""
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20-27",
                              "--top-k", "50", "--dry-run")
    assert code == 0
    estimate = document["result"]["estimate"]
    assert estimate["maxTokens"] == 2048
    assert estimate["perResponse"] == {
        "observations": 16_384, "fullVocabProjections": 32_768,
        "projectedTraceBytes": 41_287_680}
    assert estimate["responsesAllowed"] == 6
    assert estimate["limitedBy"] == "maxFullVocabProjections"
    assert estimate["budget"] == {
        "maxArmedLayers": 8, "maxWatchlist": 64, "maxTopK": 50,
        "maxObservations": 2_000_000, "maxTraceBytes": 2 << 30,
        "maxFullVocabProjections": 200_000}
    # A dry run writes nothing, and says how the Mac spells the same block.
    assert document["changed"] is False
    assert document["result"]["written"] is False
    assert "jlensReadout" not in store.load_raw("demo", workspace)
    command = document["result"]["macCommand"]
    assert command.startswith(
        f"steerlab-cli experiment {readout.VERB} demo lens-a --lens-sha256 "
        f"{LENS_SHA} --lens-model {MODEL} --fitted-layers 0-60 --layers 20-27 "
        "--top-k 50 --tokenizer-hash ")


def test_a_declared_raise_moves_the_ceiling_and_is_pinned(workspace, capsys):
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20-27",
                              "--top-k", "50",
                              "--max-full-vocab-projections", "1000000")
    assert code == 0
    assert document["result"]["estimate"]["responsesAllowed"] == 30
    assert document["result"]["jlensReadout"]["budget"] == {
        "maxFullVocabProjections": 1_000_000}


def test_a_readout_that_cannot_fit_one_response_is_refused(workspace, capsys):
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "0-11",
                              "--top-k", "5")
    assert code == 64
    reason = document["error"]["reason"]
    assert "even for one response at the 2048-token cap" in reason
    assert "12 armed layers exceeds 8" in reason
    assert "--max-armed-layers" in document["error"]["repairAction"]
    assert "jlensReadout" not in store.load_raw("demo", workspace)
    # Raising the declared ceilings is the repair, and it is accepted.
    code, _ = _declare(workspace, capsys, "lens-a", "--layers", "0-11",
                       "--top-k", "5", "--max-armed-layers", "12",
                       "--max-full-vocab-projections", "1000000")
    assert code == 0


# =============================================================================
# Refusals
# =============================================================================


@pytest.mark.parametrize("args, fragment", [
    (["--top-k", "3"], "needs --layers"),
    (["--layers", "20"], "neither a --watchlist nor a --top-k"),
    (["--layers", "20", "--top-k", "51"], "--top-k must be a whole number"),
    (["--layers", "20", "--top-k", "-1"], "--top-k must be a whole number"),
    (["--layers", "26-20", "--top-k", "3"], "runs backwards"),
    (["--layers", "twenty", "--top-k", "3"], "layer indices or ranges"),
    (["--layers", "20", "--watchlist", "5,5"], "names token 5 twice"),
    (["--layers", "20", "--watchlist", ",".join(map(str, range(65)))],
     "watches at most 64"),
    (["--layers", "20", "--top-k-layers", "20"], "no --top-k width"),
    (["--layers", "20", "--top-k", "3", "--top-k-layers", "21"],
     "--layers does not arm"),
    (["--layers", "70", "--top-k", "3"],
     "layer 70 is not a fitted source layer"),
    (["--layers", "20", "--top-k", "3", "--max-trace-bytes", "0"],
     "--max-trace-bytes must be a whole number"),
    (["--layers", "20", "--top-k", "3", "--qualification", "q-none"],
     "does not pass"),
    (["--layers", "20", "--top-k", "3", "--tokenizer-hash", "ABC"],
     "SHA-256 digest"),
])
def test_out_of_range_values_are_malformed(workspace, capsys, args, fragment):
    code, document = _declare(workspace, capsys, "lens-a", *args)
    assert code == 64, document
    assert document["state"] == "blocked"
    assert document["error"]["code"] == "usage"
    assert fragment in document["error"]["reason"]
    assert document["error"]["repairAction"]
    assert "jlensReadout" not in store.load_raw("demo", workspace)


def test_a_missing_lens_is_a_missing_prerequisite(workspace, capsys):
    code, document = _declare(workspace, capsys, "no-such-lens", "--layers",
                              "20", "--top-k", "3")
    assert code == 65
    assert document["error"]["gate"] == "missingPrerequisite"
    assert "jlens import" in document["error"]["repairAction"]


def test_an_unconverted_lens_is_a_missing_prerequisite(workspace, capsys):
    lens_store.save(_record("raw-lens", converted=False), workspace)
    code, document = _declare(workspace, capsys, "raw-lens", "--layers", "20",
                              "--top-k", "3")
    assert code == 65
    assert document["error"]["gate"] == "missingPrerequisite"


def test_a_lens_for_another_model_or_revision_is_refused(workspace, capsys):
    lens_store.save(_record("other", model="org/other-model"), workspace)
    code, document = _declare(workspace, capsys, "other", "--layers", "20",
                              "--top-k", "3")
    assert code == 65
    assert document["error"]["gate"] == "artifactPin"
    assert "org/other-model" in document["error"]["reason"]

    lens_store.save(_record("old-rev", revision="0" * 40), workspace)
    code, document = _declare(workspace, capsys, "old-rev", "--layers", "20",
                              "--top-k", "3")
    assert code == 65
    assert document["error"]["gate"] == "artifactPin"
    assert "pin-revision" in document["error"]["repairAction"]


def test_a_frozen_study_refuses_with_the_duplicate_repair(workspace, capsys):
    document = store.load_raw("demo", workspace)
    document["status"] = "frozen"
    store.save_raw(document, workspace, freeze_transition=True)
    # Even with a lens that does not exist: the status answers first.
    code, document = _declare(workspace, capsys, "no-such-lens", "--layers",
                              "20", "--top-k", "3")
    assert code == 65
    assert document["error"]["gate"] == "statusImmutable"
    repair = document["error"]["repairAction"]
    assert "steerlab experiment duplicate demo demo-v2" in repair
    assert f"steerlab experiment {readout.VERB} demo-v2" in repair


# =============================================================================
# Clearing
# =============================================================================


def test_the_empty_lens_id_clears_and_keeps_token_retention(workspace, capsys):
    _declare(workspace, capsys, "lens-a", "--layers", "20", "--top-k", "3",
             "--record-token-ids")
    code, document = _declare(workspace, capsys, "")
    assert code == 0 and document["changed"] is True
    assert document["result"]["jlensReadout"] is None
    after = store.load_raw("demo", workspace)
    assert "jlensReadout" not in after and after["recordTokenIDs"] is True
    # Nothing left: not a change.
    code, document = _declare(workspace, capsys, "")
    assert code == 0 and document["changed"] is False


def test_a_clear_takes_no_settings(workspace, capsys):
    code, document = _declare(workspace, capsys, "", "--layers", "20")
    assert code == 64
    assert "a clear takes no settings" in document["error"]["reason"]


def test_the_client_never_takes_a_lens_pin(workspace, capsys):
    """The pins are read here, never typed: the Mac's pin flags are not this
    client's flags."""
    spec = client_cli.spec_for("experiment", readout.VERB)
    assert not {"--lens-sha256", "--lens-model", "--fitted-layers",
                "--config-hash"} & set(spec.value_flags)
    code, document = _declare(workspace, capsys, "lens-a", "--layers", "20",
                              "--top-k", "3", "--lens-sha256", "a" * 64)
    assert code == 64


# =============================================================================
# The light install
# =============================================================================


def test_declaring_a_readout_pulls_no_heavy_dependency(workspace):
    script = f"""
import io, json, sys, contextlib
from steerlab_server import client_cli
with contextlib.redirect_stdout(io.StringIO()):
    code = client_cli.main(["--root", {workspace!r}, "experiment",
        "set-jlens-readout", "demo", "lens-a", "--layers", "20",
        "--top-k", "3", "--json"])
heavy = sorted(m for m in ("torch", "transformers", "fastapi") if m in sys.modules)
sys.stderr.write(json.dumps({{"code": code, "heavy": heavy}}))
"""
    proc = subprocess.run([sys.executable, "-c", script], text=True,
                          capture_output=True, check=False,
                          env={**os.environ, "PYTHONPATH": SERVER_DIR})
    report = json.loads(proc.stderr[proc.stderr.rindex("{"):])
    assert report == {"code": 0, "heavy": []}, proc.stderr
