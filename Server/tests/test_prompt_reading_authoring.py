"""``promptReading``: the study's declaration, its verb, and its capability.

What this file asserts:

* ``experiment set-prompt-reading`` writes exactly the block the hand-authored
  cross-engine contract (``Tests/Fixtures/cross-engine/prompt-reading.json``)
  states, prints its line, and refuses malformed flags at 64 with its
  sentences; the Swift twin (``PromptReadingTests``) checks the same file;
* a frozen study refuses with the duplicate-to-iterate repair, on a dry run
  too; a dry run writes nothing; ``--clear`` removes the key;
* the key is in the manifest's content hash, survives a load, and a block
  outside the contract is a verify violation (and so blocks freeze);
* the engine advertises ``remoteStudy.promptReading`` and says in plain
  words how it reads long prompts and that it never splits a model across
  GPUs; ``runner submit`` refuses a declaring study to an engine without it.
"""

import contextlib
import io
import json
import os

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import experiment_store as store
from steerlab_server.experiment import prompt_reading as pr
from steerlab_server.experiment.manifest import Manifest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACT = os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                        "prompt-reading.json")


def _contract():
    with open(CONTRACT, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def workspace(tmp_path):
    root = str(tmp_path / "ws")
    store.create("demo", model_id="org/m", revision="0" * 40, root=root)
    return root


def _verb(root, *flags, name="demo"):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = client_cli.main(["--root", root, "experiment",
                                "set-prompt-reading", name, *flags, "--json"])
    return code, json.loads(out.getvalue())


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# --- the contract ---------------------------------------------------------------

@pytest.mark.parametrize("case", _contract()["declarations"],
                         ids=lambda c: " ".join(c["flags"]))
def test_the_verb_writes_the_contract_block_and_line(workspace, case):
    code, document = _verb(workspace, *case["flags"])
    assert code == 0, document
    written = store.load_raw("demo", workspace)["promptReading"]
    assert _canonical(written) == case["canonical"]
    result = document["result"]
    assert result["promptReading"] == written
    assert result["reading"] == case["reading"]
    assert result["written"] is True
    assert result["previousPromptReading"] is None
    assert document["message"] == case["line"]


@pytest.mark.parametrize("case", _contract()["malformedFlags"],
                         ids=lambda c: " ".join(c["flags"]) or "no flags")
def test_malformed_flags_are_refused_with_the_contract_sentence(workspace, case):
    code, document = _verb(workspace, *case["flags"])
    assert code == 64, document
    assert document["error"]["reason"] == case["reason"]
    assert ("steerlab experiment set-prompt-reading demo --single-pass"
            in document["error"]["repairAction"])
    assert "promptReading" not in store.load_raw("demo", workspace)


@pytest.mark.parametrize("case", _contract()["violations"],
                         ids=lambda c: _canonical(c["block"]))
def test_a_hand_edited_block_reads_as_the_contract_says(case):
    assert pr.violations(case["block"]) == case["violations"]


@pytest.mark.parametrize("case", _contract()["chunking"],
                         ids=lambda c: f"{_canonical(c['rule'])}@{c['promptTokens']}")
def test_the_rule_chunks_as_the_contract_says(case):
    assert pr.chunk_size(case["rule"], case["promptTokens"]) == case["chunkTokens"]


def test_the_outdated_engine_sentence_is_the_contract_one():
    outdated = _contract()["outdatedEngine"]
    assert pr.outdated_engine_reason(outdated["where"]) == outdated["reason"]
    assert pr.outdated_engine_repair() == outdated["repair"]


def test_the_device_defaults_are_the_shared_contract_constants():
    # FR2-SHARED: CUDA chunked 4096 above 8192; MPS 1024 above 4096; CPU one pass.
    assert pr.CUDA_DEFAULT == {"mode": "chunked", "chunkTokens": 4096,
                               "aboveTokens": 8192}
    assert pr.MPS_DEFAULT == {"mode": "chunked", "chunkTokens": 1024,
                              "aboveTokens": 4096}
    assert pr.CPU_DEFAULT == {"mode": "singlePass"}
    assert pr.STEP_DOWN_CHUNKS == (2048, 1024, 512)


# --- the verb's other paths -------------------------------------------------------

def test_a_dry_run_writes_nothing(workspace):
    code, document = _verb(workspace, "--single-pass", "--dry-run")
    assert code == 0
    assert document["message"].startswith("(dry run, nothing written) would declare")
    assert document["result"]["written"] is False
    assert "promptReading" not in store.load_raw("demo", workspace)


def test_the_same_declaration_again_changes_nothing(workspace):
    _verb(workspace, "--single-pass")
    code, document = _verb(workspace, "--single-pass")
    assert code == 0
    assert document["result"]["written"] is False
    assert document["message"].startswith("'demo' already declares")


def test_clear_removes_the_declaration(workspace):
    _verb(workspace, "--chunk-tokens", "2048", "--above-tokens", "8192")
    code, document = _verb(workspace, "--clear")
    assert code == 0
    assert document["result"]["promptReading"] is None
    assert document["result"]["previousPromptReading"] == {
        "mode": "chunked", "chunkTokens": 2048, "aboveTokens": 8192}
    assert "promptReading" not in store.load_raw("demo", workspace)
    code, document = _verb(workspace, "--clear")
    assert code == 0 and "nothing to clear" in document["message"]


def test_a_frozen_study_refuses_with_the_duplicate_repair(workspace):
    document = store.load_raw("demo", workspace)
    document["status"] = "frozen"
    store.save_raw(document, workspace, freeze_transition=True)
    for flags in (["--single-pass"], ["--single-pass", "--dry-run"]):
        code, refused = _verb(workspace, *flags)
        assert code == 65, refused
        assert refused["error"]["gate"] == "statusImmutable"
        repair = refused["error"]["repairAction"]
        assert "steerlab experiment duplicate demo demo-v2" in repair
        assert "steerlab experiment set-prompt-reading demo-v2" in repair


def test_a_missing_study_is_not_found(workspace):
    code, document = _verb(workspace, "--single-pass", name="nope")
    assert code == 66, document


# --- the manifest key ---------------------------------------------------------------

def test_the_declaration_is_content_and_survives_a_load(workspace):
    before = Manifest.load("demo", workspace).content_hash()
    _verb(workspace, "--chunk-tokens", "4096", "--above-tokens", "8192")
    loaded = Manifest.load("demo", workspace)
    assert loaded.prompt_reading == {"mode": "chunked", "chunkTokens": 4096,
                                     "aboveTokens": 8192}
    assert loaded.content_hash() != before
    assert pr.declared(loaded) == loaded.prompt_reading
    assert loaded.verify(workspace) == [] or not any(
        "promptReading" in v for v in loaded.verify(workspace))


def test_an_absent_or_null_key_is_no_declaration():
    base = {"name": "x", "modelID": "org/m", "status": "draft"}
    assert Manifest.from_dict(base).prompt_reading is None
    assert Manifest.from_dict({**base, "promptReading": None}).prompt_reading is None
    assert pr.declared({**base, "promptReading": None}) is None
    assert not pr.submission_needs_capability({**base, "promptReading": None}, "run")


def test_a_malformed_block_is_a_verify_violation(workspace):
    document = store.load_raw("demo", workspace)
    document["promptReading"] = {"mode": "chunked", "chunkTokens": 0,
                                 "aboveTokens": 8192}
    store.save_raw(document, workspace)
    violations = Manifest.load("demo", workspace).verify(workspace)
    assert ("promptReading chunkTokens must be a whole number from 1 to "
            "100,000,000") in violations


def test_only_a_model_loading_verb_needs_the_capability():
    declared = {"promptReading": {"mode": "singlePass"}}
    assert pr.submission_needs_capability(declared, "run")
    assert pr.submission_needs_capability(declared, "sweep")
    assert not pr.submission_needs_capability(declared, "analyze")
    assert not pr.submission_needs_capability({}, "run")


# --- the capability ---------------------------------------------------------------

def test_the_engine_advertises_it_and_says_how_it_reads_long_prompts():
    from steerlab_server.api.profile import capability_snapshot
    snapshot = capability_snapshot()
    assert snapshot["remoteStudy"]["promptReading"] is True
    assert pr.remote_honors(snapshot)
    described = snapshot["longPromptReading"]
    assert described["devices"]["cuda"] == pr.CUDA_DEFAULT
    assert described["devices"]["mps"] == pr.MPS_DEFAULT
    assert described["devices"]["cpu"] == pr.CPU_DEFAULT
    assert described["splitsModelAcrossGPUs"] is False
    assert "does not split a model across GPUs" in described["summary"]
    # Nested wins over dotted, and a nested null does not count as present.
    assert pr.remote_honors({"features": {"remoteStudy.promptReading": True}})
    assert not pr.remote_honors({"remoteStudy": {"promptReading": False},
                                 "features": {"remoteStudy.promptReading": True}})
    assert pr.remote_honors({"remoteStudy": {"promptReading": None},
                             "features": {"remoteStudy.promptReading": True}})
    assert not pr.remote_honors({"remoteStudy": {"declaredSeedPolicy": True}})
    assert not pr.remote_honors(None)


def test_runner_submit_refuses_a_declaring_study_to_an_older_engine(workspace):
    from steerlab_server.client import runner as runner_api
    _verb(workspace, "--single-pass")
    staged = {"experiment": "demo", "experimentContentHash":
              Manifest.load("demo", workspace).content_hash()}

    def check(capabilities, *, verb="run", bundle=staged, root=workspace):
        runner_api.require_prompt_reading(bundle, verb, capabilities,
                                          "http://runner.invalid", root)

    with pytest.raises(runner_api.RunnerRefusal) as refusal:
        check({"remoteStudy": {"declaredSeedPolicy": True}})
    assert refusal.value.code == "promptReadingUnsupported"
    assert "remoteStudy.promptReading" in str(refusal.value)
    check({"remoteStudy": {"promptReading": True}})       # the engine has it
    check({}, verb="analyze")                             # loads no model
    check({}, bundle={**staged, "experimentContentHash": "0" * 64})  # not ours
    check({}, root=None)                                  # nothing to vouch for
