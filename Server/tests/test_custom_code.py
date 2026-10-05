"""Custom code a shared study carries is noticed, acknowledged, and recorded.

An intervention policy's expert provider is Python the engine runs with
``exec``. A study shared as a pack or a bundle carries that code to a new
machine, so:

* applying a pack, importing a bundle, and attaching an agent each return the
  notice, naming the provider's source hash, while it is unacknowledged;
* ``experiment acknowledge-custom-code`` shows the code, then records who
  acknowledged which hash and when, in the workspace;
* ``run`` refuses a step that would execute an unacknowledged provider, with
  the acknowledgement as its repair, and nothing else.

Swift twin: ``CustomCodeNoticeTests``. The literals below are duplicated there.
"""
import hashlib
import json
import os

import pytest

from steerlab_server import client_cli
from steerlab_server.client import runner as runner_api, study_packs
from steerlab_server.experiment import bundles, custom_code
from steerlab_server.experiment import experiment_store as store

# The client verbs below set the workspace for the process; the managed-runner
# harness's autouse fixture keeps that from leaking between tests, and the run
# orchestration fixtures supply a scripted runner for the provenance check.
from test_local_runner import _isolated_environment  # noqa: F401 - autouse
from test_run_orchestration import (  # noqa: F401 - fixtures used by name
    RUNNER_URL,
    STUDY_NAME,
    _evidence_reference,
    evidence_archive,
    script,
)

SOURCE = ("def decide(context, tensor, scores, state, rng, assets):\n"
          "    return [Decision('change', 1)]\n")
DIGEST = hashlib.sha256(SOURCE.encode()).hexdigest()
REVISION = "0" * 40
MODEL = "org/tiny"

# Cross-engine literals (Swift twin: CustomCodeNoticeTests).
FILENAME_LITERAL = "custom-code-acknowledgements.json"
NOTICE_LITERAL = ("This study contains custom code from its author. It runs "
                  "with your permissions when the study runs. Run it only if "
                  "you trust the source.")
EXECUTING_LITERAL = ("pipeline", "run", "sweep")


def policy(source=SOURCE, *, name="expert-policy", declared=None):
    """A version-1 policy with an expert provider; valid without a model."""
    doc = {"schemaVersion": 1, "name": name,
           "binding": {"modelID": MODEL, "revision": REVISION,
                       "tokenizerSHA256": None, "rendering": "chatTemplate",
                       "coordinateConvention": "hf-decoder-block-v1/model.layers"},
           "site": {"kind": "residualPost", "layer": 0},
           "stages": ["decode"], "positions": "lastPosition", "probes": [],
           "actions": [{"id": "change", "kind": "add", "bounds": [0, 2],
                        "vector": [1.0, 0.0]}],
           "rules": [], "onError": "stop", "maxEvents": 8}
    if source is not None:
        doc["provider"] = {"sourceText": source,
                           "sourceSHA256": declared or hashlib.sha256(source.encode()).hexdigest(),
                           "assets": {}}
    else:
        doc["rules"] = [{"action": "change", "kind": "fixed", "value": 1}]
    raw = json.dumps(doc, sort_keys=True)
    return {"json": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def agent(name="expert-agent", *, policies=None):
    artifact = {"name": name, "baseModelID": MODEL, "baseRevision": REVISION,
                "adapters": [], "injections": [], "temperature": 0.0,
                "promptMode": "chatAssistant"}
    artifact["interventionPolicies"] = [policy()] if policies is None else policies
    return artifact


def write_agent(root, artifact):
    relative = f"runs/model-variants/{artifact['name']}/model-variant.json"
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1)
    with open(path, "rb") as handle:
        return relative, hashlib.sha256(handle.read()).hexdigest()


def condition(root, artifact):
    relative, digest = write_agent(root, artifact)
    return {"name": artifact["name"], "artifactPath": relative,
            "artifactHash": digest, "artifact": artifact}


def study_with_code(root, name="shared-study", *, frozen=True):
    root = str(root)
    store.create(name, model_id=MODEL, revision=REVISION, root=root)
    document = store.load_raw(name, root)
    document["variantConditions"] = [condition(root, agent())]
    document["taskPromptsFile"] = "prompts/tasks/items.jsonl"
    items = os.path.join(root, document["taskPromptsFile"])
    os.makedirs(os.path.dirname(items), exist_ok=True)
    with open(items, "w", encoding="utf-8") as handle:
        handle.write('{"id": "item-1", "prompt": "Describe the room."}\n')
    with open(items, "rb") as handle:
        document["taskPromptsHash"] = hashlib.sha256(handle.read()).hexdigest()
    store.save_raw(document, root)
    if frozen:
        store.freeze(name, force=True, root=root)
    return name


def main(argv, capsys):
    code = client_cli.main([*argv, "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


# --- the owner --------------------------------------------------------------

def test_the_cross_engine_literals_are_the_swift_ones():
    assert custom_code.FILENAME == FILENAME_LITERAL
    assert custom_code.NOTICE == NOTICE_LITERAL
    assert custom_code.EXECUTING_VERBS == EXECUTING_LITERAL


def test_providers_are_found_inside_attached_policy_bytes_and_named_by_their_code():
    nested = {"panel": {"agents": [{"artifact": agent("seat")}]}}
    plain = agent("plain", policies=[policy(None)])
    document = {"variantConditions": [{"artifact": agent()}, {"artifact": plain}],
                "multiAgent": nested}
    assert custom_code.providers(document) == [
        {"sha256": DIGEST, "policyNames": ["expert-policy"]}]
    assert custom_code.providers({"variantConditions": [{"artifact": plain}]}) == []
    # The hash is the code's own, never the document's claim about it.
    claimed = agent(policies=[policy(declared="f" * 64)])
    assert custom_code.providers(claimed)[0]["sha256"] == DIGEST
    assert custom_code.provider_sources(document) == {DIGEST: SOURCE}


def test_a_panel_seats_agent_is_found_through_its_references(tmp_path):
    """A multi-agent study reaches its seats' agents only by path: manifest →
    compiled panel → seat's agent file."""
    relative, digest = write_agent(str(tmp_path), agent("seat-agent"))
    panel = tmp_path / "prompts/panels/compiled/panel.json"
    panel.parent.mkdir(parents=True)
    panel.write_text(json.dumps({"agents": [{"id": "a", "variantArtifactPath": relative,
                                             "variantArtifactHash": digest}]}))
    document = {"studyKind": "multiAgent",
                "multiAgentScenarioPath": "prompts/panels/compiled/panel.json",
                "variantConditions": [{"artifactPath": "../outside.json"}]}
    assert custom_code.providers(document) == []
    assert custom_code.providers(document, tmp_path) == [
        {"sha256": DIGEST, "policyNames": ["expert-policy"]}]
    assert custom_code.run_refusal(document, tmp_path, study="s", verb="run",
                                   program="steerlab") is not None


def test_acknowledge_records_who_which_hash_and_when_once(tmp_path):
    document = {"variantConditions": [{"artifact": agent()}]}
    assert custom_code.notice(document, tmp_path, study="s", program="steerlab")["notice"] \
        == custom_code.NOTICE
    first = custom_code.acknowledge(tmp_path, document, [DIGEST], study="s",
                                    client="steerlab", program="steerlab",
                                    account="researcher", now="2026-10-05T00:00:00Z")
    assert first["acknowledged"] == [{
        "providerSHA256": DIGEST, "acknowledgedAt": "2026-10-05T00:00:00Z",
        "acknowledgedBy": "researcher", "client": "steerlab", "study": "s",
        "policyNames": ["expert-policy"]}]
    record = json.loads((tmp_path / custom_code.FILENAME).read_text())
    assert record == {"schemaVersion": 1, "acknowledgements": first["acknowledged"]}
    again = custom_code.acknowledge(tmp_path, document, [DIGEST], study="other",
                                    client="steerlab", program="steerlab")
    assert again["acknowledged"] == [] and again["alreadyAcknowledged"] == [DIGEST]
    assert json.loads((tmp_path / custom_code.FILENAME).read_text()) == record
    block = custom_code.notice(document, tmp_path, study="s", program="steerlab")
    assert block["notice"] is None and block["acknowledged"] is True
    assert block["providers"][0]["acknowledgedBy"] == "researcher"


def test_a_hash_the_study_does_not_carry_is_refused_and_nothing_is_written(tmp_path):
    document = {"variantConditions": [{"artifact": agent()}]}
    with pytest.raises(custom_code.CustomCodeError, match="carries no custom code with"):
        custom_code.acknowledge(tmp_path, document, ["a" * 64], study="s",
                                client="steerlab", program="steerlab")
    with pytest.raises(custom_code.CustomCodeError, match="nothing to acknowledge"):
        custom_code.acknowledge(tmp_path, {}, [DIGEST], study="s",
                                client="steerlab", program="steerlab")
    assert not (tmp_path / custom_code.FILENAME).exists()


def test_a_damaged_record_is_refused_rather_than_read_as_empty(tmp_path):
    (tmp_path / custom_code.FILENAME).write_text("{not json")
    document = {"variantConditions": [{"artifact": agent()}]}
    with pytest.raises(custom_code.CustomCodeError, match="cannot be read") as raised:
        custom_code.run_refusal(document, tmp_path, study="s", verb="run",
                                program="steerlab")
    assert custom_code.FILENAME in raised.value.repair_action


def test_only_steps_that_execute_agents_are_held(tmp_path):
    document = {"variantConditions": [{"artifact": agent()}]}
    for verb in ("extract", "validate", "evaluate", "analyze", "verify"):
        assert custom_code.run_refusal(document, tmp_path, study="s", verb=verb,
                                       program="steerlab") is None
    for verb in (*custom_code.EXECUTING_VERBS, None):
        refusal = custom_code.run_refusal(document, tmp_path, study="s", verb=verb,
                                          program="steerlab")
        assert refusal is not None and DIGEST in str(refusal)
        assert custom_code.NOTICE in str(refusal)
        assert f"--sha256 {DIGEST}" in refusal.repair_action
    assert custom_code.run_refusal({}, tmp_path, study="s", verb="run",
                                   program="steerlab") is None


# --- the client surfaces ----------------------------------------------------

def test_pack_apply_gives_the_notice_with_the_hash_until_acknowledged(tmp_path, capsys):
    root = str(tmp_path)
    artifact = agent()
    relative, digest = write_agent(root, artifact)
    pack = {"study": {"name": "shared-study", "experimentDescription": "shared",
                      "createdAt": "2026-10-05T00:00:00Z", "modelID": MODEL,
                      "modelRevision": REVISION, "status": "draft",
                      "variantConditions": [{"name": artifact["name"],
                                             "artifactPath": relative,
                                             "artifactHash": digest,
                                             "artifact": artifact}]},
            "files": {}}
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack))
    review = study_packs.preview(path.read_bytes(), root=tmp_path)
    code, document, _ = main(["--root", root, "pack", "apply", str(path),
                              "--review-sha256", review["reviewSHA256"]], capsys)
    assert code == 0, document
    block = document["result"]["customCode"]
    assert block["notice"] == NOTICE_LITERAL
    assert block["providers"][0]["sha256"] == DIGEST
    assert block["acknowledgeCommand"] == (
        f"steerlab experiment acknowledge-custom-code shared-study --sha256 {DIGEST}")
    # Human mode names the hash too.
    pack["study"]["name"] = "second-copy"
    path.write_text(json.dumps(pack))
    review = study_packs.preview(path.read_bytes(), root=tmp_path)
    assert client_cli.main(["--root", root, "pack", "apply", str(path),
                            "--review-sha256", review["reviewSHA256"]]) == 0
    out = capsys.readouterr().out
    assert NOTICE_LITERAL in out and DIGEST in out

    # Review mode shows the code and writes nothing.
    code, document, _ = main(["--root", root, "experiment",
                              "acknowledge-custom-code", "shared-study"], capsys)
    assert code == 0 and document["changed"] is False
    assert document["result"]["providers"][0]["sourceText"] == SOURCE
    assert not (tmp_path / custom_code.FILENAME).exists()

    code, document, _ = main(["--root", root, "experiment", "acknowledge-custom-code",
                              "shared-study", "--sha256", DIGEST], capsys)
    assert code == 0 and document["changed"] is True
    assert document["result"]["acknowledged"][0]["client"] == "steerlab"
    # The same code in another study is not noticed again.
    pack["study"]["name"] = "third-copy"
    path.write_text(json.dumps(pack))
    review = study_packs.preview(path.read_bytes(), root=tmp_path)
    code, document, _ = main(["--root", root, "pack", "apply", str(path),
                              "--review-sha256", review["reviewSHA256"]], capsys)
    assert document["result"]["customCode"]["notice"] is None


def test_acknowledging_a_hash_the_study_lacks_is_a_typed_refusal(tmp_path, capsys):
    root = str(tmp_path)
    name = study_with_code(root, frozen=False)
    code, document, _ = main(["--root", root, "experiment", "acknowledge-custom-code",
                              name, "--sha256", "b" * 64], capsys)
    assert code == 65
    assert document["error"]["code"] == "missingPrerequisite"
    assert document["error"]["repairAction"] == (
        f"steerlab experiment acknowledge-custom-code {name}")
    assert not (tmp_path / custom_code.FILENAME).exists()


def test_attach_agent_gives_the_notice(tmp_path, capsys):
    root = str(tmp_path)
    store.create("attach-study", model_id=MODEL, revision=REVISION, root=root)
    relative, digest = write_agent(root, agent())
    with open(os.path.join(root, "experiments/attach-study/experiment.json"), "rb") as handle:
        manifest_digest = hashlib.sha256(handle.read()).hexdigest()
    code, document, _ = main(["--root", root, "experiment", "attach-agent", "attach-study",
                              "--artifact", relative, "--artifact-sha256", digest,
                              "--manifest-sha256", manifest_digest], capsys)
    assert code == 0, document
    assert document["result"]["customCode"]["providers"][0]["sha256"] == DIGEST
    assert document["result"]["customCode"]["notice"] == NOTICE_LITERAL


def test_bundle_import_gives_the_notice(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    name = study_with_code(source)
    monkeypatch.setenv("STEERLAB_ROOT", str(source))
    meta = bundles.package_experiment(name, root=str(source))
    target = tmp_path / "target"
    target.mkdir()
    code, document, _ = main(["--root", str(target), "bundle", "import",
                              meta["bundlePath"]], capsys)
    assert code == 0, document
    block = document["result"]["customCode"]
    assert block["notice"] == NOTICE_LITERAL
    assert block["providers"][0]["sha256"] == DIGEST
    assert name in block["acknowledgeCommand"]


class _UnreachableRunner:
    """Stands in for the runner adapter: the refusal must come before any of
    it, and after acknowledgement the run gets as far as addressing it."""
    constructed = 0

    def __init__(self, **kwargs):
        type(self).constructed += 1
        self.base_url = kwargs.get("base_url")
        self.has_token = False

    def __getattr__(self, name):
        raise runner_api.RunnerError("test runner", repair_action="none",
                                     code="runnerUnreachable", state="failed")

    def close(self):
        pass


def test_run_refuses_unacknowledged_custom_code_before_packaging(tmp_path, monkeypatch, capsys):
    root = str(tmp_path)
    name = study_with_code(root)
    _UnreachableRunner.constructed = 0
    monkeypatch.setattr(runner_api, "RunnerClient", _UnreachableRunner)
    code, document, _ = main(["--root", root, "run", name, "--runner",
                              "http://127.0.0.1:9"], capsys)
    assert code == 65, document
    error = document["error"]
    assert error["code"] == "missingPrerequisite"
    assert DIGEST in error["reason"] and NOTICE_LITERAL in error["reason"]
    assert f"experiment acknowledge-custom-code {name} --sha256 {DIGEST}" in error["repairAction"]
    assert document["result"]["failedStage"] == "load"
    assert document["result"]["customCode"]["providers"][0]["sha256"] == DIGEST
    stages = {row["stage"]: row for row in document["result"]["stages"]}
    assert stages["package"]["state"] == client_cli.STAGE_NOT_REACHED
    assert _UnreachableRunner.constructed == 0
    assert not [entry for entry in os.listdir(os.path.join(root, "runs"))
                if entry.startswith("bundle-")]

    # Steps that run no agent are not held.
    code, document, _ = main(["--root", root, "run", name, "--runner",
                              "http://127.0.0.1:9", "--verb", "evaluate"], capsys)
    stages = {row["stage"]: row for row in document["result"]["stages"]}
    assert stages["load"]["state"] == client_cli.STAGE_OK, document

    assert client_cli.main(["--root", root, "experiment", "acknowledge-custom-code",
                            name, "--sha256", DIGEST]) == 0
    capsys.readouterr()
    _UnreachableRunner.constructed = 0
    code, document, _ = main(["--root", root, "run", name, "--runner",
                              "http://127.0.0.1:9"], capsys)
    stages = {row["stage"]: row for row in document["result"]["stages"]}
    assert stages["load"]["state"] == client_cli.STAGE_OK, document
    assert stages["package"]["state"] == client_cli.STAGE_OK
    assert _UnreachableRunner.constructed == 1


def test_the_provenance_record_shows_who_acknowledged_the_code_that_ran(
        tmp_path, script, evidence_archive, capsys):
    root = str(tmp_path / "workspace")
    name = study_with_code(root, STUDY_NAME)
    assert client_cli.main(["--root", root, "experiment", "acknowledge-custom-code",
                            name, "--sha256", DIGEST]) == 0
    capsys.readouterr()
    script.evidence = _evidence_reference(evidence_archive)
    code, document, _ = main(["--root", root, "run", name, "--runner", RUNNER_URL],
                             capsys)
    assert code == 0, document
    with open(document["result"]["provenancePath"], encoding="utf-8") as handle:
        stamp = json.load(handle)
    [row] = stamp["manifest"]["customCode"]
    assert row["sha256"] == DIGEST and row["acknowledged"] is True
    assert row["acknowledgedAt"] and row["acknowledgedBy"]
