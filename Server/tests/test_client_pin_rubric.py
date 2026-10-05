"""``steerlab experiment pin-rubric``: pin a judge rubric by path, hash computed.

Pinning a rubric on the cross-platform client used to mean typing its SHA-256
into ``set-protocol``. The verb takes the Mac verb's shape — ``<name>
<rubric> [--judges …] [--judge-pin …]`` — computes the hash from the file's
bytes, writes the judge panel and the ``evaluation`` block the pair implies,
and edits only a reviewed draft (``--manifest-sha256`` from ``experiment
inspect``).
"""

import hashlib
import json

import pytest

from steerlab_server import client_cli
from steerlab_server.client import judge_rubrics
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import manifest_declaration_policy as policy

RUBRIC = b"Score each response from 1 to 7 for warmth.\n"


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "prompts" / "rubrics").mkdir(parents=True)
    (root / "prompts" / "rubrics" / "warmth.md").write_bytes(RUBRIC)
    es.create("study", model_id="org/model", revision="abc1234", root=str(root))
    return root


def _cli(capsys, root, *arguments):
    capsys.readouterr()
    code = client_cli.main([*arguments, "--root", str(root), "--json"])
    return code, json.loads(capsys.readouterr().out)


def _digest(capsys, root):
    code, document = _cli(capsys, root, "experiment", "inspect", "study")
    assert code == 0
    return document["result"]["manifestFileSHA256"]


def test_the_rubric_is_pinned_with_the_hash_of_its_bytes(workspace, capsys):
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--judges", "first:claude,second:local:org/judge",
        "--judge-pin", "second=abc1234:bf16",
        "--manifest-sha256", _digest(capsys, workspace))
    assert code == 0, document
    expected = hashlib.sha256(RUBRIC).hexdigest()
    assert document["result"]["judgeRubricHash"] == expected
    saved = es.load_raw("study", str(workspace))
    assert saved["judgeRubricFile"] == "prompts/rubrics/warmth.md"
    assert saved["judgeRubricHash"] == expected
    assert saved["judges"] == [
        {"name": "first", "kind": "claude"},
        {"name": "second", "kind": "local", "model": "org/judge",
         "revision": "abc1234", "dtype": "bfloat16"}]
    # The declaration the pin pair implies, as the Mac verb writes it.
    assert saved["evaluation"] == {"kind": "pairedJudge", "judgeModel": "",
                                   "judgePrompt": ""}
    assert document["result"]["evaluationKind"] == "pairedJudge"
    assert document["state"] == "ready"


def test_one_judge_is_legal_and_says_what_it_costs(workspace, capsys):
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--judges", "only:claude", "--manifest-sha256", _digest(capsys, workspace))
    assert code == 0
    assert document["state"] == "okWithAdvisories"
    assert document["advisories"] == [{"code": "judgePanelTooSmall",
                                       "detail": policy.SINGLE_JUDGE_PANEL_ADVISORY}]


def test_a_changed_draft_is_refused_and_left_alone(workspace, capsys):
    stale = _digest(capsys, workspace)
    es.set_protocol("study", {"temperature": 0.5}, str(workspace))
    before = (workspace / "experiments" / "study" / "experiment.json").read_bytes()
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--manifest-sha256", stale)
    assert code == 65 and document["state"] == "refused"
    assert document["error"]["code"] == "staleManifest"
    assert (workspace / "experiments" / "study" / "experiment.json").read_bytes() == before


def test_a_frozen_study_is_refused(workspace, capsys):
    raw = es.load_raw("study", str(workspace))
    raw["status"] = "frozen"
    es.save_raw(raw, str(workspace))
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--manifest-sha256", _digest(capsys, workspace))
    assert code == 65 and document["error"]["code"] == "statusImmutable"


def test_a_missing_rubric_names_the_convention_and_the_verb(workspace, capsys):
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/nope.md",
        "--manifest-sha256", _digest(capsys, workspace))
    assert code == 65
    assert document["error"]["code"] == "missingPrerequisite"
    assert document["error"]["reason"].startswith("judge rubric file not found: ")
    assert "under prompts/rubrics/" in document["error"]["repairAction"]
    assert "steerlab experiment pin-rubric study" in document["error"]["repairAction"]


@pytest.mark.parametrize("arguments", [
    ("--judges", "first:robot"),
    ("--judges", "first:local:org/judge:provider"),
    ("--judges", "first:local:org/judge", "--judge-pin", "first=main"),
    ("--judges", "first:local:org/judge", "--judge-pin", "first=abc1234:fp8"),
    ("--judges", "first:claude", "--judge-pin", "first=abc1234"),
    ("--judges", "first:claude", "--judge-pin", "other=abc1234"),
])
def test_a_malformed_panel_is_a_usage_refusal(workspace, capsys, arguments):
    before = (workspace / "experiments" / "study" / "experiment.json").read_bytes()
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        *arguments, "--manifest-sha256", _digest(capsys, workspace))
    assert code == 64 and document["state"] == "blocked"
    assert document["error"]["code"] == "usage"
    assert (workspace / "experiments" / "study" / "experiment.json").read_bytes() == before


def test_the_pins_follow_the_judge_until_its_model_changes(workspace, capsys):
    _cli(capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
         "--judges", "a:local:org/judge,b:claude", "--judge-pin", "a=abc1234:bf16",
         "--manifest-sha256", _digest(capsys, workspace))
    # The same panel re-declared without pins keeps them, and says so.
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--judges", "a:local:org/judge,b:claude", "--manifest-sha256", _digest(capsys, workspace))
    assert code == 0
    assert document["result"]["judges"][0]["revision"] == "abc1234"
    assert document["result"]["inheritedFromExistingDeclaration"] == [
        "judge 'a' revision abc1234…", "judge 'a' dtype bfloat16"]
    # A different model drops them: the pins described the old bytes.
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--judges", "a:local:org/other,b:claude", "--manifest-sha256", _digest(capsys, workspace))
    assert code == 0
    assert "revision" not in document["result"]["judges"][0]
    assert document["result"]["inheritedFromExistingDeclaration"] == [
        "dropped judge 'a' revision/dtype pins — it now names a different model"]


def test_a_pin_alone_keeps_the_declared_panel(workspace, capsys):
    _cli(capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
         "--judges", "a:local:org/judge,b:claude", "--manifest-sha256", _digest(capsys, workspace))
    code, document = _cli(
        capsys, workspace, "experiment", "pin-rubric", "study", "prompts/rubrics/warmth.md",
        "--judge-pin", "a=abc1234", "--manifest-sha256", _digest(capsys, workspace))
    assert code == 0
    assert [judge["name"] for judge in document["result"]["judges"]] == ["a", "b"]
    assert document["result"]["judges"][0]["revision"] == "abc1234"


def test_the_grammar_is_the_mac_verbs():
    assert judge_rubrics.parse_judges("a:openrouter:vendor/model:provider, b:local") == [
        {"name": "a", "kind": "openrouter", "model": "vendor/model", "provider": "provider"},
        {"name": "b", "kind": "local"}]
    assert judge_rubrics.parse_judge_pin("a=ABC1234:fp16") == {
        "name": "a", "revision": "ABC1234", "dtype": "float16"}
