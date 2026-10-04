"""A failed ``verify()`` is three different refusals, and they are named apart.

Observed before this file existed, on a draft with nothing attached:

* ``steerlab experiment freeze`` answered ``authoringRefused`` with the
  untyped repair ("this was not a typed refusal — read the reason"), and the
  Mac command line answered ``failed`` / ``verbFailed`` for the same draft;
* ``experiment verify`` answered ``pinDrift`` and advised restoring files to
  their pinned bytes, when nothing had been pinned at all.

What is pinned here:

1. The classification itself (:mod:`verification_refusal`), over sentences
   copied from BOTH engines' ``verify()`` — real drift must stay ``pinDrift``,
   because callers switch on it.
2. The empty draft through the client: ``refused``, exit 65, ``emptyStudy``,
   and one repair naming this client's own commands, identical from ``freeze``
   and ``verify``.
3. Real drift through the client: still ``pinDrift`` with the sentence it has
   always carried.
4. A declaration problem: neither of the above.

Swift twin: ``Tests/ExperimentKitTests/VerificationRefusalTests.swift``.
"""

import json
import os

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.experiment import lifecycle_gates
from steerlab_server.experiment import verification_refusal as refusal

PINNED_NOW = 1_000.0


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(cli_envelope, "now", lambda: PINNED_NOW)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    return root


def _concept_files(root, name="french"):
    directory = os.path.join(str(root), "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    for filename, text in (("positive.jsonl", '{"text": "bonjour"}\n'),
                           ("negative.jsonl", '{"text": "hello"}\n')):
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(text)


def _document(argv, capsys) -> tuple[int, dict]:
    """Run under ``--json`` and return ``(exit code, the one document)``."""
    capsys.readouterr()
    code = client_cli.main([*argv, "--json"])
    text = capsys.readouterr().out
    assert text.count("\n}") == 1, "more than one document on stdout"
    return code, json.loads(text)


def _edit_manifest(root, name, **fields):
    path = os.path.join(str(root), "experiments", name, "experiment.json")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    document.update(fields)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)


# =============================================================================
# 1. The classification
# =============================================================================

#: Drift sentences as ``verify()`` writes them, from both engines. Each reports
#: bytes: a pinned file that changed, is gone, or appeared.
DRIFT_SENTENCES = [
    # Python engine
    "task prompts 'prompts/tasks/a.jsonl' changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
    "task prompts 'prompts/tasks/a.jsonl': file missing at prompts/tasks/a.jsonl",
    "concept 'french' stimuli changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
    "concept 'french': validation.jsonl appeared after pinning (validationHash pinned null) — re-attach to pin it (found at prompts/concepts/french/validation.jsonl)",
    "concept 'french': pinned validation.jsonl missing at prompts/concepts/french/validation.jsonl",
    "markers.json appeared after pinning (markersHash pinned null) — re-freeze on a duplicate to pin it",
    "pinned markers.json missing — no attached concept has a markers.json anymore",
    "concept 'calm': no stories.jsonl under prompts/emotions/",
    "manifest content changed after freeze (hash mismatch)",
    "pinned parser registry missing at prompts/parsers/registry.json",
    "pinned J-lens 'lens-a' is not importable in this workspace (FileNotFoundError) — import it before running, or the readout cannot be reproduced",
    # Swift engine
    "concept 'french': stimulus files changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
    "concept 'french': stimulus files missing/unreadable",
    "concept 'french': validation.jsonl appeared after attach (pinned as absent) — re-attach to pin it",
    "markersHash is pinned but no attached concept has a markers.json",
    "judge rubric missing (prompts/rubrics/r.md)",
    "variant 'tuned' artifact missing (runs/x/agent.json)",
    "multi-agent scenario changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
]

#: Declaration sentences: the study's own settings, no file's bytes involved.
DECLARATION_SENTENCES = [
    "judge rubric pin is incomplete — judgeRubricFile and judgeRubricHash must both be set",
    "unknown studyType 'survey' — one of conceptStudy, agentComparison, confirmAgent, multiAgent",
    "samplesPerItem > 1 requires temperature > 0 (greedy decoding makes every sample identical)",
    "condition 'steered': references unattached concept 'calm'",
    "variant 'tuned' uses org/other, not the study model org/m",
]


@pytest.mark.parametrize("sentence", DRIFT_SENTENCES)
def test_a_sentence_about_bytes_is_drift(sentence):
    assert refusal.is_drift(sentence)
    assert refusal.gate([sentence]) == lifecycle_gates.PIN_DRIFT


@pytest.mark.parametrize("sentence", DECLARATION_SENTENCES)
def test_a_sentence_about_settings_is_the_declaration(sentence):
    assert not refusal.is_drift(sentence)
    assert refusal.gate([sentence]) == lifecycle_gates.STUDY_DECLARATION


def test_the_empty_study_is_its_own_gate_for_both_study_kinds():
    for sentinel in (refusal.EMPTY_MODEL_OUTPUT_VIOLATION,
                     refusal.EMPTY_MULTI_AGENT_VIOLATION):
        assert not refusal.is_drift(sentinel)
        assert refusal.gate([sentinel]) == lifecycle_gates.EMPTY_STUDY
        # …and it stays the empty study when a setting is also wrong: the
        # first thing to do is still to attach something.
        assert refusal.gate([sentinel, DECLARATION_SENTENCES[0]]) \
            == lifecycle_gates.EMPTY_STUDY


def test_drift_outranks_everything_else_in_the_list():
    """Callers switch on ``pinDrift``, so any drifted pin keeps the code — a
    refusal that changed its name because a second problem appeared beside the
    first would be worse than the mislabel being fixed."""
    mixed = [refusal.EMPTY_MODEL_OUTPUT_VIOLATION, DECLARATION_SENTENCES[0],
             DRIFT_SENTENCES[0]]
    assert refusal.gate(mixed) == lifecycle_gates.PIN_DRIFT
    assert refusal.repair("demo", mixed) == refusal.drift_repair(
        "demo", "steerlab")


def test_the_new_gates_are_in_the_closed_vocabulary():
    assert lifecycle_gates.EMPTY_STUDY in lifecycle_gates.LIFECYCLE_GATE_IDS
    assert lifecycle_gates.STUDY_DECLARATION in lifecycle_gates.LIFECYCLE_GATE_IDS


def test_the_empty_reason_drops_nothing_verify_said():
    reason = refusal.empty_reason(
        "demo", [refusal.EMPTY_MODEL_OUTPUT_VIOLATION,
                 DECLARATION_SENTENCES[0]])
    assert reason.startswith("'demo' is empty: nothing is attached yet")
    assert DECLARATION_SENTENCES[0] in reason
    assert refusal.EMPTY_MODEL_OUTPUT_VIOLATION not in reason


def test_the_typed_error_keeps_site_prose_except_for_the_empty_study():
    drifted = refusal.error("demo", [DRIFT_SENTENCES[0]], "cannot freeze: x")
    assert str(drifted) == "cannot freeze: x"
    assert drifted.gate == lifecycle_gates.PIN_DRIFT

    empty = refusal.error("demo", [refusal.EMPTY_MODEL_OUTPUT_VIOLATION],
                          "cannot freeze:\n  - no concepts or variants attached")
    assert str(empty) == refusal.empty_reason(
        "demo", [refusal.EMPTY_MODEL_OUTPUT_VIOLATION])
    assert empty.gate == lifecycle_gates.EMPTY_STUDY
    assert empty.repair_action == refusal.empty_repair(
        "demo", [refusal.EMPTY_MODEL_OUTPUT_VIOLATION], "steerlab")


# =============================================================================
# 2. The empty draft, through the client
# =============================================================================


def test_freezing_and_verifying_an_empty_draft_give_one_usable_refusal(
        workspace, capsys):
    assert client_cli.main(["experiment", "create", "demo", "--model",
                            "org/m"]) == 0

    code, frozen = _document(["experiment", "freeze", "demo"], capsys)
    assert code == 65
    assert frozen["state"] == "refused"
    assert frozen["error"]["code"] == "emptyStudy"
    assert frozen["error"]["gate"] == "emptyStudy"
    repair = frozen["error"]["repairAction"]
    # Plain words, and THIS client's commands: a concept, an agent, a
    # template, or the interview.
    for command in ("steerlab experiment attach demo <concept>",
                    "steerlab experiment attach-agent demo",
                    "steerlab design list",
                    "steerlab authoring study conceptStudy --json"):
        assert command in repair, command
    assert "steerlab-cli" not in repair
    assert "not a typed refusal" not in repair
    assert "nothing is attached yet" in frozen["error"]["reason"]

    code, verified = _document(["experiment", "verify", "demo"], capsys)
    assert code == 65
    assert verified["state"] == "refused"
    # The SAME refusal: code, reason, and repair.
    for key in ("code", "gate", "reason", "repairAction"):
        assert verified["error"][key] == frozen["error"][key], key
    assert verified["result"]["violations"] == [
        refusal.EMPTY_MODEL_OUTPUT_VIOLATION]

    # The draft is untouched by either refusal.
    assert frozen["changed"] is False and verified["changed"] is False
    manifest = json.load(open(os.path.join(
        str(workspace), "experiments", "demo", "experiment.json"),
        encoding="utf-8"))
    assert manifest["status"] == "draft"


def test_the_empty_refusal_is_65_in_human_mode_too(workspace, capsys):
    assert client_cli.main(["experiment", "create", "demo", "--model",
                            "org/m"]) == 0
    capsys.readouterr()
    assert client_cli.main(["experiment", "freeze", "demo"]) == 65
    spoken = capsys.readouterr().err
    assert "nothing is attached yet" in spoken
    assert "steerlab experiment attach demo <concept>" in spoken


def test_an_empty_multi_agent_draft_is_pointed_at_a_panel(workspace, capsys):
    assert client_cli.main(["experiment", "create", "demo", "--model",
                            "org/m"]) == 0
    _edit_manifest(workspace, "demo", studyKind="multiAgent")

    code, document = _document(["experiment", "verify", "demo"], capsys)
    assert code == 65
    assert document["error"]["code"] == "emptyStudy"
    repair = document["error"]["repairAction"]
    assert "steerlab panel list" in repair
    assert "steerlab panel compile" in repair
    assert "steerlab authoring study multiAgent --json" in repair


# =============================================================================
# 3. Real drift keeps its code and its sentence
# =============================================================================


def test_real_drift_is_still_pin_drift_from_verify_and_from_freeze(
        workspace, capsys):
    _concept_files(workspace)
    assert client_cli.main(["experiment", "create", "demo", "--model",
                            "org/m"]) == 0
    assert client_cli.main(["experiment", "attach", "demo", "french"]) == 0
    with open(os.path.join(str(workspace), "prompts", "concepts", "french",
                           "positive.jsonl"), "w", encoding="utf-8") as handle:
        handle.write('{"text": "salut"}\n')

    code, verified = _document(["experiment", "verify", "demo"], capsys)
    assert code == 65
    assert verified["error"]["code"] == "pinDrift"
    assert verified["error"]["gate"] == "pinDrift"
    assert verified["error"]["reason"] == (
        "1 pinned input(s) of 'demo' no longer match their hashes")
    # Byte-for-byte the repair this refusal carried before the split.
    assert verified["error"]["repairAction"] == (
        "steerlab experiment verify demo  (names every drifted pin); then "
        "restore the named files, or duplicate the study and re-pin: "
        "steerlab experiment duplicate demo demo-v2")

    # Freeze meets the same drift and now names it the same way; it used to
    # answer `authoringRefused` with the untyped repair.
    code, frozen = _document(["experiment", "freeze", "demo"], capsys)
    assert code == 65
    assert frozen["error"]["code"] == "pinDrift"
    assert frozen["error"]["reason"].startswith("cannot freeze:\n  - ")
    assert "changed since pinning" in frozen["error"]["reason"]
    assert frozen["error"]["repairAction"] == \
        verified["error"]["repairAction"]


# =============================================================================
# 4. A declaration problem is neither
# =============================================================================


def test_a_declaration_problem_is_not_called_drift(workspace, capsys):
    _concept_files(workspace)
    assert client_cli.main(["experiment", "create", "demo", "--model",
                            "org/m"]) == 0
    assert client_cli.main(["experiment", "attach", "demo", "french"]) == 0
    # Half a pin: a rubric file named with no hash. No file changed.
    _edit_manifest(workspace, "demo",
                   judgeRubricFile="prompts/rubrics/rubric.md")

    code, verified = _document(["experiment", "verify", "demo"], capsys)
    assert code == 65
    assert verified["state"] == "refused"
    assert verified["error"]["code"] == "studyDeclaration"
    assert verified["error"]["gate"] == "studyDeclaration"
    assert "no longer match their hashes" not in verified["error"]["reason"]
    assert "declared settings of 'demo'" in verified["error"]["reason"]
    repair = verified["error"]["repairAction"]
    assert "restore" not in repair
    assert "steerlab experiment verify demo" in repair
    assert verified["result"]["violations"] == [
        "judge rubric pin is incomplete — judgeRubricFile and "
        "judgeRubricHash must both be set"]

    code, frozen = _document(["experiment", "freeze", "demo"], capsys)
    assert code == 65
    assert frozen["error"]["code"] == "studyDeclaration"
    assert frozen["error"]["repairAction"] == repair
