"""A repair is a command the reader can run — on the client in front of them.

The workspace guide promises that a refusal's ``repairAction`` is something
its reader can carry out. The store and policy code behind both command
lines is shared, so for a long time a Python-client reader was told to run
``steerlab-cli`` verbs "on the Mac", and the engine named the Mac's verbs to
readers on Linux. ``experiment/command_vocabulary.py`` now renders each
repair for whoever shows it. This file holds that in place three ways:

1. **Through the cross-platform client's own envelope.** A catalogue of real
   refusals — freeze gates, frozen-study mutations, detach, sweep grids,
   declarations, the run path, the capability record — driven through
   ``client_cli.main`` under ``--json``. None may name ``steerlab-cli``, say
   "on the Mac", or carry the engine's two-client form.
2. **The shared renderers, as each speaker.** As the client: no Mac verb. As
   the Mac (the local bridge that serves the app and ``steerlab-cli``): no
   cross-platform client verb. As the engine, for an authoring act: "on your
   authoring client" and both spellings.
3. **The Mac command line's own documents** are Swift's, and their test is
   ``RefusalVocabularyTests.swift``.

Fixtures are neutral: one concept named ``signal``.
"""

import json
import os
import re

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import command_vocabulary as vocabulary
from steerlab_server.experiment import draft_protocol_policy
from steerlab_server.experiment import exclusions
from steerlab_server.experiment import experiment_store as store
from steerlab_server.experiment import freeze_policy
from steerlab_server.experiment import manifest_declaration_policy
from steerlab_server.experiment import rubric_inputs
from steerlab_server.experiment import task_inputs
from steerlab_server.experiment import truncation_gate
from steerlab_server.experiment.manifest import Manifest

#: What a cross-platform client reader must never be told.
CLIENT_FORBIDDEN = ("steerlab-cli", "on the Mac", "Mac-authority",
                    "(Mac command line)", "(cross-platform client)",
                    "on your authoring client")

#: A command spelled for the cross-platform client: its program, then one of
#: its families. Never preceded by a word character or a hyphen, so
#: ``steerlab-cli`` and ``steerlab-server`` do not match.
CLIENT_COMMAND = re.compile(
    r"(?<![\w-])steerlab (?:" + "|".join(client_cli.FAMILIES) + r")\b")

RUNNER = "http://127.0.0.1:1"


def _texts(document: dict) -> list[str]:
    """Every sentence a refusal document shows a reader."""
    error = document.get("error") or {}
    action = document.get("nextAction") or {}
    texts = [document.get("message") or "", error.get("reason") or "",
             error.get("repairAction") or "", action.get("verb") or "",
             action.get("detail") or ""]
    texts += [a.get("detail") or "" for a in document.get("advisories") or []]
    return [t for t in texts if t]


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _client(root: str, argv, capsys) -> tuple[int, dict]:
    code = client_cli.main(["--root", root, *argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def workspace(tmp_path, monkeypatch, capsys):
    """Three studies: ``s``, a draft with a concept, a baseline and a steered
    arm; ``f``, the same frozen with --force and no task prompts; ``e``, an
    empty draft. Plus ``d``, a draft whose pinned stimuli are changed after
    attach."""
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
    root = str(tmp_path / "workspace")
    for concept in ("signal", "other"):
        directory = os.path.join(root, "prompts", "concepts", concept)
        _write(os.path.join(directory, "positive.jsonl"), '{"text": "on"}\n')
        _write(os.path.join(directory, "negative.jsonl"), '{"text": "off"}\n')

    def ok(*argv):
        code, document = _client(root, argv, capsys)
        assert code == 0, document

    for name in ("s", "f"):
        ok("experiment", "create", name, "--model", "org/m", "--revision",
           "0" * 40)
        ok("experiment", "attach", name, "signal")
        ok("experiment", "declare-condition", name, "baseline", "--baseline",
           "--alpha-units", "raw")
        ok("experiment", "declare-condition", name, "steered", "--slots",
           "signal:1:0.5", "--alpha-units", "raw")
    ok("experiment", "freeze", "f", "--force")
    ok("experiment", "create", "e", "--model", "org/m", "--revision", "0" * 40)
    ok("experiment", "create", "d", "--model", "org/m", "--revision", "0" * 40)
    ok("experiment", "attach", "d", "other")
    _write(os.path.join(root, "prompts", "concepts", "other",
                        "positive.jsonl"), '{"text": "changed"}\n')
    return root


#: Each entry: an invocation that REFUSES. Ids name the rule it reaches.
REFUSALS = {
    "validateEvidence": ["experiment", "freeze", "s"],
    "refreeze": ["experiment", "freeze", "f"],
    "frozenAttach": ["experiment", "attach", "f", "other"],
    "frozenProtocol": ["experiment", "set-protocol", "f", "--set",
                       "maxTokens=8"],
    "frozenSweepGrid": ["experiment", "set-sweep-grid", "f", "--alphas",
                        "0.05,0.1"],
    "conceptInUse": ["experiment", "detach", "s", "signal"],
    "conceptNotPinned": ["experiment", "detach", "s", "absent"],
    "sweepGridRule": ["experiment", "set-sweep-grid", "s", "--alphas",
                      "0.1,0.05"],
    "absoluteLayersNeedDepth": ["experiment", "set-sweep-grid", "s",
                                "--layers", "3,5"],
    "alphaUnits": ["experiment", "declare-condition", "s", "arm", "--slots",
                   "signal:1:0.5"],
    "unknownInstrument": ["experiment", "set-protocol", "s", "--set",
                          'outcomeInstruments=["sampledTxt"]'],
    "exclusionRules": ["experiment", "set-protocol", "s", "--set",
                       'exclusionRules=[{"rule": "outOfRange"}]'],
    "emptyStudy": ["experiment", "freeze", "e"],
    "pinDrift": ["experiment", "verify", "d"],
    "notFound": ["experiment", "freeze", "missing"],
    "draftRun": ["run", "s", "--runner", RUNNER],
    "noTaskPrompts": ["run", "f", "--runner", RUNNER],
    "capabilityRecord": ["model", "set-capability", "org/m", "systemRole",
                         "none", "--reason", "a reason"],
}


@pytest.mark.parametrize("rule", sorted(REFUSALS))
def test_no_refusal_through_the_client_names_another_clients_command(
        workspace, capsys, rule):
    """CONTRACT: every refusal the cross-platform client shows is in its own
    vocabulary. It is the reader's only machine, and ``steerlab-cli`` is not
    on it."""
    code, document = _client(workspace, REFUSALS[rule], capsys)
    assert code != 0, (rule, document)
    assert document.get("error"), (rule, document)
    for text in _texts(document):
        for phrase in CLIENT_FORBIDDEN:
            assert phrase not in text, (rule, phrase, text)


def test_the_client_repairs_name_its_own_route(workspace, capsys):
    """Not just an absence: where the engine would say "validate here", the
    client names its runner route, and its freeze is its own verb."""
    _code, document = _client(workspace, REFUSALS["validateEvidence"], capsys)
    repair = document["error"]["repairAction"]
    assert repair.startswith("steerlab run s --runner <url> --verb validate")
    assert repair.endswith("then steerlab experiment freeze s")


# =============================================================================
# 2. The shared renderers, as each speaker
# =============================================================================


def _rendered() -> dict:
    """Every authoring repair the shared code composes, as the current
    speaker says it."""
    manifest = Manifest.from_dict({"name": "s", "modelID": "org/m",
                                   "status": "frozen"})
    texts = {f"freeze:{gate}": freeze_policy.freeze_gate_repair(gate, "s")
             for gate in freeze_policy.FORCED_GATE_IDS}
    texts.update({
        "conceptInUse": store.concept_in_use_repair("s"),
        "conceptNotPinned": store.concept_not_pinned_repair("s"),
        "sweepSelectionOwns": store.sweep_selection_owns_repair(
            "s", "--objective"),
        "conflictingDepth": store.conflicting_depth_repair("s"),
        "sweepGrid": store.sweep_grid_repair("s"),
        "absoluteLayersNeedDepth": store.absolute_layers_need_depth_repair(
            "s"),
        "absoluteLayersOutOfRange": store.absolute_layers_out_of_range_repair(
            "s", 8),
        "unknownInstrument":
            draft_protocol_policy.unknown_outcome_instrument_repair("s"),
        "pinRequired": exclusions.pin_required_repair(),
        "truncation": truncation_gate.repair_action(
            "s", 8, reasoning_max_tokens=64),
        "noRubric": rubric_inputs.no_rubric_repair("s"),
        "missingRubric": rubric_inputs.missing_rubric_repair(
            "s", "prompts/rubrics/x.md"),
        "missingTaskPrompts": task_inputs.missing_task_prompts_repair(
            "s", "prompts/tasks/x.jsonl"),
        "noTaskPrompts": task_inputs.no_task_prompts_refusal(
            manifest).repair_action,
        "noJudge": manifest_declaration_policy.no_judge_declared_reason("s"),
    })
    return texts


def test_as_the_client_no_renderer_names_a_mac_command():
    with vocabulary.speaking_as(vocabulary.CLIENT):
        rendered = _rendered()
    for key, text in rendered.items():
        for phrase in CLIENT_FORBIDDEN:
            assert phrase not in text, (key, phrase, text)
        assert CLIENT_COMMAND.search(text) or "steerlab run" in text, \
            (key, text)


def test_as_the_mac_no_renderer_names_a_cross_platform_client_command():
    """The local bridge serves the app and ``steerlab-cli``; whatever it
    composes is shown there, so it names that command line's verbs alone."""
    with vocabulary.speaking_as(vocabulary.MAC):
        rendered = _rendered()
    for key, text in rendered.items():
        assert not CLIENT_COMMAND.search(text), (key, text)
        assert "cross-platform client" not in text, (key, text)
        assert "on your authoring client" not in text, (key, text)


def test_as_the_engine_an_authoring_repair_names_both_clients():
    """The engine executes and never authors, and it cannot know which
    client its reader has: an authoring repair says where, and spells it for
    both."""
    rendered = _rendered()
    for key, text in rendered.items():
        if "steerlab-cli" not in text:
            continue        # an engine verb only (e.g. validate here)
        assert "on your authoring client: steerlab-cli " in text, (key, text)
        assert "(Mac command line), or " in text, (key, text)
        assert CLIENT_COMMAND.search(text), (key, text)


def test_the_bridge_speaks_as_the_mac(monkeypatch):
    """``python -m steerlab_server.client.diagnostic_workspace`` is how the
    app and ``steerlab-cli`` reach the shared code; its entry point declares
    the Mac as the speaker for the whole call, and leaves nothing behind."""
    from steerlab_server.client import diagnostic_workspace

    seen = []
    monkeypatch.setattr(diagnostic_workspace, "main",
                        lambda: seen.append(vocabulary.surface()) or 0)
    assert diagnostic_workspace.main_for_the_mac() == 0
    assert seen == [vocabulary.MAC]
    assert vocabulary.surface() == vocabulary.ENGINE


def test_the_client_entry_point_restores_the_speaker(tmp_path, capsys):
    """An in-process client call must not leave the process speaking as the
    client: the engine's routes share the process in tests and in the
    managed runner's own parent."""
    client_cli.main(["--root", str(tmp_path), "experiment", "list"])
    capsys.readouterr()
    assert vocabulary.surface() == vocabulary.ENGINE
