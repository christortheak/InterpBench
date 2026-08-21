"""Duplicate task-prompt item ids (review finding F1).

Duplicate ids silently corrupt pairing on both engines — the report's choice
readouts (``_choice_readouts``) and the paired statistics key on
``promptID`` — and used to TRAP the Swift engine's report assembly. The
refusal therefore lives at LOAD (``tasks._load_prompts``), so run, validate,
sweep, and logprob all inherit it, and the message string is a cross-engine
contract replayed from the committed fixture
``prompts/fixtures/task-prompts-validation/cases.json`` (Swift twin:
``Tests/ExperimentKitTests/TaskPromptDuplicateIDTests.swift``).
"""

import json
import os
from pathlib import Path

import pytest

from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest

FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "fixtures" / "task-prompts-validation" / "cases.json"
)


def _cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _manifest(**extra):
    d = {"name": "t", "modelID": "Qwen/Qwen3-4B", "concepts": [], "conditions": []}
    d.update(extra)
    return Manifest.from_dict(d)


def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@pytest.mark.parametrize("case", _cases(), ids=[c["name"] for c in _cases()])
def test_load_validation_matches_committed_fixture(case, tmp_path):
    path = str(tmp_path / "items.jsonl")
    _write(path, case["jsonl"])
    manifest = _manifest(taskPromptsFile=path)
    if case["expect"] is None:
        prompts = tasks._load_prompts(manifest, None, str(tmp_path))
        assert len(prompts) == len(case["jsonl"])
    else:
        with pytest.raises(RuntimeError) as err:
            tasks._load_prompts(manifest, None, str(tmp_path))
        assert str(err.value) == case["expect"], (
            f"{case['name']}: this engine's refusal diverged from the "
            "cross-engine fixture (the Swift twin replays the same file)")


def test_fixture_covers_the_duplicate_rule_and_a_valid_case():
    expects = [c["expect"] for c in _cases()]
    assert any(e and "ids must be unique for pairing and reporting" in e
               for e in expects)
    assert any(e is None for e in expects), "fixture needs a valid case too"


def test_duplicate_refusal_is_inherited_by_the_run_entrypoint(tmp_path):
    """The unified executor loads prompts through ``_load_prompts`` before any
    model work — a duplicate-id file refuses at run start, never after
    generation compute (the acceptance criterion for the fix)."""
    path = str(tmp_path / "items.jsonl")
    _write(path, [
        json.dumps({"id": "case-a", "prompt": "First framing."}),
        json.dumps({"id": "case-b", "prompt": "Another item."}),
        json.dumps({"id": "case-a", "prompt": "Repeat framing."}),
    ])
    manifest = _manifest(taskPromptsFile=path)
    with pytest.raises(RuntimeError, match="duplicate item id 'case-a'"):
        tasks._load_prompts(manifest, None, str(tmp_path))
