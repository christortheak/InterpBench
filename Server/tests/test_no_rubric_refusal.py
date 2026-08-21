"""`evaluate` with a completed run and NO rubric anywhere (WP0 dry run #2's
skipped check, Christian-flagged).

Two faults, both measured against a scratch workspace before the fix:

1. The refusal that DID happen (no ``evaluation`` block at all) arrived
   untyped — ``verbFailed`` / exit 70 / "read the reason and repair the named
   input", which tells an agent nothing it can act on. Same on the Mac.
2. Worse, and only here: a DRAFT study with an evaluation block and no rubric
   anywhere reached the judges with the EMPTY STRING as its rubric. The
   observed run emitted twelve blinded judging packets built on nothing and
   exited 0 ("state": "ready"). Swift has always refused that input.

Both are now ``missingPrerequisite`` — the verb needs something the study
never declared — and the no-rubric sentence is byte-identical to Swift's
``JudgeRubricStore.noRubricRefusal``. It names ``steerlab-cli`` on BOTH
engines on purpose: authoring is Mac-authority (audit §10.x) and this CLI has
no ``pin-rubric`` verb to point at.
"""

import pytest

from steerlab_server.experiment import lifecycle_gates, tasks
from steerlab_server.experiment.manifest import Manifest


def _manifest(**overrides):
    raw = {"name": "no-rubric", "modelID": "org/m", "status": "draft"}
    raw.update(overrides)
    return Manifest.from_dict(raw)


def test_the_no_rubric_sentence_is_the_cross_engine_literal():
    assert tasks.no_rubric_refusal("s") == (
        "study 's' has no judge rubric — pin one: 'steerlab-cli experiment "
        "pin-rubric s prompts/rubrics/default-paired-v1.md' (any file under "
        "prompts/rubrics/; inline draft text is draft-only and cannot freeze)")


def test_an_empty_inline_rubric_refuses_instead_of_judging_on_nothing(tmp_path):
    manifest = _manifest(evaluation={"kind": "pairedJudge",
                                     "judgeModel": "claude-x",
                                     "judgePrompt": ""})
    logged = []
    with pytest.raises(lifecycle_gates.LifecycleError) as caught:
        tasks._resolve_rubric(manifest, str(tmp_path), logged.append)
    error = caught.value
    assert error.gate == lifecycle_gates.MISSING_PREREQUISITE
    assert str(error) == tasks.no_rubric_refusal("no-rubric")
    assert "pin-rubric" in error.repair_action
    assert "steerlab-server experiment evaluate no-rubric" in \
        error.repair_action
    # It must not have warned its way past the gate.
    assert logged == []


def test_whitespace_is_not_a_rubric(tmp_path):
    manifest = _manifest(evaluation={"kind": "pairedJudge",
                                     "judgeModel": "claude-x",
                                     "judgePrompt": "   \n\t "})
    with pytest.raises(lifecycle_gates.LifecycleError):
        tasks._resolve_rubric(manifest, str(tmp_path), lambda *_: None)


def test_a_pinned_rubric_file_that_is_gone_refuses_typed(tmp_path):
    """The other half of the same gap, found on the Mac 2026-08-18 and fixed on
    both engines: a rubric path that is not on disk. Swift let an
    ``NSCocoaErrorDomain`` dump out of ``pin-rubric``; this engine let a bare
    ``FileNotFoundError`` traceback out of ``evaluate`` (exit 1, no gate, no
    repair). Same gate, same sentence, engine-specific repair."""
    manifest = _manifest(judgeRubricFile="prompts/rubrics/nope.md",
                         judgeRubricHash="a" * 64)
    with pytest.raises(lifecycle_gates.LifecycleError) as caught:
        tasks._resolve_rubric(manifest, str(tmp_path), lambda *_: None)
    error = caught.value
    assert error.gate == lifecycle_gates.MISSING_PREREQUISITE
    # The cross-engine sentence (Swift: JudgeRubricStore.missingRubricRefusal).
    assert str(error).startswith("judge rubric file not found: ")
    assert str(error).endswith("prompts/rubrics/nope.md")
    # The repair names the CONVENTION directory, the Mac verb that pins, and
    # the re-run here — none of which a traceback carried.
    assert "under prompts/rubrics/" in error.repair_action
    assert ("steerlab-cli experiment pin-rubric no-rubric "
            "prompts/rubrics/nope.md") in error.repair_action
    assert "steerlab-server experiment evaluate no-rubric" in \
        error.repair_action


def test_a_task_prompt_file_that_is_gone_refuses_typed(tmp_path):
    """Swift's ``ExperimentTasks.loadTaskPrompts`` has typed this since step 7
    and this engine had not, so the identical input answered ``refused``/65
    with a runnable repair on the Mac and a traceback + 1 here."""
    manifest = _manifest(taskPromptsFile="prompts/tasks/nope.jsonl",
                         taskPromptsHash="b" * 64)
    with pytest.raises(lifecycle_gates.LifecycleError) as caught:
        tasks._load_prompts(manifest, None, str(tmp_path))
    error = caught.value
    assert error.gate == lifecycle_gates.MISSING_PREREQUISITE
    # Byte-identical to Swift's sentence for the same rule.
    assert str(error).startswith("task prompt file not found: ")
    assert ("steerlab-cli experiment pin-prompts no-rubric "
            "prompts/tasks/nope.jsonl") in error.repair_action
    assert "steerlab-server experiment run no-rubric" in error.repair_action


def test_a_real_inline_draft_rubric_still_judges_loudly(tmp_path):
    """The draft fallback is unchanged — only the EMPTY case moved. A draft
    may still judge from inline text with a warning; freezing still cannot."""
    manifest = _manifest(evaluation={"kind": "pairedJudge",
                                     "judgeModel": "claude-x",
                                     "judgePrompt": "Prefer the calmer one."})
    logged = []
    text, sha, path = tasks._resolve_rubric(
        manifest, str(tmp_path), logged.append)
    assert text == "Prefer the calmer one."
    assert (sha, path) == (None, None)
    assert any("UNPINNED inline rubric" in line for line in logged)
