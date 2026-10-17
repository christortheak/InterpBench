"""The choice-prompt template's README says which item keys reach run records.

It used to say every other per-item key was "carried into run records". Only
a fixed set is: ``condition_execution._PROMPT_META_KEYS`` (Swift twin: the
science-layer fields of ``GenerationRecord`` and ``ChoiceRecord``), plus the
item's id and text, ``options`` on an answer-option reading, and a scripted
``transcript``. A researcher who believed the old sentence would look for a
column in the records that was never written.
"""
from pathlib import Path

from steerlab_server.experiment import condition_execution
from steerlab_server.client.workspace_bootstrap import SEED

README = SEED / "prompts" / "templates" / "task-prompts-choice" / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_the_readme_names_every_key_a_record_carries():
    text = _readme()
    for key in condition_execution._PROMPT_META_KEYS:
        assert f"`{key}`" in text, f"the README does not name `{key}`"
    for key in ("id", "promptID", "options", "transcript"):
        assert f"`{key}`" in text, f"the README does not name `{key}`"


def test_the_readme_does_not_promise_that_other_keys_are_carried():
    text = " ".join(_readme().split())
    assert "carried into run records" not in text
    assert "no record carries it" in text


def test_the_checkout_copy_matches_the_packaged_seed():
    checkout = Path(__file__).parents[2] / "prompts" / "templates" / "task-prompts-choice" / "README.md"
    if checkout.is_file():
        assert checkout.read_text(encoding="utf-8") == _readme()
