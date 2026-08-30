"""The ``caseFamily: "sentencing"`` magic trigger, deprecated 2026-08-18.

Two claims, and they pull in opposite directions on purpose:

1. **It still works.** A manifest that already depends on the trigger keeps
   parsing ``parsedMonths`` out of every sampled record, byte for byte. A
   deprecation that changed a measured number would silently invalidate
   finished studies — the one thing the firewall exists to prevent.
2. **It says so.** Every site where it fires emits the closed-vocabulary
   advisory ``deprecatedImplicitSelection`` — in the verb's log, in the run
   directory's ``advisories.txt``, and in the ``--json`` envelope — and a study
   that DECLARES a ``numericParser`` emits nothing, because nothing was
   selected implicitly.

Swift twin: ``Tests/ExperimentKitTests/DeprecatedCaseFamilySelectionTests.swift``.
"""

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

from steerlab_server import cli, cli_envelope
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import manifest as manifest_mod
from steerlab_server.experiment import parser_registry, tasks
from steerlab_server.experiment.manifest import (IMPLICIT_CASE_FAMILY_ADVISORY,
                                                 Manifest,
                                                 implicit_case_family_endpoint)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SHIPPED_REGISTRY = os.path.join(REPO_ROOT, "prompts", "parsers",
                                "parser-registry.json")


# --- harness (the fake-model pattern from test_parser_registry) ---------------


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _fake_generate(output_text):
    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False,
                 **kwargs):
        return output_text
    return generate


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _install_registry(root):
    destination = parser_registry.registry_path(root)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(SHIPPED_REGISTRY, destination)
    with open(destination, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _study(root, name, *, numeric_parser=None, case_family=None,
           study_kind=None):
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["seeds"] = [0]
    d["temperature"] = 0.0
    d["maxTokens"] = 16
    if numeric_parser:
        d["numericParser"] = numeric_parser
    if case_family:
        d["caseFamily"] = case_family
    if study_kind:
        d["studyKind"] = study_kind
    d["variantConditions"] = []
    d["conditions"] = [{"name": "baseline", "slots": []}]
    es.save_raw(d, root)
    prompts_path = os.path.join(root, "prompts", "tasks", f"{name}.jsonl")
    _write(prompts_path,
           json.dumps({"id": "p0", "prompt": "Pass sentence."}) + "\n")
    return prompts_path


def _records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def _advisories_text(run_dir) -> str:
    path = os.path.join(run_dir, "advisories.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# --- 1. compatibility: the trigger still selects the endpoint ----------------


def test_the_trigger_still_selects_the_built_in_duration_endpoint(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "legacy", case_family="sentencing")
    monkeypatch.setattr(tasks, "generate",
                        _fake_generate("I impose 8 years and 3 months."))
    run_dir = tasks.run("legacy", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    assert records and records[0]["parsedMonths"] == 99.0


def test_any_other_label_selects_nothing(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "other", case_family="katzZamir")
    monkeypatch.setattr(tasks, "generate",
                        _fake_generate("I impose 8 years and 3 months."))
    run_dir = tasks.run("other", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    assert records and "parsedMonths" not in records[0]


# --- 2. the predicate: exactly when the trigger fires -------------------------


def _manifest(**kwargs) -> Manifest:
    return Manifest(name="demo", model_id="org/m", **kwargs)


def test_the_predicate_fires_only_on_the_undeclared_sentencing_label():
    assert implicit_case_family_endpoint(_manifest(case_family="sentencing"))
    assert not implicit_case_family_endpoint(_manifest())
    assert not implicit_case_family_endpoint(_manifest(case_family="katzZamir"))
    assert not implicit_case_family_endpoint(
        _manifest(case_family="siliconFormalism"))


def test_a_declared_numeric_parser_silences_the_predicate():
    """The row that keeps the advisory from becoming noise every migrated
    study has to ignore."""
    assert not implicit_case_family_endpoint(
        _manifest(case_family="sentencing", numeric_parser="sentencing-months"))
    # Whitespace is not a declaration.
    assert implicit_case_family_endpoint(
        _manifest(case_family="sentencing", numeric_parser="  "))
    assert implicit_case_family_endpoint(
        _manifest(case_family="sentencing", numeric_parser=""))


def test_multi_agent_studies_fire_on_the_label_alone():
    """The one asymmetry, created by the panel-effects decomposition: that site
    reads the label ALONE, so a declared parser does not silence it."""
    assert implicit_case_family_endpoint(
        _manifest(case_family="sentencing", numeric_parser="sentencing-months",
                  study_kind="multiAgent"))
    assert not implicit_case_family_endpoint(_manifest(study_kind="multiAgent"))


# --- 3. the advisory ----------------------------------------------------------


def test_the_advisory_code_is_in_the_closed_vocabulary():
    assert "deprecatedImplicitSelection" in cli_envelope.ADVISORY_CODES
    # A code outside the vocabulary cannot be minted at all.
    assert cli_envelope.advisory(
        "deprecatedImplicitSelection", "detail")["code"] \
        == "deprecatedImplicitSelection"


def test_the_advisory_sentence_names_the_trigger_and_the_replacement():
    """Matched on by agents and by the Swift twin, so its load-bearing parts
    are pinned rather than left to prose drift."""
    assert "caseFamily 'sentencing'" in IMPLICIT_CASE_FAMILY_ADVISORY
    assert "numericParser" in IMPLICIT_CASE_FAMILY_ADVISORY
    assert "deprecated" in IMPLICIT_CASE_FAMILY_ADVISORY
    assert "sentencing-months" in IMPLICIT_CASE_FAMILY_ADVISORY


def test_run_start_logs_and_stamps_the_advisory(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "legacy", case_family="sentencing")
    monkeypatch.setattr(tasks, "generate", _fake_generate("18 months."))
    lines = []
    run_dir = tasks.run("legacy", prompts, root, model_provider=_fake_model,
                        log=lines.append)
    assert any(IMPLICIT_CASE_FAMILY_ADVISORY in line for line in lines)
    assert any(line.startswith("ADVISORY:") for line in lines)
    # The durable half: a reader of the run directory alone learns how the
    # endpoint was chosen.
    assert IMPLICIT_CASE_FAMILY_ADVISORY in _advisories_text(run_dir)


def test_a_declared_parser_produces_no_advisory_anywhere(tmp_path, monkeypatch):
    root = str(tmp_path)
    _install_registry(root)
    prompts = _study(root, "declared", case_family="sentencing",
                     numeric_parser="sentencing-months")
    monkeypatch.setattr(tasks, "generate", _fake_generate("18 months."))
    lines = []
    run_dir = tasks.run("declared", prompts, root, model_provider=_fake_model,
                        log=lines.append)
    assert not any(IMPLICIT_CASE_FAMILY_ADVISORY in line for line in lines)
    assert IMPLICIT_CASE_FAMILY_ADVISORY not in _advisories_text(run_dir)
    # …and the endpoint is still produced, by the declared grammar.
    assert _records(run_dir)[0]["parsedMonths"] == 18.0


def test_no_case_family_produces_no_advisory(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "plain")
    monkeypatch.setattr(tasks, "generate", _fake_generate("18 months."))
    lines = []
    run_dir = tasks.run("plain", prompts, root, model_provider=_fake_model,
                        log=lines.append)
    assert not any(IMPLICIT_CASE_FAMILY_ADVISORY in line for line in lines)
    assert IMPLICIT_CASE_FAMILY_ADVISORY not in _advisories_text(run_dir)


def test_analyze_advises_when_the_trigger_picks_the_rescue_grammar(
        tmp_path, monkeypatch):
    """The analyze site: the null-only endpoint rescue's grammar came from the
    deprecated trigger."""
    root = str(tmp_path)
    prompts = _study(root, "legacy", case_family="sentencing")
    monkeypatch.setattr(tasks, "generate", _fake_generate("18 months."))
    tasks.run("legacy", prompts, root, model_provider=_fake_model,
              log=lambda *_: None)
    lines = []
    tasks.analyze("legacy", root, log=lines.append)
    assert any(IMPLICIT_CASE_FAMILY_ADVISORY in line for line in lines)


def test_the_advisory_stamp_appends_rather_than_overwrites(tmp_path):
    """``advisories.txt`` accumulates: more than one advisory can be true of
    the same run, and a truncating write would let the second erase the
    first."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "advisories.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("first advisory\n")
    tasks._advise_implicit_case_family(True, run_dir, lambda *_: None,
                                       write_file=True)
    text = _advisories_text(run_dir)
    assert text.startswith("first advisory\n")
    assert IMPLICIT_CASE_FAMILY_ADVISORY in text


def test_the_emitter_is_silent_when_the_trigger_does_not_fire(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    lines = []
    tasks._advise_implicit_case_family(False, run_dir, lines.append,
                                       write_file=True)
    assert lines == []
    assert not os.path.exists(os.path.join(run_dir, "advisories.txt"))


# --- 4. the envelope half -----------------------------------------------------


def test_the_envelope_advisory_fires_exactly_when_the_trigger_does(tmp_path):
    root = str(tmp_path)
    _install_registry(root)
    _study(root, "legacy", case_family="sentencing")
    _study(root, "declared", case_family="sentencing",
           numeric_parser="sentencing-months")
    _study(root, "unlabelled")

    fired = cli._implicit_case_family_advisories("legacy", root)
    assert len(fired) == 1
    assert fired[0]["code"] == "deprecatedImplicitSelection"
    assert fired[0]["detail"] == IMPLICIT_CASE_FAMILY_ADVISORY

    assert cli._implicit_case_family_advisories("declared", root) == []
    assert cli._implicit_case_family_advisories("unlabelled", root) == []
    # An unreadable manifest is the verb's problem to report, never the
    # advisory helper's: it must not raise and must not invent an advisory.
    assert cli._implicit_case_family_advisories("no-such-experiment", root) == []


def test_the_advisory_promotes_the_state_and_not_the_exit_code():
    """Advisories never change the exit code — the rule a ``set -e`` wrapper
    depends on. A run that fires this one still succeeds."""
    envelope = cli_envelope.success(
        "experiment run", "ran 'legacy'", changed=True,
        advisories=[cli_envelope.advisory("deprecatedImplicitSelection",
                                          IMPLICIT_CASE_FAMILY_ADVISORY)])
    assert envelope.state == "okWithAdvisories"
    assert envelope.exit_code == 0


# --- 5. the constant is not re-spelled per site -------------------------------


def test_every_site_says_the_same_sentence():
    """One string, one definition. Four sites emit this advisory; a second
    spelling would make an agent match on one of them and miss the rest."""
    assert manifest_mod.IMPLICIT_ENDPOINT_CASE_FAMILY == "sentencing"
    source_files = [
        os.path.join(REPO_ROOT, "Server", "steerlab_server", "cli.py"),
        os.path.join(REPO_ROOT, "Server", "steerlab_server", "experiment",
                     "tasks.py"),
    ]
    for path in source_files:
        with open(path, encoding="utf-8") as handle:
            body = handle.read()
        assert "caseFamily 'sentencing' selected" not in body, (
            f"{os.path.basename(path)} re-spells the advisory instead of "
            "importing IMPLICIT_CASE_FAMILY_ADVISORY")
