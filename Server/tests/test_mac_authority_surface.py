"""Gate-5 dry run #2's P1/P2/P3 punch list, server side.

Five separate promises the dry run found unkept, each with its own test here
because each fails for a different reason:

1. **The Mac-authority redirect** (punch-list #8). Authoring is Mac-authority
   BY POLICY (audit §10.x), and ``AGENTS.md`` §7 promises "when a verb is
   unavailable here the refusal says so". It did not: an authoring verb typed
   against this engine fell through to a verb ROSTER, which says the verb is
   unknown, not that it lives somewhere else — and typed with no positional it
   never reached the roster at all, answering ``usage: experiment attach
   <name>`` and thereby asserting the verb is real here.

2. **``data check <experiment>``** (#8). This engine's ``data check`` takes two
   directory-driven dataset templates; the manifest-driven readiness checklist
   is the Mac verb. Answering an experiment name with a two-template usage line
   said "you typed a bad subject" when the truth is "that check is on the other
   engine".

3. **Re-freeze** (#9). The last status guard in ``experiment_store`` left
   untyped, so the commonest possible retry answered ``verbFailed``/70 —
   indistinguishable from a crash — while its two siblings already said
   ``statusImmutable``/65 with a runnable repair.

4. **``notFound`` names the experiment** (#13). The commonest agent mistake is
   a mistyped study name, and the envelope answered with Python's own
   ``[Errno 2] No such file or directory: '/…/experiments/typo.json'`` — an
   absolute path, with the name the caller typed nowhere in the document.

5. **One roster** (#17). ``experiment --help`` listed ten verbs and the
   bare-``experiment`` refusal listed sixteen, from two hand-maintained
   sources with nothing relating them.
"""

import json

import pytest

from steerlab_server import cli, cli_envelope, cli_help
from steerlab_server.experiment import experiment_store, lifecycle_gates


def _run(monkeypatch, tmp_path, argv):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    return cli.main(argv)


def _document(capsys):
    return json.loads(capsys.readouterr().out)


# =============================================================================
# 1. The Mac-authority redirect
# =============================================================================


@pytest.mark.parametrize("verb", sorted(
    cli_envelope.MAC_AUTHORITY_VERBS["experiment"]))
def test_every_mac_authority_verb_answers_the_redirect(
        verb, tmp_path, monkeypatch, capsys):
    """With a positional AND without one — the no-positional form is the shape
    that used to print a usage line for a verb that does not exist."""
    for argv in (["experiment", verb, "--json"],
                 ["experiment", verb, "demo", "--json"]):
        code = _run(monkeypatch, tmp_path, argv)
        document = _document(capsys)
        assert code == 65, argv
        assert document["state"] == "refused"
        assert document["error"]["code"] == cli_envelope.MAC_AUTHORITY_CODE
        # NOT a gate: the closed vocabularies describe a study, this describes
        # the engine, and an agent switching on `error.gate` must not absorb it.
        assert "gate" not in document["error"]
        repair = document["error"]["repairAction"]
        assert repair.startswith("steerlab-cli experiment ")
        assert verb in repair


def test_the_redirect_is_a_document_even_though_the_verb_is_unrecognised(
        tmp_path, monkeypatch, capsys):
    """`--json` is normally suppressed for an unrecognised sub-verb (the
    dispatch has to print its historical roster). The redirect is the one
    exemption: it is the answer an agent most needs machine-readable."""
    assert _run(monkeypatch, tmp_path, ["experiment", "freeze", "d", "--json"]) == 65
    assert _document(capsys)["error"]["code"] == cli_envelope.MAC_AUTHORITY_CODE
    # …and an ordinary typo is NOT exempt: it keeps the roster, exit 64.
    assert _run(monkeypatch, tmp_path, ["experiment", "attch", "--json"]) == 64
    assert "verbs: " in capsys.readouterr().err


def test_an_unknown_verb_with_no_positional_no_longer_claims_to_exist(
        tmp_path, monkeypatch, capsys):
    assert _run(monkeypatch, tmp_path, ["experiment", "attch"]) == 64
    err = capsys.readouterr().err
    assert "usage: experiment attch" not in err
    assert "verbs: " in err
    # A verb this engine DOES dispatch still gets its usage line.
    assert _run(monkeypatch, tmp_path, ["experiment", "validate"]) == 64
    assert "usage: experiment validate <name>" in capsys.readouterr().err


# =============================================================================
# 2. `data check <experiment>`
# =============================================================================


def test_data_check_on_an_experiment_name_redirects_to_the_mac(
        tmp_path, monkeypatch, capsys):
    assert _run(monkeypatch, tmp_path, ["data", "check", "my-study", "--json"]) == 65
    document = _document(capsys)
    assert document["error"]["code"] == cli_envelope.MAC_AUTHORITY_CODE
    assert document["error"]["repairAction"] == "steerlab-cli data check my-study"
    # The reason still names what THIS engine's data check accepts.
    for template in cli._DATA_TEMPLATES:
        assert template in document["error"]["reason"]


# =============================================================================
# 3. Re-freeze is typed
# =============================================================================


def test_refreezing_a_frozen_manifest_is_status_immutable(tmp_path):
    root = str(tmp_path)
    experiment_store.save_raw(
        {"name": "done", "modelID": "m", "status": "frozen", "concepts": [],
         "conditions": []},
        root, freeze_transition=True)
    with pytest.raises(experiment_store.ExperimentStoreError) as caught:
        experiment_store.freeze("done", root=root)
    # Prose byte-stable; the structure is what moved.
    assert str(caught.value) == "'done' is already frozen"
    assert caught.value.gate == lifecycle_gates.STATUS_IMMUTABLE
    assert caught.value.repair_action.startswith(
        "steerlab-cli experiment duplicate done done-v2")


def test_the_validate_evidence_gate_repair_names_this_engine():
    """The gate reads evidence stamped with THIS substrate and there is no
    run-substrate seam here, so only this engine's validate can satisfy it —
    naming `steerlab-cli` (or nothing at all, as the boilerplate did) is a
    repair that cannot work."""
    repair = experiment_store._freeze_gate_repair("validateEvidence", "demo")
    assert repair.startswith("steerlab-server experiment validate demo")
    assert "python-hf-transformers" in repair
    # Authoring gates still name the Mac.
    assert experiment_store._freeze_gate_repair(
        "judgeValidity", "demo").startswith("steerlab-cli experiment pin-rubric")
    # And every gate id in the closed vocabulary has one.
    for gate in experiment_store.FORCED_GATE_IDS:
        assert experiment_store._freeze_gate_repair(gate, "demo")


# =============================================================================
# 4. `notFound` names the experiment
# =============================================================================


@pytest.mark.parametrize("path,expected", [
    ("/w/experiments/typo/experiment.json", "typo"),
    ("/w/experiments/typo.json", "typo"),
    ("/w/experiments/typo", None),
    ("/w/prompts/rubrics/missing.md", None),
    ("", None),
    (None, None),
])
def test_the_manifest_shape_is_recognised_in_both_layouts(path, expected):
    assert cli._experiment_name_in_missing_path(path) == expected


def test_a_missing_experiment_is_named_not_pathed(tmp_path, monkeypatch, capsys):
    assert _run(monkeypatch, tmp_path, ["experiment", "verify", "nope", "--json"]) == 66
    document = _document(capsys)
    assert document["state"] == "notFound"
    assert document["error"]["reason"] == (
        "experiment 'nope' not found in this workspace")
    assert "Errno" not in document["error"]["reason"]
    assert document["error"]["repairAction"].startswith(
        "steerlab-server experiment list")


# =============================================================================
# 5. One roster
# =============================================================================


def test_the_help_page_and_the_refusal_roster_name_the_same_surface():
    page = cli_help.family_text("experiment")
    declared = {spec.verb for spec in cli_envelope.VERB_SPECS
                if spec.family == "experiment"}
    # The envelope verbs are the page's table…
    assert declared <= set(cli.EXPERIMENT_VERBS), (
        "an envelope verb the refusal roster does not list: "
        f"{sorted(declared - set(cli.EXPERIMENT_VERBS))}")
    # …and every OTHER dispatched verb is named on the page too, so a caller
    # reading only `--help` cannot conclude that `rescore-style` is absent.
    for verb in cli.EXPERIMENT_VERBS:
        assert verb in page, f"{verb} is in the refusal roster but not in --help"


def test_the_untyped_repair_no_longer_claims_an_input_was_named():
    assert "read the reason and repair the named input" not in cli._UNTYPED_REPAIR
    assert cli._UNTYPED_REPAIR.startswith("this was not a typed refusal")
