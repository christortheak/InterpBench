"""Exit codes of the ``steerlab-server optvec`` family — the shared vocabulary
(``cli_envelope.STATE_EXIT_CODES``) applied to a family that used to answer
every typed error with a 2 and let the untyped ones escape as tracebacks.

Two layers, as in ``test_cli_envelope.py``:

1. **The classifier**, pinned exception class by exception class against the
   state it must mean — plus a drift gate that every error class an optvec
   module defines is classified, so a class added later cannot land at 70 by
   omission.
2. **The verbs**, driven through ``cli.main`` on hand-built inputs so each
   code is reached the way a caller reaches it: a config that names no file,
   a config that breaks its contract, an artifact with no optvec block, a
   lens that was never imported, a survey with no such item, an exception
   nobody typed. No model is loaded anywhere here: every pinned path is
   reached before the load, or through the loader seam the toy model fills.

The contract, stated once: 0 done · 64 usage or a malformed config · 65 a
typed refusal by an input · 66 a named file, artifact, lens, dataset, survey
or campaign directory that does not exist · 70 an operational failure (with
its traceback) · and ``campaign submit``'s historical 3 for a fan-out failure,
which this change deliberately leaves alone.
"""

import glob
import json
import os

import pytest

from steerlab_server import cli, cli_envelope
from steerlab_server.experiment import (optvec_campaign, optvec_eval,
                                        optvec_geometry, optvec_gradient,
                                        optvec_interpret, optvec_jspace,
                                        optvec_train, sweep_selection)
from tests.test_optvec_campaign import FakeRunner, _campaign_payload
from tests.test_optvec_eval import (HIDDEN, LAYERS, OPTVEC_LAYER, _eval_config,
                                    _optvec_direction, write_artifact,
                                    write_optvec_artifact)
from tests.test_optvec_interpret import _config as interpret_config
from tests.test_optvec_interpret import _entry, _interpret_fixture, _tokens
from tests.test_optvec_jspace import _config as jspace_config
from tests.test_optvec_jspace import _write_lens

OPTVEC_MODULES = (optvec_train, optvec_eval, optvec_geometry, optvec_campaign,
                  optvec_interpret, optvec_jspace, optvec_gradient)

#: Every config-contract class → ``blocked`` (64).
CONFIG_ERRORS = (
    optvec_train.OptVecConfigError,
    optvec_eval.OptVecEvalConfigError,
    optvec_geometry.OptVecGeometryConfigError,
    optvec_campaign.CampaignConfigError,
    optvec_interpret.OptVecInterpretConfigError,
    optvec_interpret.OptVecFamilyConfigError,
    optvec_jspace.OptVecJSpaceConfigError,
    optvec_gradient.OptVecGradientConfigError,
)

#: Every input-declined class → ``refused`` (65).
REFUSALS = (
    optvec_train.OptVecDataError,
    optvec_eval.OptVecEvalDataError,
    optvec_interpret.OptVecInterpretDataError,
    optvec_eval.OptVecArtifactError,
    optvec_geometry.OptVecGeometryError,
    optvec_campaign.CampaignError,
    optvec_interpret.OptVecFamilyError,
    optvec_jspace.OptVecJSpaceError,
    optvec_gradient.OptVecGradientDataError,
)

#: The verbs that take ``--config <path.json>`` and nothing else.
CONFIG_VERBS = ("train", "eval", "interpret", "family", "jspace", "gradient",
                "fracture")


def _write(path, payload) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle)
    return str(path)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return tmp_path


# =============================================================================
# 1. The classifier
# =============================================================================


@pytest.mark.parametrize("cls", CONFIG_ERRORS)
def test_a_config_contract_error_is_blocked_64(cls):
    document = cli._optvec_exception_envelope(
        "optvec x", cls("unknown key: nonsense"), config_path="/w/c.json")
    assert document.state == "blocked"
    assert document.exit_code == 64 == cli_envelope.exit_code_for("blocked")
    assert document.error["code"] == "malformedConfig"
    assert "/w/c.json" in document.error["repairAction"]
    assert document.error["reason"] == "unknown key: nonsense"


@pytest.mark.parametrize("cls", REFUSALS)
def test_a_typed_refusal_is_refused_65_and_names_its_class(cls):
    document = cli._optvec_exception_envelope("optvec x", cls("declined"))
    assert document.state == "refused"
    assert document.exit_code == 65 == cli_envelope.exit_code_for("refused")
    assert document.error["code"] == cls.__name__
    assert "gate" not in document.error       # not a lifecycle gate
    assert document.error["repairAction"]


def test_a_json_file_that_is_not_json_is_a_malformed_config():
    try:
        json.loads("nope")
    except json.JSONDecodeError as exc:
        broken = exc
    document = cli._optvec_exception_envelope("optvec x", broken,
                                              config_path="/w/c.json")
    assert document.state == "blocked" and document.exit_code == 64


def test_a_missing_file_is_not_found_66_wherever_it_sits_in_the_chain():
    """Direct, one wrap deep (the artifact loader's shape), two deep (the
    interpret loader re-wrapping the eval loader), and implicit context."""
    direct = FileNotFoundError(2, "No such file or directory", "/w/v.json")

    one_deep = optvec_eval.OptVecArtifactError("could not be loaded")
    one_deep.__cause__ = direct

    middle = optvec_eval.OptVecEvalDataError("neutral text file not found")
    middle.__cause__ = direct
    two_deep = optvec_interpret.OptVecInterpretDataError("same")
    two_deep.__cause__ = middle

    try:
        try:
            raise direct
        except FileNotFoundError:
            raise optvec_campaign.CampaignError("implicit context")
    except optvec_campaign.CampaignError as exc:
        contextual = exc

    for exc in (direct, one_deep, two_deep, contextual):
        document = cli._optvec_exception_envelope("optvec x", exc)
        assert document.state == "notFound", exc
        assert document.exit_code == 66 == cli_envelope.exit_code_for("notFound")
        assert document.error["code"] == "notFound"
        assert "/w/v.json" in document.error["repairAction"]

    # A suppressed context (`raise … from None`) is NOT followed: the author
    # said the earlier failure is not the cause.
    try:
        try:
            raise direct
        except FileNotFoundError:
            raise optvec_campaign.CampaignError("unrelated") from None
    except optvec_campaign.CampaignError as exc:
        suppressed = exc
    assert cli._optvec_exception_envelope("optvec x", suppressed).state == "refused"


def test_a_lifecycle_refusal_keeps_its_gate_ahead_of_everything():
    """The dataset loaders raise the sweep-selection gate; the same exception
    must read the same way here as under `experiment sweep` — even when its
    message says "not found"."""
    with pytest.raises(ValueError) as caught:
        sweep_selection.load_choice_rows("/w/nope.jsonl", "targetTrain")
    document = cli._optvec_exception_envelope("optvec train", caught.value)
    assert document.state == "refused" and document.exit_code == 65
    assert document.error["gate"] == "sweepSelectionRule"
    assert document.error["code"] == "sweepSelectionRule"


def test_an_untyped_exception_is_failed_70():
    for exc in (RuntimeError("boom"), KeyError("x"), ZeroDivisionError(),
                PermissionError(13, "Permission denied", "/w/c.json"),
                IsADirectoryError(21, "Is a directory", "/w")):
        document = cli._optvec_exception_envelope("optvec x", exc)
        assert document.state == "failed", exc
        assert document.exit_code == 70 == cli_envelope.exit_code_for("failed")
        assert document.error["code"] == "verbFailed"
        assert document.error["repairAction"] == cli._UNTYPED_REPAIR


def test_every_optvec_error_class_is_classified():
    """The drift gate: an exception class defined in an optvec module that is
    in neither tuple would fall to 70 by omission, silently."""
    malformed, refusals = cli._optvec_error_classes()
    for module in OPTVEC_MODULES:
        for name, obj in vars(module).items():
            if not (isinstance(obj, type) and issubclass(obj, Exception)):
                continue
            if obj.__module__ != module.__name__:
                continue
            assert issubclass(obj, malformed + refusals), (
                f"{module.__name__}.{name} is not classified")
    # And the split is the one this file states, not an accident.
    assert set(malformed) == set(CONFIG_ERRORS) | {json.JSONDecodeError}
    assert set(refusals) == (set(REFUSALS)
                             - {optvec_interpret.OptVecInterpretDataError})
    # A config refinement is decided BEFORE its refusal base class.
    for refined in (optvec_geometry.OptVecGeometryConfigError,
                    optvec_interpret.OptVecFamilyConfigError):
        assert cli._optvec_exception_envelope(
            "optvec x", refined("rule")).state == "blocked"


def test_exit_reports_reason_and_repair_and_a_traceback_only_on_failure(capsys):
    assert cli._optvec_exit("optvec x",
                            optvec_eval.OptVecArtifactError("no block")) == 65
    err = capsys.readouterr().err
    assert err.startswith("optvec x: no block\n  ")
    assert "Traceback" not in err

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        assert cli._optvec_exit("optvec x", exc) == 70
    err = capsys.readouterr().err
    assert err.startswith("optvec x: boom\n  ")
    assert "Traceback (most recent call last)" in err and "boom" in err


# =============================================================================
# 2. The verbs — usage, missing config, malformed config, for the family
# =============================================================================


def test_usage_is_64_at_every_dispatch_site(root, capsys):
    for argv in (["optvec"], ["optvec", "bogus"], ["optvec", "campaign"],
                 ["optvec", "campaign", "submit"],
                 ["optvec", "campaign", "status"],
                 ["optvec", "gradient", "mint", "only-one"],
                 *[["optvec", verb] for verb in CONFIG_VERBS]):
        assert cli.main(argv) == 64, argv
        assert "usage: steerlab-server optvec" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    *[["optvec", verb, "--config"] for verb in CONFIG_VERBS],
    ["optvec", "geometry", "--config"],
    ["optvec", "campaign", "materialize", "--config"],
])
def test_a_config_that_names_no_file_is_66(root, capsys, argv):
    missing = str(root / "missing.json")
    assert cli.main(argv + [missing]) == 66
    err = capsys.readouterr().err
    assert err.startswith(f"{argv[0]} {argv[1]}: ")
    assert f"no file at {missing}" in err


@pytest.mark.parametrize("argv", [
    *[["optvec", verb, "--config"] for verb in CONFIG_VERBS],
    ["optvec", "geometry", "--config"],
    ["optvec", "campaign", "materialize", "--config"],
])
def test_a_config_that_breaks_its_contract_is_64(root, capsys, argv):
    bad = _write(root / "bad.json", {"nonsense": 1})
    assert cli.main(argv + [bad]) == 64
    err = capsys.readouterr().err
    assert "unknown" in err and "nonsense" in err
    assert f"fix the config at {bad}" in err
    # Nothing ran: no run directory was minted for a config that never parsed.
    assert not os.path.isdir(root / "runs") or os.listdir(root / "runs") == []


def test_a_config_that_is_not_json_is_64(root, capsys):
    broken = _write(root / "broken.json", "{not json")
    assert cli.main(["optvec", "fracture", "--config", broken]) == 64
    assert "optvec fracture: " in capsys.readouterr().err


def test_a_config_path_that_is_a_directory_is_an_operational_failure(root,
                                                                    capsys):
    os.makedirs(root / "adir")
    assert cli.main(["optvec", "fracture", "--config", str(root / "adir")]) == 70
    err = capsys.readouterr().err
    assert "Is a directory" in err and "Traceback" in err


# =============================================================================
# 3. The verbs — refusals and not-founds, one input at a time
# =============================================================================


def _interpret_payload(artifact, root) -> dict:
    """The interpret config as JSON. Its serializer writes every optional stage,
    absent ones as ``null``, and its parser wants absent ones absent."""
    payload = interpret_config(artifact, root).to_dict()
    return {key: value for key, value in payload.items() if value is not None}


def _plain_artifact(root):
    """A CAA-style artifact with a direction but no optvec block."""
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[OPTVEC_LAYER] = _optvec_direction()
    return write_artifact(root / "lib", "plain", per_layer=per_layer,
                          optvec=None, extraction_method="meanDifference")


def test_eval_refuses_a_plain_artifact_and_cannot_find_a_missing_one(root,
                                                                      capsys):
    plain = _plain_artifact(root)
    config = _write(root / "eval-plain.json",
                    _eval_config(root, plain).to_dict())
    assert cli.main(["optvec", "eval", "--config", config]) == 65
    err = capsys.readouterr().err
    assert "optvec eval: " in err and "meanDifference" in err

    payload = _eval_config(root, str(root / "lib" / "nope")).to_dict()
    config = _write(root / "eval-missing.json", payload)
    assert cli.main(["optvec", "eval", "--config", config]) == 66
    assert "could not be loaded" in capsys.readouterr().err


def test_eval_refuses_dataset_hash_drift(root, capsys):
    reference = write_optvec_artifact(root / "v", "toy")
    payload = _eval_config(root, reference).to_dict()
    config = _write(root / "eval.json", payload)
    # The pinned test split changes after the config was written.
    with open(payload["datasets"]["targetTest"]["path"], "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "late", "prompt": "the ruling is",
                                 "options": ["alpha", "beta"],
                                 "target": "alpha"}) + "\n")
    assert cli.main(["optvec", "eval", "--config", config]) == 65
    err = capsys.readouterr().err
    assert "pins" in err and "OptVecEvalDataError" not in err  # class is in the envelope, prose stays prose


def test_interpret_refuses_a_plain_artifact_and_cannot_find_a_missing_one(
        root, capsys):
    plain = _plain_artifact(root)
    config = _write(root / "interpret-plain.json",
                    _interpret_payload(plain, root))
    assert cli.main(["optvec", "interpret", "--config", config]) == 65
    assert "meanDifference" in capsys.readouterr().err

    config = _write(root / "interpret-missing.json",
                    _interpret_payload(str(root / "lib" / "nope"), root))
    assert cli.main(["optvec", "interpret", "--config", config]) == 66
    assert "could not be loaded" in capsys.readouterr().err


def test_family_distinguishes_too_few_runs_a_missing_run_and_a_wrong_run(
        root, capsys):
    a = _interpret_fixture(root, "a", condition="s2",
                           library=[_entry("lib/anger", 0.8, 100.0)],
                           promoted=_tokens([1.0]))
    # One run is a config that breaks its own contract.
    one = _write(root / "one.json", {"interpretRuns": [a]})
    assert cli.main(["optvec", "family", "--config", one]) == 64
    assert "at least 2" in capsys.readouterr().err

    # A run directory that does not exist is notFound.
    missing = _write(root / "missing-run.json",
                     {"interpretRuns": [a, str(root / "runs" / "nowhere")]})
    assert cli.main(["optvec", "family", "--config", missing]) == 66
    assert "interpret.json" in capsys.readouterr().err

    # A run of the wrong kind exists and declines.
    other = root / "runs" / "not-an-interpret"
    os.makedirs(other)
    _write(other / optvec_interpret.INTERPRET_JSON, {"runType": "optvec-eval"})
    wrong = _write(root / "wrong-run.json",
                   {"interpretRuns": [a, str(other)]})
    assert cli.main(["optvec", "family", "--config", wrong]) == 65
    assert "optvec-eval" in capsys.readouterr().err


def test_jspace_cannot_find_a_missing_lens_or_artifact_and_refuses_a_bad_layer(
        root, capsys):
    reference = write_optvec_artifact(root / "v", "c")
    config = _write(root / "jspace.json",
                    jspace_config(root, [reference]).to_dict())
    # No lens imported into this workspace: notFound, naming both steps.
    assert cli.main(["optvec", "jspace", "--config", config]) == 66
    err = capsys.readouterr().err
    assert "jlens acquire" in err and "jlens import" in err

    missing = _write(root / "jspace-missing.json",
                     jspace_config(root, [str(root / "v" / "nope")]).to_dict())
    assert cli.main(["optvec", "jspace", "--config", missing]) == 66
    assert "could not be loaded" in capsys.readouterr().err

    # The lens exists (fitted 0..2, target 3) but the vector sits at layer 3:
    # a well-formed request the lens declines.
    _write_lens(root, root=str(root))
    top = write_optvec_artifact(root / "v", "top", layer=LAYERS - 1)
    off = _write(root / "jspace-off.json",
                 jspace_config(root, [top]).to_dict())
    assert cli.main(["optvec", "jspace", "--config", off]) == 65
    assert "not a fitted source layer" in capsys.readouterr().err


def test_gradient_mint_cannot_find_a_survey_and_refuses_an_unknown_item(
        root, monkeypatch, capsys):
    from tests.test_optvec_gradient import _run_survey

    assert cli.main(["optvec", "gradient", "mint", str(root / "nowhere"),
                     "g-0"]) == 66
    assert "gradient-survey.json" in capsys.readouterr().err

    _model, _config, result = _run_survey(root, monkeypatch)
    assert cli.main(["optvec", "gradient", "mint", result["runDirectory"],
                     "not-an-item"]) == 65
    assert "not-an-item" in capsys.readouterr().err


def test_train_refuses_through_the_loader_seam(root, monkeypatch, capsys):
    """Train loads the model BEFORE it reads a dataset, so its refusals are
    reached with the toy model in the loader's place — the same end-to-end
    path a real run takes, minus the weights."""
    from steerlab_server.steering import model_loader
    from tests.test_optvec_train import _s1_config, _tiny_steered_model

    monkeypatch.setattr(model_loader, "load",
                        lambda *args, **kwargs: _tiny_steered_model())
    payload = _s1_config(root, dtype="float32", device="cpu").to_dict()
    config = _write(root / "train.json", payload)

    # Hash drift on the pinned train split: a typed data refusal.
    target = payload["datasets"]["targetTrain"]["path"]
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "late", "prompt": "x the ruling is",
                                 "options": ["alpha", "beta"],
                                 "target": "alpha"}) + "\n")
    assert cli.main(["optvec", "train", "--config", config]) == 65
    assert "pins" in capsys.readouterr().err

    # The split file gone entirely: the dataset loader's lifecycle gate — the
    # exception `experiment sweep` raises for the same file, read the same way.
    os.remove(target)
    assert cli.main(["optvec", "train", "--config", config]) == 65
    assert "file not found" in capsys.readouterr().err


def test_geometry_codes_and_the_unchanged_success_path(root, capsys):
    a = write_optvec_artifact(root / "v", "a", layer=1)
    b = write_optvec_artifact(root / "v", "b", layer=2)
    c = write_optvec_artifact(root / "v", "c", layer=1)

    assert cli.main(["optvec", "geometry", a]) == 64
    assert "at least 2" in capsys.readouterr().err
    assert cli.main(["optvec", "geometry", "--layer", "x", a, c]) == 64
    assert "must be an integer" in capsys.readouterr().err

    assert cli.main(["optvec", "geometry", a, b]) == 65
    assert "different optvec layers" in capsys.readouterr().err

    assert cli.main(["optvec", "geometry", a, str(root / "v" / "zzz")]) == 66
    assert "could not be loaded" in capsys.readouterr().err

    assert cli.main(["optvec", "geometry", a, c]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["count"] == 2 and os.path.isdir(report["runDirectory"])


def test_campaign_codes_and_the_unchanged_submit_three(root, monkeypatch,
                                                       capsys):
    runner = FakeRunner(job_ids=[None, "12", "13", "14"])
    monkeypatch.setattr(optvec_campaign, "SubprocessRunner", lambda: runner)

    config = _write(root / "campaign.json", _campaign_payload(root))
    assert cli.main(["optvec", "campaign", "materialize", "--config",
                     config]) == 0
    campaign_dir = json.loads(capsys.readouterr().out)["campaignDirectory"]

    assert cli.main(["optvec", "campaign", "status", campaign_dir]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]

    nowhere = str(root / "nowhere")
    assert cli.main(["optvec", "campaign", "status", nowhere]) == 66
    assert "not a materialized campaign" in capsys.readouterr().err
    assert cli.main(["optvec", "campaign", "submit", nowhere]) == 66
    capsys.readouterr()

    # A fan-out failure is still the historical 3: the report is on stdout
    # and the failed cell is in it.
    assert cli.main(["optvec", "campaign", "submit", campaign_dir]) == 3
    report = json.loads(capsys.readouterr().out)
    assert [f["cellID"] for f in report["failed"]] == ["s0-L30-s0"]

    # A campaign whose cells lost their scripts declines to submit them — on
    # a fresh campaign, because the one above has already filled its queue
    # and a full queue never reaches a cell.
    assert cli.main(["optvec", "campaign", "materialize", "--config",
                     config]) == 0
    fresh = json.loads(capsys.readouterr().out)["campaignDirectory"]
    scripts = glob.glob(os.path.join(fresh, "cells", "*",
                                     optvec_campaign.CELL_SCRIPT_FILENAME))
    assert scripts
    for script in scripts:
        os.remove(script)
    assert cli.main(["optvec", "campaign", "submit", fresh]) == 65
    assert "materialize the campaign before submitting" in capsys.readouterr().err


# =============================================================================
# 4. The verbs — an exception nobody typed is 70, with its traceback
# =============================================================================


def _boom(*args, **kwargs):
    raise RuntimeError("boom from the verb body")


@pytest.mark.parametrize("argv,target", [
    (["optvec", "train", "--config", "c.json"], (optvec_train, "load_config")),
    (["optvec", "eval", "--config", "c.json"], (optvec_eval, "load_config")),
    (["optvec", "interpret", "--config", "c.json"],
     (optvec_interpret, "load_config")),
    (["optvec", "jspace", "--config", "c.json"], (optvec_jspace, "load_config")),
    (["optvec", "gradient", "--config", "c.json"],
     (optvec_gradient, "load_config")),
    (["optvec", "gradient", "mint", "survey", "item"],
     (optvec_gradient, "mint")),
    (["optvec", "fracture", "--config", "c.json"],
     (optvec_geometry, "load_fracture_config")),
    (["optvec", "geometry", "--config", "c.json"],
     (optvec_geometry, "load_config")),
    (["optvec", "geometry", "one", "two"], (optvec_geometry, "geometry")),
    (["optvec", "campaign", "materialize", "--config", "c.json"],
     (optvec_campaign, "load_config")),
    (["optvec", "campaign", "submit", "dir"], (optvec_campaign, "submit")),
    (["optvec", "campaign", "status", "dir"], (optvec_campaign, "status")),
])
def test_an_untyped_exception_is_70_with_its_traceback(root, monkeypatch,
                                                       capsys, argv, target):
    module, name = target
    monkeypatch.setattr(module, name, _boom)
    assert cli.main(argv) == 70
    err = capsys.readouterr().err
    assert err.startswith(f"optvec {argv[1]}: boom from the verb body\n  ")
    assert "Traceback (most recent call last)" in err


def test_family_untyped_exception_is_70(root, monkeypatch, capsys):
    monkeypatch.setattr(optvec_interpret.FamilySummaryConfig, "from_dict",
                        classmethod(lambda cls, payload: _boom()))
    config = _write(root / "family.json", {"interpretRuns": ["a", "b"]})
    assert cli.main(["optvec", "family", "--config", config]) == 70
    assert "Traceback" in capsys.readouterr().err
