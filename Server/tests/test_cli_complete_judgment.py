"""``experiment complete-judgment`` — the headless CLI intake for a deferred
evaluate (2026-08-10, the Cowork judging pilot's entry point).

The verb is a thin shell over ``tasks.complete_evaluate_judgment`` (the same
intake the HTTP route runs): it resolves the awaiting run from a directory
path or basename, loads the judging client's completed judgment file (JSON
list, ``{"judgments": [...]}`` object, or JSONL), prints the written judgment
run path on success, and surfaces the engine's refusal text on stderr with a
nonzero exit. Pin verification itself is the engine's job and is covered by
``test_evaluate_deferred.py`` — here we only prove one refusal propagates.
"""

import json
import os

from steerlab_server import cli
from steerlab_server.experiment import tasks

from test_evaluate_deferred import _fixture, _judgments_for

JUDGES = ["opus-judge", "or-judge"]


def _clear_root(monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", "placeholder")  # register restore
    monkeypatch.delenv("STEERLAB_ROOT")


def _awaiting(tmp_path):
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    return root, eval_dir


def test_cli_completes_a_deferred_evaluate_from_a_json_file(
        tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    root, eval_dir = _awaiting(tmp_path)
    judgments_file = os.path.join(root, "judgments.json")
    with open(judgments_file, "w", encoding="utf-8") as handle:
        json.dump(_judgments_for(eval_dir, judges=JUDGES), handle)

    # The awaiting run is accepted as a full directory path, not just a
    # basename (the flag is --awaiting-run <run-dir-or-basename>).
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", eval_dir,
                   "--judgments", judgments_file,
                   "--root", root])
    assert rc == 0
    out_dir = capsys.readouterr().out.strip().splitlines()[-1]
    assert os.path.isdir(out_dir)
    report = json.load(open(os.path.join(out_dir, "judge-report.json")))
    assert report["judgedOn"] == "client"
    assert report["pairs"] == 2
    assert tasks.list_awaiting_evaluate_judgment("ev", root) == []

    # Idempotent, like the intake: a second completion prints the same path.
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", os.path.basename(eval_dir),
                   "--judgments", judgments_file,
                   "--root", root])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == out_dir


def test_cli_accepts_wrapped_object_and_jsonl_shapes(
        tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    # The HTTP body shape ({"judgments": [...]}).
    root, eval_dir = _awaiting(tmp_path / "wrapped")
    wrapped = os.path.join(root, "judgments.json")
    with open(wrapped, "w", encoding="utf-8") as handle:
        json.dump({"judgments": _judgments_for(eval_dir, judges=JUDGES)},
                  handle)
    assert cli.main(["experiment", "complete-judgment", "ev",
                     "--awaiting-run", os.path.basename(eval_dir),
                     "--judgments", wrapped, "--root", root]) == 0
    capsys.readouterr()

    # One judgment object per line.
    root, eval_dir = _awaiting(tmp_path / "jsonl")
    jsonl = os.path.join(root, "judgments.jsonl")
    with open(jsonl, "w", encoding="utf-8") as handle:
        for row in _judgments_for(eval_dir, judges=JUDGES):
            handle.write(json.dumps(row) + "\n")
    assert cli.main(["experiment", "complete-judgment", "ev",
                     "--awaiting-run", os.path.basename(eval_dir),
                     "--judgments", jsonl, "--root", root]) == 0


def test_cli_surfaces_the_engines_refusal_and_exits_nonzero(
        tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    root, eval_dir = _awaiting(tmp_path)
    # Incomplete coverage — the engine's own refusal, not a CLI re-check.
    partial = _judgments_for(eval_dir, judges=JUDGES)[:-1]
    judgments_file = os.path.join(root, "judgments.json")
    with open(judgments_file, "w", encoding="utf-8") as handle:
        json.dump(partial, handle)
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", os.path.basename(eval_dir),
                   "--judgments", judgments_file, "--root", root])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err and "full coverage" in err
    # The run still awaits — nothing was half-written.
    assert [a["run"] for a in tasks.list_awaiting_evaluate_judgment(
        "ev", root)] == [os.path.basename(eval_dir)]


def test_cli_refuses_malformed_inputs_loudly(tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    root, eval_dir = _awaiting(tmp_path)

    # Missing flags are a usage error (64), matching sibling verbs.
    assert cli.main(["experiment", "complete-judgment", "ev",
                     "--awaiting-run", os.path.basename(eval_dir),
                     "--root", root]) == 64
    assert "usage: experiment complete-judgment" in capsys.readouterr().err

    # A judgments file that is not a judgment list refuses before any pin.
    empty = os.path.join(root, "empty.json")
    with open(empty, "w", encoding="utf-8") as handle:
        handle.write("[]")
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", os.path.basename(eval_dir),
                   "--judgments", empty, "--root", root])
    assert rc == 1
    assert "non-empty list" in capsys.readouterr().err

    garbage = os.path.join(root, "garbage.txt")
    with open(garbage, "w", encoding="utf-8") as handle:
        handle.write("not json at all\n")
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", os.path.basename(eval_dir),
                   "--judgments", garbage, "--root", root])
    assert rc == 1
    assert "neither JSON nor JSONL" in capsys.readouterr().err

    # A missing file is an ordinary loud failure, not a traceback.
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", os.path.basename(eval_dir),
                   "--judgments", os.path.join(root, "no-such.json"),
                   "--root", root])
    assert rc == 1
    assert "ERROR:" in capsys.readouterr().err
