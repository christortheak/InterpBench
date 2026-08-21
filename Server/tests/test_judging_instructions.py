"""``judging-instructions.md`` — the agent-facing framing as an engine
artifact (Cowork judging pipeline, 2026-08-11).

When ``evaluate`` takes the keyless custody fork, the run directory gains a
canonical, self-contained instructions file next to the blinded packets: the
pinned rubric verbatim, the intake's per-record output schema (including
``annotatorModel``), and the custody rules as binding requirements. Its
SHA-256 joins the emission record (``judging-manifest.json``), and
``complete-judgment`` intake verifies the hash the judgments CLAIM against
that stamp — a mismatch is a loud stamped warning, never a refusal
(post-submit drift policy). Contract under test: emission + hash, no
unblinding material in the instructions, claim verification/stamping on
every branch, and annotatorModel passthrough into the completed rows.
"""

import hashlib
import json
import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import tasks
from steerlab_server.experiment.judging_instructions import (
    INSTRUCTIONS_FILENAME)

from test_evaluate_deferred import RUBRIC, _fixture, _judgments_for

JUDGES = ["opus-judge", "or-judge"]


def _emit(tmp_path, **fixture_kwargs):
    root, run_dir = _fixture(tmp_path, **fixture_kwargs)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    return root, run_dir, eval_dir


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_deferred_evaluate_emits_hashed_instructions(tmp_path):
    root, _run, eval_dir = _emit(tmp_path, structured="compare severity")
    instructions_path = os.path.join(eval_dir, INSTRUCTIONS_FILENAME)
    assert os.path.exists(instructions_path)
    text = _read(instructions_path)
    with open(instructions_path, "rb") as handle:
        live = hashlib.sha256(handle.read()).hexdigest()
    jm = json.loads(_read(os.path.join(eval_dir, "judging-manifest.json")))
    # The hash joins the emission record, next to the packet/map pins.
    assert jm["instructionsFile"] == INSTRUCTIONS_FILENAME
    assert jm["instructionsSha256"] == live

    # Self-contained framing: the pinned rubric VERBATIM, the structured
    # prompt (part of the criterion), the pinned panel, and the intake
    # schema including annotatorModel.
    assert RUBRIC in text
    assert "compare severity" in text
    assert "opus-judge" in text and "or-judge" in text
    assert "annotatorModel" in text
    assert '"winner": "A" | "B" | "tie"' in text
    assert "complete-judgment" in text

    # Custody rules are REQUIREMENTS, with the why: the map is forbidden,
    # and judging must be one independent context per packet because a
    # serial context correlates errors and voids the kappa independence
    # assumption.
    assert "Never open `judging-map.json`" in text
    assert "One independent agent context per packet" in text
    assert "independence" in text and "kappa" in text
    # Packet references are RELATIVE — the file works from the run
    # directory it lives in, on any host.
    assert "`judging-packets.jsonl`" in text
    assert eval_dir not in text

    # The awaiting listing surfaces the instructions pin to clients.
    (awaiting,) = tasks.list_awaiting_evaluate_judgment("ev", root)
    assert awaiting["instructionsFile"] == INSTRUCTIONS_FILENAME
    assert awaiting["instructionsSha256"] == live


def test_instructions_carry_no_unblinding_material(tmp_path):
    # The instructions see the rubric and the panel — never the map's
    # contents. Nothing that ties a packet to its arm may appear: no
    # condition names, no orientation key, no promptIDs.
    _root, _run, eval_dir = _emit(tmp_path)
    text = _read(os.path.join(eval_dir, INSTRUCTIONS_FILENAME))
    packet_map = json.loads(
        _read(os.path.join(eval_dir, "judging-map.json")))["packets"]
    assert packet_map  # the guard below must actually guard something
    assert "baselineIsA" not in text
    for meta in packet_map.values():
        assert str(meta["condition"]) not in text      # e.g. "fear"
        assert str(meta["promptID"]) not in text


def test_completion_verifies_a_matching_instructions_claim(tmp_path):
    root, _run, eval_dir = _emit(tmp_path)
    eval_run = os.path.basename(eval_dir)
    jm = json.loads(_read(os.path.join(eval_dir, "judging-manifest.json")))
    logs = []
    out = tasks.complete_evaluate_judgment(
        "ev", eval_run, _judgments_for(eval_dir, judges=JUDGES), root=root,
        log=logs.append, instructions_sha256=jm["instructionsSha256"])
    report = json.loads(_read(os.path.join(out, "judge-report.json")))
    assert report["judgingInstructions"] == {
        "file": INSTRUCTIONS_FILENAME,
        "emittedSha256": jm["instructionsSha256"],
        "claimedSha256": jm["instructionsSha256"],
        "verified": True}
    assert not any("WARNING" in line for line in logs)


def test_mismatched_claim_stamps_loudly_but_completes(tmp_path):
    # Post-submit drift policy: verdicts already produced are evidence —
    # a wrong-instructions claim warns loudly and is stamped, never refused.
    root, _run, eval_dir = _emit(tmp_path)
    eval_run = os.path.basename(eval_dir)
    jm = json.loads(_read(os.path.join(eval_dir, "judging-manifest.json")))
    logs = []
    out = tasks.complete_evaluate_judgment(
        "ev", eval_run, _judgments_for(eval_dir, judges=JUDGES), root=root,
        log=logs.append, instructions_sha256="0" * 64)
    assert any("WARNING" in line and "DIFFERENT instructions" in line
               for line in logs)
    report = json.loads(_read(os.path.join(out, "judge-report.json")))
    stamp = report["judgingInstructions"]
    assert stamp["verified"] is False
    assert stamp["claimedSha256"] == "0" * 64
    assert stamp["emittedSha256"] == jm["instructionsSha256"]
    # The completion is otherwise whole: the run stopped awaiting.
    assert tasks.list_awaiting_evaluate_judgment("ev", root) == []


def test_absent_claim_stamps_unverified_without_warning(tmp_path):
    # The Mac app client never reads the instructions file; its completions
    # stamp claimedSha256: null with a note, not a warning.
    root, _run, eval_dir = _emit(tmp_path)
    logs = []
    out = tasks.complete_evaluate_judgment(
        "ev", os.path.basename(eval_dir),
        _judgments_for(eval_dir, judges=JUDGES), root=root, log=logs.append)
    report = json.loads(_read(os.path.join(out, "judge-report.json")))
    stamp = report["judgingInstructions"]
    assert stamp["claimedSha256"] is None and stamp["verified"] is False
    assert not any("WARNING" in line for line in logs)
    assert any("claimed no instructions hash" in line for line in logs)


def test_legacy_emission_with_no_claim_stays_byte_identical(tmp_path):
    # An awaiting run emitted before the instructions artifact existed,
    # completed by a client that claims nothing: no stamp, no chatter.
    root, _run, eval_dir = _emit(tmp_path)
    jm_path = os.path.join(eval_dir, "judging-manifest.json")
    jm = json.loads(_read(jm_path))
    del jm["instructionsFile"], jm["instructionsSha256"]
    with open(jm_path, "w", encoding="utf-8") as handle:
        json.dump(jm, handle, indent=2, sort_keys=True)
    os.remove(os.path.join(eval_dir, INSTRUCTIONS_FILENAME))
    logs = []
    out = tasks.complete_evaluate_judgment(
        "ev", os.path.basename(eval_dir),
        _judgments_for(eval_dir, judges=JUDGES), root=root, log=logs.append)
    report = json.loads(_read(os.path.join(out, "judge-report.json")))
    assert "judgingInstructions" not in report
    assert not any("instructions" in line for line in logs)
    # ...but a claim against a legacy emission is recorded unverified.
    root2, _run2, eval_dir2 = _emit(tmp_path / "two")
    jm2_path = os.path.join(eval_dir2, "judging-manifest.json")
    jm2 = json.loads(_read(jm2_path))
    del jm2["instructionsFile"], jm2["instructionsSha256"]
    with open(jm2_path, "w", encoding="utf-8") as handle:
        json.dump(jm2, handle, indent=2, sort_keys=True)
    logs2 = []
    out2 = tasks.complete_evaluate_judgment(
        "ev", os.path.basename(eval_dir2),
        _judgments_for(eval_dir2, judges=JUDGES), root=root2,
        log=logs2.append, instructions_sha256="1" * 64)
    report2 = json.loads(_read(os.path.join(out2, "judge-report.json")))
    assert report2["judgingInstructions"] == {
        "file": INSTRUCTIONS_FILENAME, "emittedSha256": None,
        "claimedSha256": "1" * 64, "verified": False}
    assert any("WARNING" in line and "legacy emission" in line
               for line in logs2)


def test_annotator_model_lands_in_completed_rows(tmp_path):
    root, _run, eval_dir = _emit(tmp_path)
    eval_run = os.path.basename(eval_dir)
    # An empty claim is refused: a recorded field is provenance only if it
    # says something. (Checked BEFORE the successful completion — the verb
    # is idempotent afterwards and would return the finished run.)
    bad = _judgments_for(eval_dir, judges=JUDGES)
    bad[0]["annotatorModel"] = "  "
    with pytest.raises(ValueError, match="annotatorModel"):
        tasks.complete_evaluate_judgment("ev", eval_run, bad, root=root,
                                         log=lambda *_: None)
    judgments = _judgments_for(eval_dir, judges=JUDGES)
    for row in judgments:
        if row["judge"] == "opus-judge":
            row["annotatorModel"] = "claude-fable-5"
    out = tasks.complete_evaluate_judgment(
        "ev", eval_run, judgments, root=root, log=lambda *_: None)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "judgments.jsonl"), encoding="utf-8")]
    assert {r["annotatorModel"] for r in rows if r["judge"] == "opus-judge"} \
        == {"claude-fable-5"}
    # Rows that claimed nothing carry nothing — absence stays legible.
    assert all("annotatorModel" not in r for r in rows
               if r["judge"] == "or-judge")


def test_cli_wrapper_claim_reaches_the_intake(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", "placeholder")  # register restore
    monkeypatch.delenv("STEERLAB_ROOT")
    root, _run, eval_dir = _emit(tmp_path)
    jm = json.loads(_read(os.path.join(eval_dir, "judging-manifest.json")))
    judgments_file = os.path.join(root, "judgments.json")
    with open(judgments_file, "w", encoding="utf-8") as handle:
        json.dump({"instructionsSha256": jm["instructionsSha256"],
                   "judgments": _judgments_for(eval_dir, judges=JUDGES)},
                  handle)
    rc = cli.main(["experiment", "complete-judgment", "ev",
                   "--awaiting-run", os.path.basename(eval_dir),
                   "--judgments", judgments_file, "--root", root])
    assert rc == 0
    out_dir = capsys.readouterr().out.strip().splitlines()[-1]
    report = json.loads(_read(os.path.join(out_dir, "judge-report.json")))
    assert report["judgingInstructions"]["verified"] is True
