"""OptVec dataset-bundle readiness (the server side of the ``data check``
layer): the nine bundle files, strict choice-row parsing through the same
loader the sweep/train paths refuse on, the tokenizer-free single-character
option check, the 45–55% A/B balance window, bundle-wide id uniqueness, and
per-file SHA-256 emission for valid files only. Plus the
``steerlab-server data check optvec`` CLI verb (blockers-first lines, exit 2
on any blocker). Purely file-driven — no model, no tokenizer."""

import hashlib
import json
import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import data_readiness


def _row(row_id, target="A", options=("A", "B")):
    return {"id": row_id, "text": f"Question for {row_id}?\nA. yes\nB. no\n"
            "Answer with exactly one letter: A or B.",
            "options": list(options), "target": target}


def _write_choice(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _balanced_rows(prefix, n=10):
    """n rows, ids ``<prefix>-NNN``, targets alternating A/B (50/50)."""
    return [_row(f"{prefix}-{i:03d}", target=("A" if i % 2 == 0 else "B"))
            for i in range(n)]


def _write_neutral(path, texts=("The lake froze early this year.",
                                "Bread rises faster in a warm kitchen.")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}) + "\n")


def _write_bundle_json(directory, mutate=None):
    """A spec-shaped bundle.json pinning the current bytes of all nine files;
    ``mutate`` may edit the payload dict before writing."""
    files = {}
    for name in data_readiness.BUNDLE_FILES:
        with open(os.path.join(directory, name), "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        files[name.removesuffix(".jsonl")] = {"path": name, "sha256": digest}
    payload = {"bundle": "test-1",
               "targetIssue": "rule vs equity; shift toward equity",
               "shiftDirection": "toward equity",
               "caseFamilies": ["filing-deadline"],
               "anchorIssues": ["settled-evidence"],
               "files": files}
    if mutate is not None:
        mutate(payload)
    with open(os.path.join(directory, data_readiness.BUNDLE_JSON), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle)


def _write_bundle(directory):
    """A fully valid bundle (nine data files + bundle.json + REPORT.md);
    returns the directory."""
    for name in data_readiness.CHOICE_FILES:
        prefix = f"ovt-{name.removesuffix('.jsonl')}"
        _write_choice(os.path.join(directory, name), _balanced_rows(prefix))
    _write_neutral(os.path.join(directory, data_readiness.NEUTRAL_FILE))
    _write_bundle_json(directory)
    with open(os.path.join(directory, data_readiness.REPORT_FILE), "w",
              encoding="utf-8") as handle:
        handle.write("QC: no cross-split leakage found.\n")
    return directory


def _by_name(report):
    return {r.name: r for r in report.requirements}


# --- the template itself --------------------------------------------------------


def test_ready_bundle_is_all_present_with_hashes(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    report = data_readiness.check_optvec_bundle(bundle)

    assert report.ready and not report.blockers
    # Nine files + bundle.json + REPORT.md + the bundle-ids requirement.
    assert len(report.requirements) == len(data_readiness.BUNDLE_FILES) + 3
    reqs = _by_name(report)
    for name in data_readiness.BUNDLE_FILES:
        assert reqs[name].status == "present"
        # The emitted hash is exactly the file's raw-byte SHA-256 — the value
        # the researcher pastes into train/eval configs.
        with open(os.path.join(bundle, name), "rb") as handle:
            assert reqs[name].sha256 == hashlib.sha256(handle.read()).hexdigest()
    ids_req = reqs[data_readiness.BUNDLE_IDS_REQUIREMENT]
    assert ids_req.status == "present"
    assert "80 id(s)" in ids_req.detail  # 8 choice files × 10 rows


def test_missing_file_blocks_and_downgrades_id_check(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    os.remove(os.path.join(bundle, "anchor-test.jsonl"))
    report = data_readiness.check_optvec_bundle(bundle)

    reqs = _by_name(report)
    assert reqs["anchor-test.jsonl"].status == "missing"
    assert data_readiness.AUTHORING_SPEC in reqs["anchor-test.jsonl"].detail
    assert not report.ready
    # Id uniqueness cannot claim the full bundle — partial, not present.
    ids_req = reqs[data_readiness.BUNDLE_IDS_REQUIREMENT]
    assert ids_req.status == "partial"
    assert "7 of 8" in ids_req.detail
    # Partial does not block on its own; the missing file already does.
    assert [r.name for r in report.blockers] == ["anchor-test.jsonl"]


def test_malformed_row_is_invalid_via_strict_loader(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    path = os.path.join(bundle, "target-val.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    report = data_readiness.check_optvec_bundle(bundle)

    req = _by_name(report)["target-val.jsonl"]
    assert req.status == "invalid"
    assert "line 11" in req.detail and "not valid JSON" in req.detail
    # No paste-able hash for a file the engine would refuse.
    assert req.sha256 is None


def test_within_file_duplicate_id_is_invalid(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    rows = _balanced_rows("ovt-dup")
    rows[5]["id"] = rows[4]["id"]
    _write_choice(os.path.join(bundle, "capability-train.jsonl"), rows)
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)["capability-train.jsonl"]
    assert req.status == "invalid"
    assert "duplicate item id" in req.detail


def test_multi_character_option_is_invalid(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    rows = _balanced_rows("ovt-opt")
    rows[3]["options"] = ["(A)", "B"]
    rows[3]["target"] = "(A)"
    _write_choice(os.path.join(bundle, "anchor-train.jsonl"), rows)
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)["anchor-train.jsonl"]
    assert req.status == "invalid"
    assert "single character" in req.detail and "'(A)'" in req.detail


def test_target_imbalance_is_invalid(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    # 8 of 10 rows target A → 80%, far outside 45–55%.
    rows = [_row(f"ovt-skew-{i:03d}", target=("A" if i < 8 else "B"))
            for i in range(10)]
    _write_choice(os.path.join(bundle, "target-train.jsonl"), rows)
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)["target-train.jsonl"]
    assert req.status == "invalid"
    assert "80.0%" in req.detail and "45%–55%" in req.detail


def test_balance_window_edges_pass(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    # Exactly 45% (9/20) and exactly 55% (11/20) are inside the window.
    for name, a_count in (("target-train.jsonl", 9),
                          ("target-val.jsonl", 11)):
        rows = [_row(f"ovt-{name}-{i:03d}",
                     target=("A" if i < a_count else "B")) for i in range(20)]
        _write_choice(os.path.join(bundle, name), rows)
    report = data_readiness.check_optvec_bundle(bundle)
    reqs = _by_name(report)
    assert reqs["target-train.jsonl"].status == "present"
    assert reqs["target-val.jsonl"].status == "present"


def test_cross_file_duplicate_ids_block_the_bundle(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    # target-test reuses two ids from target-train: each file is internally
    # valid, but the engine keys baselines by id bundle-wide.
    train_rows = _balanced_rows("ovt-target-train")
    _write_choice(os.path.join(bundle, "target-train.jsonl"), train_rows)
    test_rows = _balanced_rows("ovt-target-test")
    test_rows[0]["id"] = train_rows[0]["id"]
    test_rows[1]["id"] = train_rows[1]["id"]
    _write_choice(os.path.join(bundle, "target-test.jsonl"), test_rows)

    report = data_readiness.check_optvec_bundle(bundle)
    reqs = _by_name(report)
    assert reqs["target-train.jsonl"].status == "present"
    assert reqs["target-test.jsonl"].status == "present"
    ids_req = reqs[data_readiness.BUNDLE_IDS_REQUIREMENT]
    assert ids_req.status == "invalid"
    assert "2 id(s) duplicated" in ids_req.detail
    assert train_rows[0]["id"] in ids_req.detail
    assert "target-train.jsonl and target-test.jsonl" in ids_req.detail
    assert not report.ready


def test_neutral_fluency_is_spec_strict(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    neutral = os.path.join(bundle, data_readiness.NEUTRAL_FILE)

    # A plain-text line: the EVAL loader would accept it, but the bundle spec
    # pins one {"text": …} object per line — readiness refuses.
    with open(neutral, "a", encoding="utf-8") as handle:
        handle.write("just some prose\n")
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)[data_readiness.NEUTRAL_FILE]
    assert req.status == "invalid" and "line 3" in req.detail

    # A JSON object without a string text refuses too.
    _write_neutral(neutral)
    with open(neutral, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": 7}) + "\n")
    req = _by_name(data_readiness.check_optvec_bundle(bundle))[
        data_readiness.NEUTRAL_FILE]
    assert req.status == "invalid"

    # An empty file has no texts for the fluency guard.
    with open(neutral, "w", encoding="utf-8"):
        pass
    req = _by_name(data_readiness.check_optvec_bundle(bundle))[
        data_readiness.NEUTRAL_FILE]
    assert req.status == "invalid" and "zero texts" in req.detail


def test_requirements_sort_blockers_first(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    with open(os.path.join(bundle, "target-train.jsonl"), "a",
              encoding="utf-8") as handle:
        handle.write("{bad\n")
    # Re-pin AFTER the mutation so bundle.json itself stays clean (its hash
    # check pins bytes, not validity), then remove a file: the missing file's
    # own line blocks, and bundle.json skips hashing what does not exist.
    _write_bundle_json(bundle)
    os.remove(os.path.join(bundle, "anchor-val.jsonl"))
    report = data_readiness.check_optvec_bundle(bundle)
    statuses = [r.status for r in report.requirements]
    order = {"invalid": 0, "missing": 1, "partial": 2, "present": 3}
    assert statuses == sorted(statuses, key=order.__getitem__)
    assert statuses[0] == "invalid" and statuses[1] == "missing"


def test_missing_bundle_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError, match="bundle directory not found"):
        data_readiness.check_optvec_bundle(str(tmp_path / "absent"))


# --- CLI verb -------------------------------------------------------------------


def _run_cli(monkeypatch, tmp_path, argv):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    return cli.main(argv)


def test_cli_usage(tmp_path, monkeypatch, capsys):
    for argv in (["data"], ["data", "check"], ["data", "check", "other"],
                 ["data", "list"]):
        assert _run_cli(monkeypatch, tmp_path, argv) == 64
        assert "data check optvec" in capsys.readouterr().err


def test_cli_ready_bundle_exits_zero_and_emits_hashes(
        tmp_path, monkeypatch, capsys):
    _write_bundle(str(tmp_path / "prompts" / "optvec"))
    rc = _run_cli(monkeypatch, tmp_path, ["data", "check", "optvec"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ready" in out and "NOT ready" not in out
    # One paste-able hash per bundle file.
    assert out.count("sha256 ") == len(data_readiness.BUNDLE_FILES)
    assert "present  bundle ids" in out


def test_cli_default_directory_is_prompts_optvec(tmp_path, monkeypatch, capsys):
    # No bundle anywhere → the default prompts/optvec is named in the error.
    rc = _run_cli(monkeypatch, tmp_path, ["data", "check", "optvec"])
    err = capsys.readouterr().err
    assert rc == 2
    assert os.path.join("prompts", "optvec") in err


def test_cli_blockers_exit_65_blockers_first(tmp_path, monkeypatch, capsys):
    """WP0 step 8: the ONE human-mode exit migration the audit schedules by
    name (§2.3's stated migration debt, §7 row 7: "`data check` blockers
    (2 → 65)"). It lands once, on both engines, gated by the envelope's
    schemaVersion rather than aliased — so `data check` answers 65 in BOTH
    modes and the family has exactly one refusal code again. The human REPORT
    is unchanged: blockers first, same lines, same verdict."""
    bundle = _write_bundle(str(tmp_path / "prompts" / "optvec"))
    os.remove(os.path.join(bundle, "neutral-fluency.jsonl"))
    rc = _run_cli(monkeypatch, tmp_path, ["data", "check", "optvec"])
    out = capsys.readouterr().out
    assert rc == 65
    assert out.splitlines()[0].startswith("missing  neutral-fluency.jsonl")
    assert "NOT ready (1 blocker(s))" in out


def test_cli_dir_flag_and_json(tmp_path, monkeypatch, capsys):
    """WP0 step 8: `--json` is the ENVELOPE flag on every agent-path verb
    (audit §2.2's normalisation — it was boolean here, a file path on
    `vectors compare`, and a file side-effect on `artifacts audit`). Nothing is
    lost: the report this used to print bare is `result.report`, key for key,
    and the envelope adds the state/exit vocabulary the bare report never had.
    """
    bundle = _write_bundle(str(tmp_path / "elsewhere" / "bundle"))
    rc = _run_cli(monkeypatch, tmp_path,
                  ["data", "check", "optvec", "--dir",
                   os.path.join("elsewhere", "bundle"), "--json"])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready"
    assert envelope["verb"] == "data check"
    assert envelope["result"]["ready"] is True
    report = envelope["result"]["report"]
    assert report["ready"] is True
    assert report["blockerCount"] == 0
    assert report["bundleDirectory"] == bundle
    assert report["authoringSpec"] == data_readiness.AUTHORING_SPEC
    by_name = {r["name"]: r for r in report["requirements"]}
    for name in data_readiness.BUNDLE_FILES:
        assert by_name[name]["status"] == "present"
        assert len(by_name[name]["sha256"]) == 64


# --- bundle.json and REPORT.md ---------------------------------------------------


def test_missing_bundle_json_blocks(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    os.remove(os.path.join(bundle, data_readiness.BUNDLE_JSON))
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)[data_readiness.BUNDLE_JSON]
    assert req.status == "missing" and req.blocker and not report.ready


def test_stale_bundle_json_hash_blocks_and_names_the_file(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    # Edit a data file AFTER bundle.json pinned it: the table is now stale.
    _write_choice(os.path.join(bundle, "target-val.jsonl"),
                  _balanced_rows("ovt-target-val-edited"))
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)[data_readiness.BUNDLE_JSON]
    assert req.status == "invalid" and "target-val.jsonl" in req.detail
    assert "stale" in req.detail


def test_bundle_json_missing_directive_blocks(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    _write_bundle_json(bundle, mutate=lambda p: p.update(anchorIssues=[]))
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)[data_readiness.BUNDLE_JSON]
    assert req.status == "invalid" and "anchorIssues" in req.detail


def test_bundle_json_omitted_file_blocks(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    _write_bundle_json(bundle,
                       mutate=lambda p: p["files"].pop("capability-eval"))
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)[data_readiness.BUNDLE_JSON]
    assert req.status == "invalid" and "capability-eval.jsonl" in req.detail


def test_missing_report_md_is_visible_but_never_blocks(tmp_path):
    bundle = _write_bundle(str(tmp_path / "optvec"))
    os.remove(os.path.join(bundle, data_readiness.REPORT_FILE))
    report = data_readiness.check_optvec_bundle(bundle)
    req = _by_name(report)[data_readiness.REPORT_FILE]
    assert req.status == "partial" and not req.blocker
    assert report.ready  # the human half is reported, never gated on
