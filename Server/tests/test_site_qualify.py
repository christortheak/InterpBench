"""``site qualify`` — the release gate that makes a cold node trustworthy.

WP6 R1 / release gate 7: *"`site qualify` passes on that deployment, and its
report is legible to someone who has never seen our baseline."* Both halves are
tested here — the verdict arithmetic (what passes, what warns, what fails, what
is merely unverified) and the legibility floor (every check carries a non-empty
``what``/``why``/``expected``/``observed``, because a status word with no
statement of what was compared is exactly the report a stranger cannot use).

The checks are unit-tested by forcing each status through the seam the check
itself reads, so the table's behaviour is pinned without needing a cluster, a
GPU, or a broken node.
"""

import json
import os
import shutil

import pytest

from steerlab_server import cli, cli_envelope, site_qualify
from steerlab_server.site_qualify import CheckSpec, Context, Outcome

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
STIMULUS_DIR = os.path.join(REPO_ROOT, site_qualify.STIMULUS_FIXTURE)
PARITY_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests", "fixtures", "parity")


def _context(**kwargs) -> Context:
    return Context(**kwargs)


# =============================================================================
# 1. The committed fixture this verb owns
# =============================================================================


def test_the_stimulus_fixtures_pinned_hash_is_the_real_one():
    """The expected digest is committed DATA, and a stale one would pass
    falsely on a remote node while failing nothing here. Recomputed from the
    committed bytes on every run."""
    from steerlab_server.steering.stimulus_set import StimulusSet

    with open(os.path.join(STIMULUS_DIR, "expected-hash.txt"),
              encoding="utf-8") as handle:
        pinned = handle.read().strip()
    assert pinned == StimulusSet.from_directory(STIMULUS_DIR).hash
    assert len(pinned) == 64


def test_the_stimulus_fixture_is_small_and_well_formed():
    from steerlab_server.steering.stimulus_set import StimulusSet

    stimuli = StimulusSet.from_directory(STIMULUS_DIR)
    assert stimuli.positive and stimuli.negative
    # Deliberately tiny: this exercises the hashing contract, never extraction.
    assert len(stimuli.positive) <= 8 and len(stimuli.negative) <= 8


# =============================================================================
# 2. Each check, forced through every status it can report
# =============================================================================


def test_build_identity_passes_with_a_resolvable_commit(monkeypatch):
    from steerlab_server import build_identity

    monkeypatch.setattr(build_identity, "build_commit", lambda: "ab12cd34")
    monkeypatch.setattr(build_identity, "engine_version",
                        lambda: "steerlab-server 0.1.0+ab12cd34")
    outcome = site_qualify._check_build_identity(_context())
    assert outcome.status == "pass"
    assert outcome.observed == "steerlab-server 0.1.0+ab12cd34"


def test_build_identity_warns_on_a_bare_version(monkeypatch):
    from steerlab_server import build_identity

    monkeypatch.setattr(build_identity, "build_commit", lambda: None)
    monkeypatch.setattr(build_identity, "engine_version",
                        lambda: "steerlab-server 0.1.0")
    outcome = site_qualify._check_build_identity(_context())
    assert outcome.status == "warn"
    assert "cannot be traced to a commit" in outcome.detail
    # The repair is concrete, not advice.
    assert "BUILD_COMMIT" in outcome.detail


def test_python_environment_always_passes_and_names_the_stack():
    outcome = site_qualify._check_python_environment(_context())
    assert outcome.status == "pass"
    assert "torch" in outcome.observed or "torch" in outcome.detail
    assert "records the stack rather than gating it" in outcome.expected


def test_dependency_lock_passes_when_the_lock_agrees(monkeypatch):
    from steerlab_server import python_environment

    monkeypatch.setattr(python_environment, "lock_path",
                        lambda platform_value: "/x/requirements-fake.lock")
    monkeypatch.setattr(python_environment, "lock_drift",
                        lambda platform_value: [])
    outcome = site_qualify._check_dependency_lock(_context())
    assert outcome.status == "pass"
    assert "requirements-fake.lock" in outcome.expected


def test_dependency_lock_warns_with_the_drift_lines(monkeypatch):
    from steerlab_server import python_environment

    monkeypatch.setattr(python_environment, "lock_path",
                        lambda platform_value: "/x/requirements-fake.lock")
    monkeypatch.setattr(
        python_environment, "lock_drift",
        lambda platform_value: ["torch: installed 9.9.9, lock pins 1.0.0"])
    outcome = site_qualify._check_dependency_lock(_context())
    # Advisory semantics: a site may deliberately run its own torch.
    assert outcome.status == "warn"
    assert "installed 9.9.9" in outcome.observed


def test_dependency_lock_skips_on_an_unlocked_platform(monkeypatch):
    from steerlab_server import python_environment

    monkeypatch.setattr(python_environment, "lock_path",
                        lambda platform_value: None)
    outcome = site_qualify._check_dependency_lock(_context())
    assert outcome.status == "skip"
    assert "no lock ships" in outcome.observed


def test_stimulus_hash_passes_against_the_committed_fixture():
    outcome = site_qualify._check_stimulus_hash(_context())
    assert outcome.status == "pass"
    assert outcome.observed == outcome.expected
    assert len(outcome.observed) == 64


def test_stimulus_hash_fails_when_the_bytes_hash_differently(
        tmp_path, monkeypatch):
    """The real failure this catches: a transfer that rewrote line endings.
    Same logical lines, different bytes, different digest — and every frozen
    experiment on that node would fail to verify its pins."""
    directory = tmp_path / "stimulus"
    directory.mkdir()
    for name in ("positive.jsonl", "negative.jsonl"):
        source = os.path.join(STIMULUS_DIR, name)
        with open(source, "rb") as handle:
            data = handle.read()
        (directory / name).write_bytes(data.replace(b"\n", b"\r\n"))
    shutil.copy(os.path.join(STIMULUS_DIR, "expected-hash.txt"),
                str(directory / "expected-hash.txt"))
    monkeypatch.setattr(site_qualify, "_fixture_candidates",
                        lambda relative, **kwargs: [str(directory)])
    outcome = site_qualify._check_stimulus_hash(_context())
    assert outcome.status == "fail"
    assert outcome.observed != outcome.expected
    assert "line endings" in outcome.detail


def test_stimulus_hash_skips_when_the_fixture_did_not_ship(
        tmp_path, monkeypatch):
    monkeypatch.setattr(site_qualify, "_fixture_candidates",
                        lambda relative, **kwargs: [str(tmp_path / "absent")])
    outcome = site_qualify._check_stimulus_hash(_context())
    assert outcome.status == "skip"
    assert "UNVERIFIED" in outcome.detail


# --- the two fixture checks ---------------------------------------------------


def _fake_fixture(path, *, model_id="fake/model", rendered="hello",
                  ids=(1, 2, 3), bos_count=0, bos_id=None):
    payload = {
        "family": "fake", "modelID": model_id, "case": "fake",
        "inputs": {"api": "render", "prompt": "hi", "promptMode": "chatAssistant",
                   "systemPrompt": None, "qwenThinkingEnabled": False},
        "rendered": rendered, "tokenIDs": list(ids),
        "promptTokenCount": len(ids), "bosCount": bos_count,
        "bosTokenID": bos_id,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class _StubTokenizer:
    bos_token_id = None


def _point_at_fixtures(monkeypatch, directory, *, paths=None):
    monkeypatch.setattr(
        site_qualify, "_render_fixture_paths",
        lambda: (str(directory), paths if paths is not None else sorted(
            os.path.join(str(directory), name)
            for name in os.listdir(str(directory)) if name.endswith(".json"))))


def test_golden_fixtures_pass_when_every_fixture_reproduces(
        tmp_path, monkeypatch):
    _fake_fixture(tmp_path / "a.json")
    _fake_fixture(tmp_path / "b.json")
    _point_at_fixtures(monkeypatch, tmp_path)
    monkeypatch.setattr(Context, "tokenizer",
                        lambda self, model_id: _StubTokenizer())
    monkeypatch.setattr(site_qualify, "_rerender",
                        lambda tok, fx: ("hello", [1, 2, 3]))
    context = _context()
    assert site_qualify._check_golden_render(context).status == "pass"
    assert site_qualify._check_golden_tokens(context).status == "pass"
    assert "2 of 2" in site_qualify._check_golden_render(context).observed


def test_a_render_difference_fails_only_the_render_check(tmp_path, monkeypatch):
    _fake_fixture(tmp_path / "a.json")
    _point_at_fixtures(monkeypatch, tmp_path)
    monkeypatch.setattr(Context, "tokenizer",
                        lambda self, model_id: _StubTokenizer())
    monkeypatch.setattr(site_qualify, "_rerender",
                        lambda tok, fx: ("goodbye", [1, 2, 3]))
    context = _context()
    render = site_qualify._check_golden_render(context)
    assert render.status == "fail"
    assert "rendered string differs" in render.observed
    # Ids still match, so tokenization is not accused of a fault it did not
    # commit — the two checks report independently off one sweep.
    assert site_qualify._check_golden_tokens(context).status == "pass"


def test_a_token_difference_fails_only_the_token_check(tmp_path, monkeypatch):
    _fake_fixture(tmp_path / "a.json")
    _point_at_fixtures(monkeypatch, tmp_path)
    monkeypatch.setattr(Context, "tokenizer",
                        lambda self, model_id: _StubTokenizer())
    monkeypatch.setattr(site_qualify, "_rerender",
                        lambda tok, fx: ("hello", [1, 2, 3, 4]))
    context = _context()
    assert site_qualify._check_golden_render(context).status == "pass"
    tokens = site_qualify._check_golden_tokens(context)
    assert tokens.status == "fail"
    assert "token ids differ" in tokens.observed


def test_a_double_bos_regression_fails_the_token_check(tmp_path, monkeypatch):
    """The tripwire the render goldens exist for: one BOS pinned, two here."""
    _fake_fixture(tmp_path / "a.json", ids=(2, 5, 6), bos_count=1, bos_id=2)
    _point_at_fixtures(monkeypatch, tmp_path)

    class _Gemma:
        bos_token_id = 2

    monkeypatch.setattr(Context, "tokenizer", lambda self, model_id: _Gemma())
    monkeypatch.setattr(site_qualify, "_rerender",
                        lambda tok, fx: ("hello", [2, 2, 5, 6]))
    tokens = site_qualify._check_golden_tokens(_context())
    assert tokens.status == "fail"
    assert "token ids differ" in tokens.observed


def test_a_partly_verified_sweep_warns_rather_than_passing(
        tmp_path, monkeypatch):
    """Four of thirty-six fixtures verified is not a verified node. Partial is
    a warning — the exact overclaim this command exists to prevent."""
    _fake_fixture(tmp_path / "a.json", model_id="cached/model")
    _fake_fixture(tmp_path / "b.json", model_id="uncached/model")
    _point_at_fixtures(monkeypatch, tmp_path)
    monkeypatch.setattr(
        Context, "tokenizer",
        lambda self, model_id: _StubTokenizer() if model_id == "cached/model"
        else None)
    monkeypatch.setattr(site_qualify, "_rerender",
                        lambda tok, fx: ("hello", [1, 2, 3]))
    outcome = site_qualify._check_golden_render(_context())
    assert outcome.status == "warn"
    assert "1 of 2" in outcome.observed
    assert "uncached/model" in outcome.observed
    assert "hole, not a result" in outcome.detail


def test_no_tokenizer_at_all_is_a_skip_not_a_pass(tmp_path, monkeypatch):
    _fake_fixture(tmp_path / "a.json")
    _point_at_fixtures(monkeypatch, tmp_path)
    monkeypatch.setattr(Context, "tokenizer", lambda self, model_id: None)
    outcome = site_qualify._check_golden_tokens(_context())
    assert outcome.status == "skip"
    assert "0 of 1" in outcome.observed
    assert "not a failure and it is not a pass" in outcome.detail


def test_skip_model_fixtures_forces_the_skip_and_says_so(tmp_path, monkeypatch):
    _fake_fixture(tmp_path / "a.json")
    _point_at_fixtures(monkeypatch, tmp_path)
    outcome = site_qualify._check_golden_render(
        _context(skip_model_fixtures=True))
    assert outcome.status == "skip"
    assert "--skip-model-fixtures" in outcome.observed


def test_missing_render_fixtures_skip_with_the_payload_reason(monkeypatch):
    monkeypatch.setattr(site_qualify, "_render_fixture_paths",
                        lambda: (None, []))
    outcome = site_qualify._check_golden_render(_context())
    assert outcome.status == "skip"
    assert "did not ship" in outcome.detail


def test_the_real_render_goldens_are_readable_here():
    """Not a parity assertion (test_golden_render_fixtures.py owns that) — a
    guard that the locator finds the committed directory from the package, so
    a shipped payload's skip reason means what it says."""
    directory, paths = site_qualify._render_fixture_paths()
    assert directory and len(paths) >= 30


# --- vector parity ------------------------------------------------------------


def test_vector_parity_passes_against_the_committed_goldens():
    outcome = site_qualify._check_vector_parity(_context())
    assert outcome.status == "pass"
    assert "4 of 4" in outcome.observed
    # The documented limit is restated in the report, not only in the source.
    assert "same-engine SYNTHETIC" in outcome.detail
    assert "cross-substrate" in outcome.detail


def test_vector_parity_fails_when_a_golden_disagrees(tmp_path, monkeypatch):
    staging = tmp_path / "parity"
    shutil.copytree(PARITY_DIR, str(staging))
    golden_path = staging / "golden-identical.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    golden["summary"]["minCosine"] = 0.5
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    monkeypatch.setattr(site_qualify, "_fixture_candidates",
                        lambda relative, **kwargs: [str(staging)])
    outcome = site_qualify._check_vector_parity(_context())
    assert outcome.status == "fail"
    assert "identical:" in outcome.observed
    assert "minCosine" in outcome.observed


def test_vector_parity_skips_when_tests_were_not_shipped(tmp_path, monkeypatch):
    monkeypatch.setattr(site_qualify, "_fixture_candidates",
                        lambda relative, **kwargs: [str(tmp_path / "absent")])
    outcome = site_qualify._check_vector_parity(_context())
    assert outcome.status == "skip"
    assert "did not ship Server/tests/" in outcome.detail


def test_the_numeric_compare_is_the_goldens_own_tolerance():
    assert site_qualify.PARITY_TOLERANCE == 1e-6
    assert site_qualify._numeric_mismatch({"a": 1.0}, {"a": 1.0 + 1e-9}) is None
    assert site_qualify._numeric_mismatch({"a": 1.0}, {"a": 1.1}) is not None
    # Key sets are compared at every level, not just values.
    assert "key set differs" in site_qualify._numeric_mismatch(
        {"a": 1}, {"a": 1, "b": 2})
    # A bool observed where the golden pins a number is not "1.0 within
    # tolerance" — it is the wrong type (`pass` is not a cosine).
    assert site_qualify._numeric_mismatch(True, 1) is not None
    # Lists compare element-wise, including length.
    assert site_qualify._numeric_mismatch([1.0], [1.0, 2.0]) is not None


# --- profile ------------------------------------------------------------------


def _profile_report(**counts):
    checks = ([{"name": "root", "status": "fail", "message": "not writable"}]
              * counts.get("failures", 0)
              + [{"name": "gres", "status": "warn", "message": "unset"}]
              * counts.get("warnings", 0)
              + [{"name": "bind", "status": "ok", "message": "binds 127.0.0.1"}])
    return {"ok": counts.get("failures", 0) == 0,
            "failures": counts.get("failures", 0),
            "warnings": counts.get("warnings", 0),
            "profile": {}, "checks": checks}


@pytest.mark.parametrize("counts,expected", [
    ({}, "pass"), ({"warnings": 1}, "warn"), ({"failures": 1}, "fail"),
])
def test_server_profile_folds_the_existing_validator(counts, expected,
                                                     monkeypatch):
    """The verdict is folded, never re-derived — `profile validate` stays the
    authority and this check says so in its own detail."""
    from steerlab_server.api import profile as profile_mod

    monkeypatch.setattr(profile_mod, "validate_profile",
                        lambda *a, **k: _profile_report(**counts))
    outcome = site_qualify._check_server_profile(_context())
    assert outcome.status == expected
    assert "profile validate" in outcome.detail


# --- cuda ---------------------------------------------------------------------


class _Properties:
    def __init__(self, name):
        self.name = name
        self.total_memory = 80 * 1024 ** 3


def _fake_cuda(monkeypatch, names):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: _Properties(names[index]))


def test_cuda_probe_skips_with_no_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("STEERLAB_SLURM_GPU_TYPES", raising=False)
    outcome = site_qualify._check_cuda_probe(_context())
    assert outcome.status == "skip"
    assert "normal on a login node" in outcome.observed
    assert "--hello" in outcome.detail


def test_cuda_probe_passes_and_reports_the_devices(monkeypatch):
    _fake_cuda(monkeypatch, ["NVIDIA H100 80GB HBM3"])
    monkeypatch.delenv("STEERLAB_SLURM_GPU_TYPES", raising=False)
    outcome = site_qualify._check_cuda_probe(_context())
    assert outcome.status == "pass"
    assert "H100" in outcome.observed and "80 GB" in outcome.observed


def test_cuda_probe_passes_when_the_declared_vocabulary_names_the_device(
        monkeypatch):
    _fake_cuda(monkeypatch, ["NVIDIA H100 80GB HBM3"])
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100,H100")
    outcome = site_qualify._check_cuda_probe(_context())
    assert outcome.status == "pass"


def test_cuda_probe_warns_on_a_device_outside_the_declared_vocabulary(
        monkeypatch):
    """String containment, not fuzzy matching: a typed gres naming a device
    the site never declared is refused before it queues."""
    _fake_cuda(monkeypatch, ["NVIDIA L40S"])
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100,H100")
    outcome = site_qualify._check_cuda_probe(_context())
    assert outcome.status == "warn"
    assert "STEERLAB_SLURM_GPU_TYPES" in outcome.detail


# =============================================================================
# 3. Assembly: the report, its legibility floor, and its completeness
# =============================================================================


def _report(**kwargs) -> dict:
    return site_qualify.qualify(skip_model_fixtures=True, **kwargs)


def test_the_report_document_has_the_declared_shape():
    report = _report()
    assert set(report) == {"schemaVersion", "generatedBy", "generatedAt",
                           "platform", "checks", "summary"}
    assert report["schemaVersion"] == site_qualify.SCHEMA_VERSION
    assert report["generatedBy"].startswith("steerlab-server ")
    assert report["generatedAt"].endswith("Z")
    assert set(report["summary"]) == {"passed", "warnings", "failed",
                                      "skipped", "total", "line"}


def test_every_check_row_meets_the_legibility_floor():
    """Gate 7's requirement, as a test: a status word with no statement of
    what was compared is a report a stranger cannot act on."""
    for row in _report()["checks"]:
        assert set(row) == {"id", "title", "status", "what", "why", "expected",
                            "observed", "detail"}
        for field in ("title", "what", "why", "expected", "observed"):
            assert row[field].strip(), f"{row['id']} has an empty {field}"
        assert row["status"] in site_qualify.STATUSES
        # Written for a stranger: never a reference to a baseline they have
        # no access to.
        assert "our baseline" not in (row["what"] + row["why"]).lower()


def test_the_checks_table_and_a_real_report_are_the_same_id_set():
    """A check added to the table and not reachable — or reachable and not
    declared — would be invisible to every other test here."""
    assert [row["id"] for row in _report()["checks"]] == list(
        site_qualify.CHECK_IDS)
    assert len(set(site_qualify.CHECK_IDS)) == len(site_qualify.CHECKS)


def test_the_summary_counts_skips_so_an_unverified_node_reads_as_one():
    line = site_qualify.summary_line(
        {"passed": 3, "warnings": 0, "failed": 0, "skipped": 6, "total": 9})
    assert line == "3 passed, 0 warnings, 0 failed, 6 skipped of 9 checks"
    assert site_qualify.summary_line(
        {"passed": 8, "warnings": 1, "failed": 0, "skipped": 0,
         "total": 9}).startswith("8 passed, 1 warning,")


def test_a_check_that_raises_becomes_its_own_failure_and_stops_nothing(
        monkeypatch):
    def _explode(context):
        raise RuntimeError("the node ate it")

    monkeypatch.setattr(site_qualify, "CHECKS", (
        CheckSpec(id="boom", title="Boom", what="w", why="y", run=_explode),
        CheckSpec(id="after", title="After", what="w", why="y",
                  run=lambda ctx: Outcome("pass", "e", "o")),
    ))
    report = site_qualify.qualify()
    assert [row["status"] for row in report["checks"]] == ["fail", "pass"]
    assert "RuntimeError: the node ate it" in report["checks"][0]["observed"]
    assert "remaining checks still ran" in report["checks"][0]["detail"]


def test_an_unknown_status_is_a_programming_error():
    with pytest.raises(ValueError, match="closed"):
        Outcome("green", "e", "o")


def test_qualify_on_this_machine_produces_a_complete_report():
    """The verb end to end, on whatever node the suite is running on: ten
    checks, every one answered, no exception escaping."""
    report = site_qualify.qualify()
    assert report["summary"]["total"] == len(site_qualify.CHECKS) == 10
    counts = report["summary"]
    assert (counts["passed"] + counts["warnings"] + counts["failed"]
            + counts["skipped"]) == counts["total"]
    assert counts["failed"] == 0, (
        "site qualify FAILS on this machine: "
        + json.dumps([row for row in report["checks"]
                      if row["status"] == "fail"], indent=2))


# =============================================================================
# 4. The verb: envelope, exit codes, and the report's two destinations
# =============================================================================


def test_the_verb_prints_the_bare_report_on_stdout(capsys):
    assert cli.main(["site", "qualify", "--skip-model-fixtures"]) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["schemaVersion"] == 1
    # The human rendering — one line per check plus the summary — is on stderr,
    # so stdout stays diffable between two nodes.
    assert "site qualify —" in captured.err
    assert document["summary"]["line"] in captured.err


def test_json_mode_emits_exactly_one_envelope_carrying_the_report(capsys):
    assert cli.main(["site", "qualify", "--json", "--skip-model-fixtures"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["verb"] == "site qualify"
    assert envelope["engine"] == cli_envelope.ENGINE
    assert envelope["state"] in ("ready", "okWithAdvisories")
    result = envelope["result"]
    assert set(result) == {"passed", "warnings", "failed", "skipped",
                           "checkCount", "failingChecks", "warningChecks",
                           "report"}
    assert result["checkCount"] == len(site_qualify.CHECKS)
    assert result["report"]["checks"][0]["id"] == site_qualify.CHECK_IDS[0]


def test_warnings_become_envelope_advisories_and_still_exit_zero(
        monkeypatch, capsys):
    monkeypatch.setattr(site_qualify, "CHECKS", (
        CheckSpec(id="warned", title="T", what="w", why="y",
                  run=lambda ctx: Outcome("warn", "e", "observed thing")),
    ))
    assert cli.main(["site", "qualify", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "okWithAdvisories"
    assert envelope["advisories"] == [
        {"code": "siteQualifyWarning", "detail": "warned: observed thing"}]
    assert envelope["result"]["warningChecks"] == ["warned"]


def test_an_all_pass_report_is_ready_with_no_advisories(monkeypatch, capsys):
    monkeypatch.setattr(site_qualify, "CHECKS", (
        CheckSpec(id="fine", title="T", what="w", why="y",
                  run=lambda ctx: Outcome("pass", "e", "o")),
    ))
    assert cli.main(["site", "qualify", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready"
    assert "advisories" not in envelope


def test_skips_alone_never_change_the_verdict(monkeypatch, capsys):
    monkeypatch.setattr(site_qualify, "CHECKS", (
        CheckSpec(id="unverified", title="T", what="w", why="y",
                  run=lambda ctx: Outcome("skip", "e", "not shipped")),
    ))
    assert cli.main(["site", "qualify", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready"
    # …but the hole is stated, in the message a caller reads first.
    assert "1 skipped of 1 checks" in envelope["message"]


def test_a_failing_check_exits_70_and_names_it(monkeypatch, capsys):
    monkeypatch.setattr(site_qualify, "CHECKS", (
        CheckSpec(id="broken", title="T", what="w", why="y",
                  run=lambda ctx: Outcome("fail", "the pinned digest",
                                          "a different digest",
                                          "re-transfer the tree")),
        CheckSpec(id="fine", title="T", what="w", why="y",
                  run=lambda ctx: Outcome("pass", "e", "o")),
    ))
    assert cli.main(["site", "qualify", "--json"]) == 70
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "failed"
    assert envelope["error"]["code"] == cli.SITE_QUALIFY_FAILED
    # A node problem is not a study gate: no gate id is claimed.
    assert "gate" not in envelope["error"]
    assert "broken" in envelope["error"]["reason"]
    assert "re-transfer the tree" in envelope["error"]["repairAction"]
    assert envelope["result"]["failingChecks"] == ["broken"]


def test_human_mode_exits_70_on_failure_too(monkeypatch, capsys):
    """Both modes agree — this verb is new, so there is no historical human
    exit code to preserve."""
    monkeypatch.setattr(site_qualify, "CHECKS", (
        CheckSpec(id="broken", title="T", what="w", why="y",
                  run=lambda ctx: Outcome("fail", "e", "o", "repair it")),
    ))
    assert cli.main(["site", "qualify"]) == 70
    assert "FAIL broken" in capsys.readouterr().err


def test_json_out_writes_the_bare_report_to_a_file(tmp_path, capsys):
    out = tmp_path / "qualify.json"
    assert cli.main(["site", "qualify", "--json", str(out),
                     "--skip-model-fixtures"]) == 0
    captured = capsys.readouterr()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    # Same document as stdout, and the wrote-line is a diagnostic.
    assert on_disk == json.loads(captured.out)
    assert f"wrote {out}" in captured.err


def test_out_writes_the_envelope(tmp_path, capsys):
    out = tmp_path / "envelope.json"
    assert cli.main(["site", "qualify", "--out", str(out),
                     "--skip-model-fixtures"]) == 0
    capsys.readouterr()
    envelope = json.loads(out.read_text(encoding="utf-8"))
    assert envelope["verb"] == "site qualify"


def test_the_family_usage_and_flag_parsing(capsys):
    assert cli.main(["site"]) == 64
    assert "site qualify" in capsys.readouterr().err
    assert cli.main(["site", "nonsense"]) == 64
    capsys.readouterr()
    assert cli.main(["site", "qualify", "--nope"]) == 64
    assert "--skip-model-fixtures" in capsys.readouterr().err
    assert cli.main(["site", "qualify", "--help"]) == 0
    assert "run nothing" in capsys.readouterr().out


def test_the_verb_is_declared_on_the_agent_path():
    spec = cli_envelope.spec_for("site", "qualify")
    assert spec is not None and spec.purpose
    assert "site" in cli_envelope.AGENT_FAMILIES
    assert "site" in cli._AGENT_FAMILY_ORDER
    assert "--skip-model-fixtures" in spec.boolean_flags
