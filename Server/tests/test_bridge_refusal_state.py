"""A refused local workspace action answers ``refused`` (65), on both clients.

The local workspace actions (``science report``, the custody and policy
verbs, ``results export``'s bridge) are reached two ways: the Python client's
own ``science`` verbs, and the Mac command line and app through the bridge
process (``client.diagnostic_workspace``). A well-formed request an action
declines — a run folder with no report, say — used to answer ``blocked``/64
on both, the class the contract keeps for a request in the wrong shape. Both
now take the classification from one place,
``diagnostic_commands.refusal_fields``: ``refused``/65 for a refusal,
``blocked``/64 only for a malformed request. The Mac side of the bridge is
tested in ``Tests/ExperimentKitTests/DiagnosticBridgeRefusalTests.swift``.
"""

import io
import json
import sys

import pytest

from steerlab_server import client_cli
from steerlab_server.client import diagnostic_workspace
from steerlab_server.client.diagnostic_commands import refusal_fields, workspace_action
from steerlab_server.client.reports.science_report import ReportRefusal
from steerlab_server.client.results_commands import bridge
from steerlab_server.experiment import diagnostic_archives as archives
from steerlab_server.experiment.manifest_errors import ExperimentStoreError
from steerlab_server.experiment.science_catalog import ScienceRefusal


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "runs").mkdir(parents=True)
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    return root.resolve()


def _client(capsys, *arguments):
    capsys.readouterr()
    code = client_cli.main([*arguments, "--json"])
    return code, json.loads(capsys.readouterr().out)


def _bridge(monkeypatch, capsys, action, payload):
    document = {"action": action, "payload": payload,
                "clientSHA256": diagnostic_workspace.source_sha256()}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(document)))
    capsys.readouterr()
    status = diagnostic_workspace.main()
    return status, json.loads(capsys.readouterr().out)


def test_the_classification():
    assert refusal_fields(archives.MalformedRequest("wrong fields")) == {
        "code": "usage", "state": "blocked"}
    assert refusal_fields(ScienceRefusal("Unknown scientific operation.")) == {
        "code": "invalidScienceRequest", "state": "blocked"}
    assert refusal_fields(ReportRefusal("no report here", "give a run folder")) == {
        "code": "reportRefused", "state": "refused"}
    assert refusal_fields(archives.Refusal("custody failed")) == {
        "code": "diagnosticTransportRefused", "state": "refused"}
    assert refusal_fields(ValueError("anything else")) == {
        "code": "diagnosticTransportRefused", "state": "refused"}
    assert refusal_fields(ExperimentStoreError("frozen", gate="statusImmutable")) == {
        "code": "statusImmutable", "state": "refused"}


def test_the_client_answers_a_refused_report_with_65(workspace, capsys):
    code, document = _client(capsys, "science", "report", "runs/missing",
                             "--root", str(workspace))
    assert code == 65
    assert document["state"] == "refused"
    assert document["error"]["code"] == "reportRefused"
    assert "Nothing was found at runs/missing" in document["error"]["reason"]
    assert "assessment-report.json" in document["error"]["repairAction"]


def test_the_client_keeps_64_for_a_malformed_request(workspace, capsys):
    code, document = _client(capsys, "science", "report", "--root", str(workspace))
    assert code == 64
    assert document["state"] == "blocked"


def test_the_bridge_carries_the_same_classification(workspace, monkeypatch, capsys):
    status, refused = _bridge(monkeypatch, capsys, "report",
                              {"workspaceRoot": str(workspace), "path": "runs/missing"})
    assert refused["ok"] is False and status == 65
    assert refused["state"] == "refused" and refused["code"] == "reportRefused"
    assert "assessment-report.json" in refused["repairAction"]
    _status, malformed = _bridge(monkeypatch, capsys, "report",
                                 {"workspaceRoot": str(workspace)})
    assert malformed["ok"] is False
    assert malformed["state"] == "blocked" and malformed["code"] == "usage"


def test_a_source_mismatch_is_not_a_refusal(monkeypatch, capsys):
    document = {"action": "report", "payload": {}, "clientSHA256": "0" * 64}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(document)))
    assert diagnostic_workspace.main() == 65
    answer = json.loads(capsys.readouterr().out)
    assert answer["ok"] is False and answer["state"] == "blocked"


def test_request_shape_checks_are_malformed(workspace):
    with pytest.raises(archives.MalformedRequest):
        workspace_action("report", {"workspaceRoot": str(workspace), "path": "x", "extra": "y"})
    with pytest.raises(archives.MalformedRequest):
        workspace_action("no-such-action", {"workspaceRoot": str(workspace)})
    with pytest.raises(archives.MalformedRequest):
        workspace_action("setup-start", {"workspaceRoot": str(workspace)})
    with pytest.raises(archives.MalformedRequest):
        bridge({"workspaceRoot": str(workspace)})
