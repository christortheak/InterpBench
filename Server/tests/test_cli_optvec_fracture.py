"""``steerlab-server optvec fracture --config`` — the CLI form of the
per-item, per-dose basin count (``optvec_geometry.fracture``). CLI-layer
concerns only: dispatch, exit codes (the shared vocabulary — 0 done / 64 usage
or malformed config / 65 refusal / 66 not found / 70 failure; the family-wide
pins live in test_cli_optvec_exit_codes.py), the printed JSON being the run's
report, and the run directory landing on disk.
The clustering math and the sidecar/config item rules are unit-tested in
test_optvec_geometry.py. Hand-built artifacts — no model, no inference.
"""

import json
import os

from steerlab_server import cli
from steerlab_server.experiment import optvec_geometry
from tests.test_optvec_eval import HIDDEN, write_optvec_artifact


def _basis(index: int, scale: float = 1.0) -> list[float]:
    row = [0.0] * HIDDEN
    row[index] = scale
    return row


def _write_config(path, payload) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return str(path)


def test_fracture_cli_runs_and_prints_the_report(tmp_path, monkeypatch,
                                                 capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Item A, one dose, three restarts: two in one basin, one orthogonal —
    # clusterCount 2. Item B has a single restart in its own group.
    a1 = write_optvec_artifact(tmp_path / "v", "a1", direction=_basis(0, 6.0))
    a2 = write_optvec_artifact(tmp_path / "v", "a2", direction=_basis(0, 3.0))
    a3 = write_optvec_artifact(tmp_path / "v", "a3", direction=_basis(1, 6.0))
    b1 = write_optvec_artifact(tmp_path / "v", "b1", direction=_basis(2, 6.0))
    config = _write_config(tmp_path / "fracture.json", {
        "artifacts": [a1, a2, a3, b1],
        "items": {a1: "item-A", a2: "item-A", a3: "item-A", b1: "item-B"},
        "name": "cli-smoke"})

    assert cli.main(["optvec", "fracture", "--config", config]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runType"] == optvec_geometry.FRACTURE_RUN_TYPE
    assert report["threshold"] == optvec_geometry.DEFAULT_CLUSTER_THRESHOLD
    assert report["count"] == 4 and report["itemCount"] == 2

    by_key = {(g["item"], g["dose"]): g for g in report["groups"]}
    (item_a,) = [k for k in by_key if k[0] == "item-A"]
    assert by_key[item_a]["restarts"] == 3
    assert by_key[item_a]["clusterCount"] == 2
    (item_b,) = [k for k in by_key if k[0] == "item-B"]
    assert by_key[item_b]["clusterCount"] == 1

    # The printed report IS the run's fracture.json (plus the directory key).
    run_dir = report["runDirectory"]
    on_disk = json.load(open(os.path.join(run_dir,
                                          optvec_geometry.FRACTURE_JSON)))
    assert on_disk == {k: v for k, v in report.items() if k != "runDirectory"}
    assert json.load(open(os.path.join(run_dir, "config.json")))[
        "runType"] == optvec_geometry.FRACTURE_RUN_TYPE


def test_fracture_cli_reads_a_gradient_reference(tmp_path, monkeypatch,
                                                 capsys):
    from safetensors.numpy import save_file
    import numpy as np

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    a1 = write_optvec_artifact(tmp_path / "v", "a1", direction=_basis(0, 6.0))
    a2 = write_optvec_artifact(tmp_path / "v", "a2", direction=_basis(0, 3.0))
    gradients = str(tmp_path / "gradients.safetensors")
    save_file({"item-A": np.asarray(_basis(0), dtype=np.float32)}, gradients)
    config = _write_config(tmp_path / "fracture.json", {
        "artifacts": [a1, a2], "items": {a1: "item-A", a2: "item-A"},
        "gradients": gradients, "threshold": 0.8})

    assert cli.main(["optvec", "fracture", "--config", config]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["threshold"] == 0.8
    assert report["gradientReference"]["path"] == gradients
    (group,) = report["groups"]
    assert group["gradientReferencePresent"] is True
    (cluster,) = group["clusters"]
    assert cluster["meanGradientCosine"] == 1.0


def test_fracture_cli_usage_and_refusals(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))

    # No --config is usage, not a refusal.
    assert cli.main(["optvec", "fracture"]) == 64
    assert "optvec fracture --config" in capsys.readouterr().err

    # A config path naming no file is notFound (66) with the verb named.
    assert cli.main(["optvec", "fracture", "--config",
                     str(tmp_path / "missing.json")]) == 66
    assert "optvec fracture:" in capsys.readouterr().err

    # A config that breaks its own contract (one artifact is not a
    # multiplicity statistic) is a malformed invocation, 64.
    lone = write_optvec_artifact(tmp_path / "v", "lone")
    one = _write_config(tmp_path / "one.json", {"artifacts": [lone]})
    assert cli.main(["optvec", "fracture", "--config", one]) == 64
    assert "at least 2" in capsys.readouterr().err

    # An artifact the config cannot group (no sidecar marker, no mapping)
    # refuses rather than guessing an item: a well-formed request an input
    # declined, 65.
    other = write_optvec_artifact(tmp_path / "v", "other")
    ungrouped = _write_config(tmp_path / "ungrouped.json",
                              {"artifacts": [lone, other]})
    assert cli.main(["optvec", "fracture", "--config", ungrouped]) == 65
    assert "names no item" in capsys.readouterr().err

    # A gradient reference that names no file is notFound, not a refusal.
    grouped = _write_config(tmp_path / "grouped.json", {
        "artifacts": [lone, other],
        "items": {lone: "item-A", other: "item-A"},
        "gradients": str(tmp_path / "nope.safetensors")})
    assert cli.main(["optvec", "fracture", "--config", grouped]) == 66
    assert "names no file" in capsys.readouterr().err
