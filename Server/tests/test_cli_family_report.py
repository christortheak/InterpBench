"""``steerlab-server sae family-report --config`` — the CLI form of the
cross-family descriptive report.

CLI-layer concerns only: dispatch, exit codes (0 done / 2 refusal / 64 usage),
the printed JSON being the run's report, and the three artifacts landing on
disk. The geometry, the matched-layer rule and the behavioural copy are
unit-tested in test_family_report.py. Hand-built artifacts — no model, no
inference, no network.
"""

import json
import os

from steerlab_server import cli
from steerlab_server.experiment import family_report
from tests.test_family_report import (DEFAULT_EFFECT_ROWS, _basis,
                                      write_analyze, write_vector)


def _write_config(path, payload) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return str(path)


def test_family_report_cli_runs_and_prints_the_report(tmp_path, monkeypatch,
                                                      capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    lib = tmp_path / "runs" / "lib"
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 4.0),
                       residual_norms=[10.0] * 4)
    sae = write_vector(lib, "sae", layer=2, row=_basis(1, 2.0),
                       method="gemmaScopeSAE", residual_norms=[10.0] * 4,
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match"})
    write_analyze(tmp_path / "runs" / "cf-analyze", DEFAULT_EFFECT_ROWS)
    config = _write_config(tmp_path / "family.json", {
        "name": "wave1",
        "artifacts": [
            caa,
            {"reference": sae, "label": "F62389",
             "behavior": {"analyze": "runs/cf-analyze",
                          "condition": "sae-f62389"}},
            {"family": "LoRA", "label": "courage-lora",
             "behavior": {"analyze": "runs/cf-analyze",
                          "condition": "lora-courage"}}]})

    assert cli.main(["sae", "family-report", "--config", config]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runType"] == family_report.RUN_TYPE
    assert payload["claim"] == "descriptive"
    assert sorted(payload["families"]) == ["CAA", "LoRA", "SAE"]
    directory = payload["runDirectory"]
    for filename in (family_report.REPORT_JSON, family_report.COSINE_CSV,
                     family_report.SUMMARY_TXT, "config.json"):
        assert os.path.isfile(os.path.join(directory, filename))
    on_disk = json.load(open(os.path.join(directory,
                                          family_report.REPORT_JSON),
                             encoding="utf-8"))
    assert on_disk["runID"] == payload["runID"]


def test_out_name_overrides_the_config_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 1.0))
    b = write_vector(lib, "b", layer=2, row=_basis(1, 1.0))
    config = _write_config(tmp_path / "family.json",
                           {"name": "from-config", "artifacts": [a, b]})
    assert cli.main(["sae", "family-report", "--config", config,
                     "--out-name", "from-flag"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "from-flag"
    assert os.path.basename(payload["runDirectory"]).endswith(
        "-family-report-from-flag")


def test_missing_config_is_a_usage_error_and_a_bad_one_is_a_refusal(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["sae", "family-report"]) == 64
    assert "family-report" in capsys.readouterr().err

    config = _write_config(tmp_path / "bad.json",
                           {"artifacts": ["runs/nope/missing",
                                          "runs/nope/other"]})
    assert cli.main(["sae", "family-report", "--config", config]) == 2
    assert "could not be loaded" in capsys.readouterr().err

    # A path that escapes runs/ never becomes a run directory.
    escape = _write_config(tmp_path / "escape.json",
                           {"name": "../escape",
                            "artifacts": ["runs/a/one", "runs/a/two"]})
    assert cli.main(["sae", "family-report", "--config", escape]) == 2
    assert "plain name component" in capsys.readouterr().err
