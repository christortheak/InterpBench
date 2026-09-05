"""Exercise orchestration through explicit capabilities, without task globals."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store, pipeline_ledger
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.pipeline_workflow import PipelineModels, PipelineStages, pipeline


def _study(root, stage_names):
    experiment_store.create("chain", model_id="test/model", revision="revision", root=str(root))
    raw = experiment_store.load_raw("chain", str(root))
    raw["pipeline"] = {"stages": stage_names}
    experiment_store.save_raw(raw, str(root))
    return Manifest.load("chain", str(root))


def _unexpected(*args, **kwargs):
    raise AssertionError("this capability must not be used")


def _stages(**overrides):
    callbacks = dict.fromkeys(("extract", "validate", "sweep", "run", "evaluate", "analyze", "promote"),
                             _unexpected)
    return PipelineStages(**{**callbacks, **overrides})


def test_cpu_continuation_skips_completed_execution_and_never_acquires_a_model(tmp_path):
    manifest = _study(tmp_path, ["run", "analyze"])
    directory = tmp_path / "runs" / "chain-pipeline"
    directory.mkdir(parents=True)
    source = tmp_path / "runs" / "completed-source-run"
    source.mkdir()
    pipeline_ledger.write_pipeline_ledger(str(directory), {
        "schema": pipeline_ledger.PIPELINE_LEDGER_SCHEMA,
        "experiment": "chain", "experimentHash": manifest.content_hash(),
        "stages": ["run", "analyze"], "disposition": None,
        "stageResults": {"run": {"status": "completed", "runDirectory": str(source)}},
    })
    calls = []

    def analyze(name, root, **kwargs):
        calls.append((name, root, kwargs["source_run"]))
        result = tmp_path / "runs" / "analysis"
        result.mkdir()
        return str(result)

    result = pipeline("chain", str(tmp_path), pipeline_run_directory=str(directory),
                      stages=_stages(analyze=analyze),
                      models=PipelineModels(acquire=_unexpected, pin_revision=_unexpected),
                      log=lambda _: None)
    assert result == str(directory)
    assert calls == [("chain", str(tmp_path), str(source))]
    ledger = pipeline_ledger.read_pipeline_ledger(result)
    assert ledger["disposition"] == "completed"
    assert ledger["stageResults"]["run"]["runDirectory"] == str(source)
    assert ledger["stageResults"]["analyze"]["status"] == "completed"


def test_stage_failure_releases_the_held_model_without_marking_the_stage_complete(tmp_path):
    _study(tmp_path, ["extract"])
    events = []
    directories = []
    held = SimpleNamespace(revision="revision")

    @contextmanager
    def acquire(manifest, dtype, device, provider):
        events.append("acquire")
        try:
            yield held
        finally:
            events.append("release")

    def pin(name, manifest, model, root, log):
        assert model is held
        events.append("pin")
        return manifest

    def extract(name, root, dtype, device, **kwargs):
        with kwargs["model_provider"]("test/model") as model:
            assert model is held
            events.append("stage")
            raise RuntimeError("stage failed")

    with pytest.raises(RuntimeError, match="stage failed"):
        pipeline("chain", str(tmp_path), stages=_stages(extract=extract),
                 models=PipelineModels(acquire=acquire, pin_revision=pin),
                 on_pipeline_directory=directories.append, log=lambda _: None)
    assert events == ["acquire", "pin", "stage", "release"]
    assert len(directories) == 1
    ledger = pipeline_ledger.read_pipeline_ledger(directories[0])
    assert ledger["disposition"] is None
    assert ledger["stageResults"].get("extract", {}).get("status") != "completed"
