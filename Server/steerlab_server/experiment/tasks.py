"""Public experiment lifecycle and deferred-judgment operations.

Workflow and helper dependencies are imported from their owning modules.
The pipeline binds public stage operations at invocation time so CLI/routes
and their integration tests share one explicit execution surface.
"""
from __future__ import annotations
from typing import Callable
from . import resume as resume_mod
from steerlab_server.experiment.extraction_workflow import extract
from steerlab_server.experiment.validation_workflow import validate
from steerlab_server.experiment.sweep_workflow import sweep
from steerlab_server.experiment.run_workflow import run
from steerlab_server.experiment.evaluation_workflow import evaluate
from steerlab_server.experiment.analysis_workflow import analyze
from steerlab_server.experiment.analysis_workflow import rescore_style
from steerlab_server.experiment.sweep_evidence import complete_sweep_judgment
from steerlab_server.experiment.deferred_evaluation import complete_evaluate_judgment
from steerlab_server.experiment.deferred_evaluation import judge_worker
from steerlab_server.experiment.sweep_evidence import list_awaiting_judgment
from steerlab_server.experiment.deferred_evaluation import list_awaiting_evaluate_judgment
from steerlab_server.experiment.judge_dispatch import read_judge_fanout_request
from steerlab_server.experiment.deferred_evaluation import read_judge_worker_artifact

__all__ = ['extract', 'validate', 'sweep', 'run', 'evaluate', 'analyze', 'rescore_style', 'pipeline', 'complete_sweep_judgment', 'complete_evaluate_judgment', 'judge_worker', 'list_awaiting_judgment', 'list_awaiting_evaluate_judgment', 'read_judge_fanout_request', 'read_judge_worker_artifact']


def pipeline(name: str, root: str | None = None, dtype: str = "auto",
             device: str | None = None, *, model_provider=None,
             should_cancel: Callable[[], bool] | None = None, log=None,
             checkpoint: "resume_mod.CheckpointFlag | None" = None,
             pipeline_run_directory: str | None = None,
             on_pipeline_directory: Callable[[str], None] | None = None,
             model_release=None) -> str:
    """Execute the declared pipeline through explicit stage/model capabilities."""
    from .pipeline_workflow import PipelineModels, PipelineStages, pipeline as execute
    from . import promote as promote_lib
    from . import model_resources
    return execute(
        name, root, dtype, device, model_provider=model_provider,
        should_cancel=should_cancel, log=log, checkpoint=checkpoint,
        pipeline_run_directory=pipeline_run_directory,
        on_pipeline_directory=on_pipeline_directory, model_release=model_release,
        stages=PipelineStages(extract=extract, validate=validate, sweep=sweep,
                              run=run, evaluate=evaluate, analyze=analyze,
                              promote=promote_lib.promote),
        models=PipelineModels(acquire=model_resources.acquire_model, pin_revision=model_resources.pin_model_revision))
