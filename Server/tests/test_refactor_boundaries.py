"""Guard dependency direction and the independently injectable runtime boundary."""
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from steerlab_server.experiment import runtime_backends


def test_offline_owners_import_without_runtime_or_task_orchestrator():
    server = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", """
import sys, importlib.abc
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'transformers', 'peft'}
                or fullname == 'steerlab_server.experiment.tasks'):
            raise AssertionError('unexpected execution dependency: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from steerlab_server.experiment import (analysis_workflow, analysis_endpoints,
    study_admission, run_artifacts, task_inputs, sampling, run_reporting,
    pipeline_ledger, runtime_backends, cancellation, judge_dispatch,
    judgment_evidence, pipeline_evidence, pipeline_policy, pipeline_workflow,
    manifest_errors, manifest_mutation_policy, manifest_declaration_policy,
    draft_protocol_policy, freeze_policy)
from pathlib import Path
assert Path(analysis_workflow.__file__).resolve().is_relative_to(Path.cwd().resolve())
"""], cwd=server, env={**os.environ, "PYTHONPATH": str(server)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shared_consumers_do_not_import_the_task_orchestrator():
    package = Path(__file__).resolve().parents[1] / "steerlab_server"
    for relative in ("experiment/battery_run.py", "experiment/multi_agent.py",
                     "experiment/sharding.py", "jlens/qualification.py"):
        tree = ast.parse((package / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[-1] != "tasks", relative
                assert all(alias.name != "tasks" for alias in node.names), relative
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[-1] != "tasks"
                           for alias in node.names), relative


def test_execution_owners_import_without_the_task_facade():
    server = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", """
import sys, importlib.abc
class NoFacade(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'steerlab_server.experiment.tasks':
            raise AssertionError('reverse workflow dependency: ' + fullname)
sys.meta_path.insert(0, NoFacade())
from steerlab_server.experiment import (
    model_resources, vector_materialization, condition_execution,
    extraction_workflow, validation_workflow, layer_resolution,
    sweep_workflow, choice_scoring, sweep_judging, sweep_evidence,
    run_workflow, run_preflight, panel_workflow, run_readouts,
    rubric_inputs, forward_resolution, judge_resources,
    evaluation_evidence, evaluation_workflow, deferred_evaluation,
    execution_reporting)
"""], cwd=server, env={**os.environ, "PYTHONPATH": str(server)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_battery_capabilities_receive_the_battery_rendering_and_intervention():
    calls = []

    def generate(model, prompt, **kwargs):
        calls.append(("generate", model, prompt, kwargs))
        return "answer"

    def score(model, prompt, options, **kwargs):
        calls.append(("score", model, prompt, options, kwargs))
        return SimpleNamespace(selected="a", probability={"a": 0.75, "b": 0.25})

    arming = SimpleNamespace(max_tokens=31, prompt_mode="chat",
                             system_prompt="battery system", qwen_thinking_enabled=False)
    generate_fn, choice_fn = runtime_backends.battery_backends(
        "model", "model-id", ["injection"], ["latent"],
        generate_text=generate, score_choices=score)
    assert generate_fn("question", arming) == "answer"
    assert choice_fn("question", ("a", "b"), arming) == ("a", {"a": 0.75, "b": 0.25})
    common = dict(model_id="model-id", injections=["injection"],
                  latent_edits=["latent"], prompt_mode="chat",
                  system_prompt="battery system", qwen_thinking_enabled=False)
    assert calls == [
        ("generate", "model", "question", {**common, "max_tokens": 31, "temperature": 0.0}),
        ("score", "model", "question", ["a", "b"], common),
    ]
