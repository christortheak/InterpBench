"""Run identity snapshots and discovery; no model acquisition or task execution."""
from __future__ import annotations
import json
import os
from . import paths
from .manifest import Manifest
from .run_config import write_run_config


def _model_dtype(model) -> str:
    try:
        return str(next(model.model.parameters()).dtype)
    except (StopIteration, AttributeError):
        return ""



def _actual_dtype(model) -> str | None:
    """Canonical spelling of the dtype a loaded model ACTUALLY runs in, for
    the run stamp — or None when no model was loaded.

    Prefers the wrapper's own stamp (`model_loader.load` records it at load,
    already `torch.`-stripped); falls back to reading the parameters, which
    also works for the test fakes that never went through `load`.
    """
    if model is None:
        return None
    stamped = getattr(model, "dtype", None)
    if stamped:
        return str(stamped).removeprefix("torch.")
    raw = _model_dtype(model)
    return raw.removeprefix("torch.") or None



def _write_config_snapshot(manifest: Manifest, run_directory: str, task: str,
                           notes: dict | None = None, *, model=None,
                           job_id: str | None = None) -> None:
    with open(os.path.join(run_directory, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest.raw, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "experiment-hash.txt"), "w", encoding="utf-8") as handle:
        handle.write(manifest.content_hash() + "\n")
    with open(os.path.join(run_directory, "task.txt"), "w", encoding="utf-8") as handle:
        handle.write(task + "\n")
    # Canonical cross-engine per-run stamp (additive — never replaces the
    # richer per-task artifacts above). Sampling-policy fields are stamped
    # only for generation-bearing tasks (study runs — incl. the multi-agent
    # study path, which passes task "run" — and sweeps); extract/validate/
    # evaluate/analyze do not sample by the manifest's policy, so they stamp
    # null rather than inventing one.
    generates = task in ("run", "sweep")
    write_run_config(run_directory, task,
                     model_id=manifest.model_id, revision=manifest.model_revision,
                     experiment=manifest.name, experiment_hash=manifest.content_hash(),
                     temperature=manifest.temperature if generates else None,
                     samples_per_item=manifest.samples_per_item if generates else None,
                     seed_policy=manifest.seed_policy if generates else None,
                     # Schema 3: the precision the model ACTUALLY ran in.
                     # Stamped for every task that loaded a model, not just
                     # generation-bearing ones — extraction reads activations
                     # in that precision too, and the residual-norm
                     # denominator that alpha is expressed in comes with it.
                     dtype=_actual_dtype(model),
                     # Explicit for directories created OUTSIDE the producing
                     # job (shard merges, pipeline seeds, assembled on the
                     # controller): without it the env fallback stamped the
                     # CONTROLLER's own Slurm allocation id — four merged
                     # runs shared one stale jobId while their true shard
                     # ids sat in report.json's sharded block (2026-08-06).
                     job_id=job_id,
                     notes=notes)



def _latest_run(name: str, root: str | None) -> str | None:
    runs = paths.runs_directory(root)
    if not os.path.isdir(runs):
        return None
    candidates = sorted(
        (e for e in os.listdir(runs)
         if e.endswith(f"-exp-{name}-run")
         and os.path.isfile(os.path.join(runs, e, "generations.jsonl"))),
        reverse=True)
    return os.path.join(runs, candidates[0]) if candidates else None
