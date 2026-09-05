"""Run identity snapshots and discovery; no model acquisition or task execution."""
from __future__ import annotations
import json
import os
from . import paths, prompt_render
from .manifest import Manifest
from .run_config import write_run_config


def model_dtype(model) -> str:
    try:
        return str(next(model.model.parameters()).dtype)
    except (StopIteration, AttributeError):
        return ""



def actual_dtype(model) -> str | None:
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
    raw = model_dtype(model)
    return raw.removeprefix("torch.") or None



def write_config_snapshot(manifest: Manifest, run_directory: str, task: str,
                           notes: dict | None = None, *, model=None,
                           job_id: str | None = None,
                           root: str | None = None, log=None) -> None:
    with open(os.path.join(run_directory, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest.raw, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "experiment-hash.txt"), "w", encoding="utf-8") as handle:
        handle.write(manifest.content_hash() + "\n")
    with open(os.path.join(run_directory, "task.txt"), "w", encoding="utf-8") as handle:
        handle.write(task + "\n")
    # The model's chat-template capabilities this run rendered under
    # (2026-09-05): with a model in hand the loaded tokenizer is probed and
    # the workspace record ensured (written if absent, re-probed loudly if
    # the template changed); without one the workspace record is read. The
    # stamp lands in `notes.modelCapabilities` — the closed config.json key
    # set gains nothing — and the declared reasoning effort is checked
    # against it, as an ADVISORY: a submitted run never dies on a fact
    # probed after its freeze (post-submit drift policy), it continues
    # loudly and stamps.
    capabilities = _model_capabilities_for_run(manifest, model, root, log, task)
    if capabilities is not None:
        notes = dict(notes or {})
        notes["modelCapabilities"] = capabilities.stamp()
        if task in ("run", "sweep"):
            _advise_capabilities(manifest, capabilities, run_directory, log)
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
                     dtype=actual_dtype(model),
                     # Explicit for directories created OUTSIDE the producing
                     # job (shard merges, pipeline seeds, assembled on the
                     # controller): without it the env fallback stamped the
                     # CONTROLLER's own Slurm allocation id — four merged
                     # runs shared one stale jobId while their true shard
                     # ids sat in report.json's sharded block (2026-08-06).
                     job_id=job_id,
                     notes=notes)




def latest_run(name: str, root: str | None) -> str | None:
    runs = paths.runs_directory(root)
    if not os.path.isdir(runs):
        return None
    candidates = sorted(
        (e for e in os.listdir(runs)
         if e.endswith(f"-exp-{name}-run")
         and os.path.isfile(os.path.join(runs, e, "generations.jsonl"))),
        reverse=True)
    return os.path.join(runs, candidates[0]) if candidates else None


_RENDERING_RUN_TYPES = ("run", "sweep", "pipeline", "multi-agent")


def _model_capabilities_for_run(manifest: Manifest, model, root, log, task: str):
    """The capability view a run stamps: probed from the loaded tokenizer
    (ensuring the workspace record) when a model is present; the workspace
    record when a rendering run type was written without one; else None."""
    from . import model_capabilities as mc
    _log = log or print
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        config = getattr(getattr(model, "model", None), "config", None)
        revision = getattr(model, "revision", None) or manifest.model_revision
        try:
            return mc.ensure_record(
                tokenizer, model_id=manifest.model_id, revision=revision,
                config=config, root=root, log=_log)
        except Exception as exc:  # noqa: BLE001 - a probe must never sink a run
            _log(f"ADVISORY: could not probe the chat template of "
                 f"{manifest.model_id}: {exc}; stamping the workspace record "
                 "instead")
    if task not in _RENDERING_RUN_TYPES:
        return None
    try:
        return mc.lookup(manifest.model_id, manifest.model_revision, root)
    except Exception:  # noqa: BLE001
        return None



def _advise_capabilities(manifest: Manifest, capabilities, run_directory: str,
                         log) -> None:
    """Run-start advisory (never a refusal): the declared reasoning effort
    or system prompt against what the probed template actually does, plus
    the record's own notes. Appended to the run directory's advisories.txt
    like every other run-start advisory."""
    _log = log or print
    effort = manifest.reasoning_effort
    lines: list[str] = []
    if effort not in (prompt_render.REASONING_OFF, prompt_render.REASONING_ON):
        if capabilities.has_thinking_switch:
            lines += prompt_render.effort_level_violations(
                effort, manifest.model_id, capabilities)
        else:
            lines.append(prompt_render.effort_without_thinking_mode_reason(
                effort, manifest.model_id))
    lines += prompt_render.system_prompt_violations(
        system_prompt=manifest.system_prompt, model_id=manifest.model_id,
        prompt_mode=manifest.prompt_mode, capabilities=capabilities)
    lines += prompt_render.reasoning_protocol_advisories(
        effort=effort, model_id=manifest.model_id, capabilities=capabilities)
    for line in lines:
        _log(f"ADVISORY: {line}")
    if not lines:
        return
    try:
        with open(os.path.join(run_directory, "advisories.txt"), "a",
                  encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    except OSError:  # the advisory must never sink a run
        pass
