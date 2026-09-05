"""Choice/logprob measurements and explicit battery backend binding.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations

from . import cancellation
from . import condition_execution
from . import generate


def _score_choice(model, manifest, prompt: str, options, injections):
    """One choice-instrument readout, armed exactly as the study path arms it
    (`_run_impl`): same renderer, same manifest prompt config, same injector
    gating. Split out so tests can fake the scoring boundary."""
    from . import logprob
    return logprob.score_options(
        model, prompt, list(options), model_id=manifest.model_id,
        injections=injections, prompt_mode=manifest.prompt_mode,
        system_prompt=manifest.system_prompt,
        qwen_thinking_enabled=manifest.qwen_thinking_enabled)


def battery_backends(model, model_id: str, injections, latent_edits=None):
    """Compatibility boundary for callers injecting tasks.generate."""
    from .runtime_backends import battery_backends
    return battery_backends(model, model_id, injections, latent_edits,
                            generate_text=generate.generate)


def choice_target_logprobs(model, manifest, choice_rows, injections, *,
                            should_cancel=None, log=None) -> dict[str, float]:
    """Per-row joint logprob of the TARGET option under the given injections
    (``{row id: logP(target)}``). The option-length guard runs per row — the
    baseline pass calls this before any grid generation, so a guard failure
    aborts the sweep at start, never mid-grid. A cancel is observed between
    rows (``TaskCancelled``)."""
    out: dict[str, float] = {}
    total = len(choice_rows)
    for i, row in enumerate(choice_rows, start=1):
        cancellation.cancel_checkpoint(should_cancel, log, f"choice row {i}/{total}")
        choice = _score_choice(model, manifest, row.prompt, row.options, injections)
        condition_execution.check_option_lengths(choice, manifest, row.id)
        out[row.id] = next(
            s.logprob for s in choice.options if s.option == row.target)
    return out


def mean_logprob_shift(cell: dict[str, float], baseline: dict[str, float]) -> float:
    """logprobShift objective: mean over choice rows of
    logP(target | cell) − logP(target | baseline). 0 for the baseline cell
    by construction."""
    if not baseline:
        return 0.0
    return sum(cell.get(k, 0.0) - v for k, v in baseline.items()) / len(baseline)
