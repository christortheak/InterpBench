"""Shared cooperative cancellation contract; callers own partial evidence."""
from __future__ import annotations



def observe_cancel(should_cancel, log, where: str) -> bool:
    """True (and logs where) when a cancellation has been requested. Called at
    loop boundaries so a long task stops promptly instead of only at the end."""
    if should_cancel is not None and should_cancel():
        log(f"cancellation observed at {where}")
        return True
    return False



class TaskCancelled(Exception):
    """Internal sentinel: a cancellation was observed inside a generation loop
    (dev prompts, battery items, choice rows, judge calls). The loop that owns
    partial-output bookkeeping catches it and takes the SAME path as a
    concept/layer-boundary cancel — completed rows kept, partial CSV +
    recommendations written — so the run directory stays consistent."""



def cancel_checkpoint(should_cancel, log, where: str) -> None:
    """The per-generation twin of ``observe_cancel``: called between
    generations inside the inner loops so cancel latency is at most one
    generation, and raises ``TaskCancelled`` for the owning loop to catch."""
    if should_cancel is not None and should_cancel():
        if log is not None:
            log(f"cancel observed at {where} — stopping after the current generation")
        raise TaskCancelled(where)
