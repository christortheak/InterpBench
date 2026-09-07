"""Per-record seed identity and scoped RNG state. Importable without torch.

The seed algorithm and RNG draw order are scientific contracts. Runtime RNG
state is touched only when entering a non-greedy generation scope.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
from threading import RLock


# torch's default generators are process-global. Forking alone does not
# serialize overlapping scopes on different worker threads.
_generation_lock = RLock()


def derive_seed(experiment_hash: str, condition: str, prompt_id: str,
                sample_index: int) -> int:
    """Deterministic per-record seed for samplesPerItem runs (seedPolicy
    'derivedSHA256'): any (condition, prompt, sampleIndex) cell reproduces its
    exact sample stream on this substrate without a seeds table.

    Policy (asserted by tests, do not change silently): the derivation
    includes CONDITION identity, so baseline and every saved-agent/steered
    condition draw DISTINCT random streams for the same (prompt,
    sampleIndex) — the design is paired at the prompt level, not a
    common-random-numbers design across conditions. Anything that PAIRS
    records across conditions must therefore join on (promptID,
    sampleIndex), never the seed (``paired_judge._pair_generations``)."""
    blob = f"{experiment_hash}|{condition}|{prompt_id}|{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2 ** 63)



@contextmanager
def seeded_generation(temperature: float, seed: int):
    """Generation-local per-record seeding (RNG isolation, 2026-07-13).

    ``torch.manual_seed`` mutates PROCESS-GLOBAL RNG state, so two concurrent
    seeded studies interleaving records would corrupt each other's sample
    streams. Serialize overlapping scoped records and save the global RNG
    state (CPU, every visible CUDA device, and MPS) for the duration of one
    record's generation and seed INSIDE the scope: the
    record draws exactly the stream its seed names, and the global state is
    restored on exit — interleaved seeded records draw identically to serial
    ones. Unrelated code that bypasses this scope must not mutate torch's
    generators concurrently. Greedy records (``temperature <= 0``) never touch the RNG at all
    (preserves the resume byte-equality contract unchanged).
    """
    if not temperature or temperature <= 0:
        yield
        return
    import torch

    with _generation_lock:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        # fork_rng defaults to CUDA and does not restore MPS, although
        # manual_seed seeds MPS too. Restore it explicitly on every exit.
        mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
