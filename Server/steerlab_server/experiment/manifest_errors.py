"""Shared manifest authoring and lifecycle refusal contract."""
from __future__ import annotations

class ExperimentStoreError(Exception):
    """Authoring/lifecycle refusal.

    Carries an optional GATE id (WP0 step 3). The freeze path computed a
    closed-vocabulary id for every gate and then dropped it on the refusal
    path — ``raise ExperimentStoreError(gate_failures[0][1])`` — so only the
    ``forcedGatesSkipped`` stamp ever named a gate, and only the FIRST of N
    failures survived a refusal at all
    (``docs/WP0-AGENT-SURFACE-AUDIT.md`` §2.4). ``gate`` is the gate the
    message describes; ``gates`` is every gate that failed, in
    :data:`FORCED_GATE_IDS` order — the same order the stamp uses.

    Strictly additive: this class is caught broadly (CLI, HTTP routes,
    tasks), and ``str(exc)`` renders exactly the message it always did, so
    no refusal's human-visible prose or exit code changes. Swift twin:
    ``ExperimentError.freezeRefusal`` / ``FreezeRefusal``.
    """

    def __init__(self, message: str, *, gate: str | None = None,
                 gates: tuple[str, ...] | list[str] | None = None,
                 repair: str = ""):
        super().__init__(message)
        self.gate = gate
        #: The runnable repair (WP0 step 8). Read by the CLI's envelope
        #: builder through ``lifecycle_gates.repair_of``, so a refusal that
        #: knows its own remedy stops handing an agent boilerplate.
        self.repair_action = repair
        #: Every failing gate in vocabulary order; always contains ``gate``.
        self.gates: tuple[str, ...] = tuple(gates) if gates is not None else (
            (gate,) if gate else ())
