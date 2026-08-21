"""Canonical per-run ``config.json`` (pinned cross-engine schema, schema 4).

Every run-directory writer — extraction, validate, sweep, run, evaluate,
analyze, reader fit, norm backfill, LoRA train, variant save — writes this ONE
uniform stamp through this ONE helper, so writers cannot drift on field names.
It is **additive**: existing per-run artifacts (``experiment.json``,
``substrate.json``, ``validation-evidence.json``, …) are never replaced by it.

The Swift engine writes the same file with the same keys; ``substrate``
distinguishes the engines (``python-hf-transformers`` here, ``swift-mlx``
there). Unknown/inapplicable fields are ``null``, never omitted, so readers can
rely on the shape.

Schema rules (cross-engine contract, mirrored by Swift ``RunMetadata``):

- The top-level key set is **CLOSED**: adding, removing, or renaming a key
  requires bumping ``RUN_CONFIG_SCHEMA_VERSION`` and updating the pinned
  ``RUN_CONFIG_KEYS`` list in the SAME change (both engines' closed-key tests
  compare against the same literal list).
- Engine-specific extras live ONLY inside ``notes`` (an object, ``{}`` by
  default) — never as new top-level keys.
- No user paths, hostnames, or secrets anywhere in the payload: ``platform``
  is OS + architecture only.
- Run directories are immutable: an existing ``config.json`` is NEVER
  rewritten (a resumed run keeps its creation stamp; the legacy
  ``toy_concept`` run type keeps its historical, differently-shaped file).

Schema 2 (2026-07-12) added: ``runId`` (directory basename), ``platform``,
``temperature`` / ``samplesPerItem`` / ``seedPolicy`` (the sampling policy of
generation-bearing runs; null elsewhere), ``jobId`` (Slurm job id when run
under one), and ``notes``. Schema 1 keys kept their spellings so existing
readers (catalog listing, Swift RunBrowser) read both versions.

Schema 3 (2026-07-24) added: ``dtype`` — the numeric precision the model
ACTUALLY ran in, read off the loaded parameters (null for runs that load no
model). Greedy decoding is not precision-proof, so a measured run that does
not record its precision cannot be reproduced from its own record. The
server also stamps this per generation record and in ``substrate.json``;
schema 3 brings it into the canonical cross-engine stamp. Swift writes null:
its MLX models are quantized repos with no single parameter dtype, and Mac
runs are a testing substrate rather than a measurement one.

Schema 4 (2026-08-18, WP6 R1) added: ``pythonEnvironment`` — the resolved
versions of the science-relevant Python packages this run actually imported,
plus the interpreter version. ``pyproject.toml`` declares FLOORS, so
``appVersion`` pinned OUR code while the stack underneath it (torch,
transformers, numpy, scipy) was free to differ between two sites running the
same frozen manifest. Committed per-platform locks state the intended
resolution; this key records the achieved one, which is the half a reader can
verify. **Swift writes null** — the same engine-conditional shape ``dtype``
established in schema 3: the key is always present so the cross-engine shape
stays identical and a null reads as "this engine has no Python environment",
never as "nobody recorded it".
"""

from __future__ import annotations

import contextvars
import json
import os
import platform as _platform_mod
from datetime import datetime, timezone

from ..build_identity import engine_version
from ..python_environment import python_environment
from ..steering.vector_store import SUBSTRATE

RUN_CONFIG_SCHEMA_VERSION = 4
RUN_CONFIG_FILENAME = "config.json"

#: The CLOSED top-level key set (sorted). Byte-identical across engines —
#: Swift pins the same list in ``RunMetadata.contractKeys``.
RUN_CONFIG_KEYS = (
    "appVersion",
    "createdAt",
    "dtype",
    "experiment",
    "experimentHash",
    "jobId",
    "modelID",
    "notes",
    "platform",
    "pythonEnvironment",
    "revision",
    "runId",
    "runType",
    "samplesPerItem",
    "schemaVersion",
    "seedPolicy",
    "substrate",
    "temperature",
)


# The in-process job worker sets this so run directories it stamps carry the
# job id without threading a parameter through every task signature; Slurm
# children rely on the SLURM_JOB_ID env instead. ContextVar, not a global:
# concurrent worker threads each see only their own job.
current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "steerlab_current_job_id", default=None)


def run_platform() -> str:
    """OS + architecture, e.g. ``linux-x86_64`` / ``macOS-arm64``. Never a
    hostname (the stamp must carry no machine identity)."""
    system = {"Darwin": "macOS", "Linux": "linux", "Windows": "windows"}.get(
        _platform_mod.system(), _platform_mod.system().lower() or "unknown")
    machine = _platform_mod.machine() or "unknown"
    return f"{system}-{machine}"


def write_run_config(run_directory: str, run_type: str, *,
                     model_id: str | None = None,
                     revision: str | None = None,
                     experiment: str | None = None,
                     experiment_hash: str | None = None,
                     temperature: float | None = None,
                     samples_per_item: int | None = None,
                     seed_policy: str | None = None,
                     dtype: str | None = None,
                     job_id: str | None = None,
                     notes: dict | None = None) -> str:
    """Write the canonical ``config.json`` into ``run_directory``; returns the
    path. ``run_type`` is one of ``extract | validate | sweep | run | evaluate |
    analyze | reader-fit | norm-backfill | lora-train | variant-save |
    neutral-pcs | …`` (open set; hyphenated lowercase).

    Never rewrites an existing file: run directories are immutable, a resumed
    run's creation stamp stands, and some legacy run types own a
    differently-shaped ``config.json`` (mirror of Swift ``RunMetadata.write``).
    """
    path = os.path.join(run_directory, RUN_CONFIG_FILENAME)
    if os.path.exists(path):
        return path
    payload = {
        "schemaVersion": RUN_CONFIG_SCHEMA_VERSION,
        "runId": os.path.basename(os.path.normpath(run_directory)),
        "runType": run_type,
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "substrate": SUBSTRATE,
        # The engine build that wrote this run, matching the manifest's
        # freeze-time "appVersion" stamp. Since 2026-08-10 the string carries
        # the build identity ("steerlab-server 0.1.0+ab12cd34[-dirty]",
        # mirroring Swift's "swift-app …+sha8") — a declared version alone
        # could not distinguish two analyses of the same frozen source made
        # under different code (the endpoint-rescue reviewer finding).
        "appVersion": engine_version(),
        "platform": run_platform(),
        "modelID": model_id,
        "revision": revision,
        "experiment": experiment,
        "experimentHash": experiment_hash,
        "temperature": temperature,
        "samplesPerItem": samples_per_item,
        "seedPolicy": seed_policy,
        # The numeric precision the model ACTUALLY ran in, read off the
        # loaded parameters — not the dtype anything requested. Null for a
        # run that loaded no model (analyze, merges, pipeline roots).
        #
        # Greedy decoding is not precision-proof: at a near-tie between two
        # tokens, bf16 and fp16 round differently, the argmax flips, and the
        # continuation diverges. A measured run that does not say what
        # precision produced it cannot be reproduced from its own record.
        # The server also stamps this per generation record and in
        # substrate.json; here it joins the canonical cross-engine stamp.
        "dtype": dtype,
        # Schema 4: the measurement stack that actually ran, read off
        # importlib.metadata at write time. pyproject declares floors, so
        # without this a run's own record could not distinguish torch 2.11
        # from torch 2.13 — a difference that moves numbers on the substrate
        # the reproducibility claims live on. The committed platform lock
        # states the INTENDED resolution; this states the achieved one.
        "pythonEnvironment": python_environment(),
        "jobId": job_id if job_id is not None
        else (current_job_id.get() or os.environ.get("SLURM_JOB_ID") or None),
        "notes": dict(notes) if notes else {},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
