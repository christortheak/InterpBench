"""OptVec Slurm campaign: a grid of independent training cells, submitted in
top-up batches under the site's queue limits (plan of record:
``docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md`` §7 WP6).

A campaign is conditions × layers × seeds — and, since S4, optionally ×
ITEMS: an ``items`` axis turns each cell into a single-item optimization
(``itemFilter: [<id>]`` in the cell's train config), which is how the per-item
local directions and the multiplicity/fracture readout are produced (plan
§S4). The axis is purely additive: a campaign that declares no items plans,
materializes, submits, and reports exactly as it did before, cell ids
included.

One ``optvec train`` job per cell — deliberately NOT one Slurm array.
A site's GPU QOS commonly caps a user at
20 submitted / 8 running GPU jobs, so an array of 54 would be refused wholesale;
the campaign keeps at most ``maxQueued`` (default 15) of ITS OWN cells in the
queue and is re-invoked to top up as jobs drain (cron/loop friendly).

Four engine functions, deliberately separated so three of them are pure:

- :func:`plan` — deterministic cell list. Same config in, byte-identical
  plan out. Each cell's fully merged payload is validated by CONSTRUCTING an
  ``OptVecTrainConfig``, so a broken base config refuses here, at the desk,
  rather than 54 times on a billed allocation.
- :func:`materialize` — writes the immutable campaign directory (per-cell
  ``train-config.json`` + ``submit.sbatch``). Idempotent; REFUSES if a cell's
  train config already exists with different bytes (a drifted campaign is a
  NEW campaign, not an edit of this one).
- :func:`submit` — the top-up submission. The one mutating verb.
- :func:`status` — the read-only table.

Two rules are load-bearing and come from the operator's field experience:

1. **Never trust an exit code.** The submission path has exited 0 on a
   fan-out failure before. A submission is successful only when sbatch's
   STDOUT yields a job id ("Submitted batch job N"); anything else is
   recorded FAILED and reported loudly, whatever the exit status was.
   A failed submission never entered the scheduler, so it consumes NO
   attempt budget (observed live 2026-08-12: QOS submit-cap refusals burned
   two cells to "exhausted" without a single Slurm job existing for them) —
   only attempts that yielded a job id count toward ``1 + maxResubmits``.
2. **Absence must be positive.** A scheduler query that FAILS says nothing
   about a job; it never licenses a resubmit (the doctrine
   ``SlurmExecutor.find_job_by_name`` already states: at-most-once beats
   liveness). Only a successful ``squeue`` that does not list a job, plus a
   missing completion marker, makes a cell resubmittable.

Mutable state: ``campaign-state.json`` in the campaign directory is the ONE
file this module rewrites, and it is rewritten ATOMICALLY (tmp + fsync +
``os.replace``) after EVERY attempt — immediately on each submission,
adoption, or refusal, before the next ``sbatch`` goes out. Batching those
writes to the end of a cycle would mean a cron kill or a dropped login
session could erase N job ids that exist on the cluster, and the next top-up
would resubmit every one of them: duplicate 27B training jobs, billed. The
residual window is now one submission wide, and the name-based adoption in
:func:`submit` closes even that.

Everything else — the plan, the cell configs, the scripts, ``config.json`` —
is written once and never touched again, keeping the run directory immutable
in the sense the rest of the engine means it. Completion is recorded OUTSIDE
the state file, by the ``COMPLETED`` marker the cell's own sbatch script
drops on success, so a lost state file can never make a finished cell look
unfinished-and-resubmittable.

Imports here are deliberately light at module scope (stdlib + the executor's
site-data helpers): ``optvec_train`` pulls in torch, and a campaign is
planned/submitted/polled from a login node where importing torch is pure
tax. It is imported lazily inside the functions that validate a cell payload.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from ..api import executors
from ..api.profile import ServerProfile
from . import paths
from .run_config import write_run_config

#: Run type stamped into the campaign directory's ``config.json``.
RUN_TYPE = "optvec-campaign"

#: Campaign-artifact schema (``campaign.json`` / ``campaign-state.json``).
CAMPAIGN_SCHEMA = 1

CAMPAIGN_FILENAME = "campaign.json"
STATE_FILENAME = "campaign-state.json"
CELLS_DIRNAME = "cells"
CELL_CONFIG_FILENAME = "train-config.json"
CELL_SCRIPT_FILENAME = "submit.sbatch"

#: Written by the cell's own sbatch script, and ONLY on a zero-exit train
#: (``… && touch``). Its presence is the completion authority: state files can
#: be lost or stale, a marker cannot be written by a job that failed.
COMPLETION_MARKER = "COMPLETED"

#: Sanity cap on the grid. A typo'd seedCount of 800 must refuse at plan time,
#: not become 800 sbatch calls. Config-overridable via ``maxCells``.
DEFAULT_MAX_CELLS = 500

#: Observed GPU QOS: 20 submitted / 8 running GPU jobs per user. 15 leaves head
#: room for the researcher's interactive/GPU-session jobs under the same cap.
DEFAULT_MAX_QUEUED = 15

DEFAULT_JOB_NAME_PREFIX = "optvec"

#: Characters of a slugged item id kept in a cell id before the disambiguating
#: hash. Cell ids become directory names and scheduler job names, so an item id
#: like ``"Case Family B §4(b) / variant 2"`` cannot ride verbatim.
ITEM_SLUG_MAX = 32

#: Hex characters of ``sha256(raw item id)`` appended when the slug is lossy
#: (see :func:`item_slug_map`). Eight is the project's usual short-hash width.
ITEM_SLUG_HASH_CHARS = 8

#: Resubmits per cell, beyond the first attempt (so 3 attempts total). A cell
#: that dies three times is a bug, not bad luck: it becomes ``exhausted`` and
#: waits for a human. Only attempts that actually ENTERED the scheduler
#: (sbatch returned a job id, or an existing job was adopted by name) count
#: against this budget — see :func:`counted_attempts`.
DEFAULT_MAX_RESUBMITS = 2

#: Consecutive submission failures that stop a top-up cycle. The usual cause
#: (queue limit, account/QoS refusal) applies to every subsequent sbatch in
#: the same cycle, and re-invocation is free — but one permanently broken
#: cell must not block the rest of the grid forever, so this is a small
#: number rather than "stop on the first failure".
MAX_CONSECUTIVE_SUBMIT_FAILURES = 3

#: sbatch's success line. Stricter than ``SlurmExecutor.submit``'s
#: ``stdout.split()[-1]`` on purpose: that form turns any chatty zero-exit
#: output into a "job id".
_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")

#: Scheduler states that mean the job still exists for the queue accounting.
_QUEUED_STATES = frozenset({"submitted"})
_RUNNING_STATES = frozenset({"running", "checkpointed"})
ALIVE_STATES = _QUEUED_STATES | _RUNNING_STATES


class CampaignConfigError(ValueError):
    """A campaign config that cannot be planned as written."""


class CampaignError(RuntimeError):
    """A campaign directory that cannot be materialized or submitted."""


# ------------------------------------------------------------------ config


#: Campaign JSON key → dataclass field for the slurm block. Everything that
#: maps onto ``SlurmResources`` keeps that dataclass's meaning exactly (site
#: facts are data, not code — the executor's rule); the three extras
#: (maxQueued / jobNamePrefix / maxResubmits) are campaign SUBMISSION POLICY,
#: which the executor has no concept of. Site directives with no
#: ``SlurmResources`` field (``--constraint``, ``--qos``, …) ride in
#: ``extraSbatch`` verbatim, exactly as they do for every other bundle.
_SLURM_KEYS = {
    "partition": "partition", "account": "account", "gres": "gres",
    "gpus": "gpus", "memory": "memory", "walltime": "walltime",
    "cpusPerTask": "cpus_per_task", "extraSbatch": "extra_sbatch",
    "requeue": "requeue", "useSrun": "use_srun", "exportNone": "export_none",
    "signalSeconds": "signal_seconds", "signalTarget": "signal_target",
    "maxQueued": "max_queued", "jobNamePrefix": "job_name_prefix",
    "maxResubmits": "max_resubmits",
}


@dataclass(frozen=True)
class CampaignSlurm:
    partition: str | None = None
    account: str | None = None
    gres: str | None = None
    gpus: int = 1
    memory: str | None = None
    walltime: str | None = None
    cpus_per_task: int = 4
    extra_sbatch: tuple[str, ...] = ()
    requeue: bool = False
    use_srun: bool = True
    export_none: bool = True
    #: 0 = no ``--signal`` directive, the campaign DEFAULT. The training
    #: driver installs no SIGUSR1 handler (unlike the generation loop's
    #: resume machinery), and Python's default action for SIGUSR1 is to die —
    #: so a walltime-warning signal would kill a cell early and gain nothing.
    #: Sites that wire a handler can set it.
    signal_seconds: int = 0
    signal_target: str = "step"
    max_queued: int = DEFAULT_MAX_QUEUED
    job_name_prefix: str = DEFAULT_JOB_NAME_PREFIX
    max_resubmits: int = DEFAULT_MAX_RESUBMITS

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CampaignSlurm":
        payload = payload or {}
        if not isinstance(payload, dict):
            raise CampaignConfigError("'slurm' must be a JSON object")
        unknown = sorted(set(payload) - set(_SLURM_KEYS))
        if unknown:
            raise CampaignConfigError(
                "unknown slurm key(s): " + ", ".join(unknown)
                + " — site directives without a dedicated field ride in "
                  "'extraSbatch' (e.g. [\"--constraint=A100\"])")
        kwargs = {field_name: payload[key]
                  for key, field_name in _SLURM_KEYS.items() if key in payload}
        if "extra_sbatch" in kwargs:
            raw = kwargs["extra_sbatch"]
            if not isinstance(raw, list) or any(
                    not isinstance(item, str) for item in raw):
                raise CampaignConfigError(
                    "slurm.extraSbatch must be a list of raw sbatch "
                    "directives, e.g. [\"--constraint=A100\"]")
            kwargs["extra_sbatch"] = tuple(raw)
        slurm = cls(**kwargs)
        if slurm.max_queued < 1:
            raise CampaignConfigError("slurm.maxQueued must be at least 1")
        if slurm.max_resubmits < 0:
            raise CampaignConfigError("slurm.maxResubmits must be >= 0")
        if not str(slurm.job_name_prefix).strip():
            raise CampaignConfigError("slurm.jobNamePrefix must be non-empty")
        return slurm

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for key, field_name in _SLURM_KEYS.items():
            value = getattr(self, field_name)
            out[key] = list(value) if isinstance(value, tuple) else value
        return out

    def resources(self, job_name: str) -> executors.SlurmResources:
        """The executor's own resource record — the site vocabulary (GPU
        types, VRAM table, poll-command names, requeue/auto-resubmit env
        defaults) comes from ``SlurmResources.from_env`` and only the keys
        the campaign declared are overridden."""
        base = executors.SlurmResources.from_env(job_name=job_name)
        return executors.SlurmResources(
            job_name=job_name,
            partition=self.partition if self.partition is not None else base.partition,
            gres=self.gres if self.gres is not None else base.gres,
            gpus=self.gpus,
            memory=self.memory if self.memory is not None else base.memory,
            walltime=self.walltime or base.walltime,
            cpus_per_task=self.cpus_per_task,
            signal_seconds=self.signal_seconds,
            signal_target=self.signal_target,
            use_srun=self.use_srun,
            export_none=self.export_none,
            account=self.account if self.account is not None else base.account,
            requeue=self.requeue,
            gpu_types=base.gpu_types,
            gpu_vram_gb=base.gpu_vram_gb,
            sacct_command=base.sacct_command,
            squeue_command=base.squeue_command,
            sbatch_command=base.sbatch_command,
            scancel_command=base.scancel_command,
            # Site placement directives + required headers ride along too (WP5
            # Step 8): a campaign cell is an ordinary job at this site, so the
            # site's QOS, node features, reservation and header requirements
            # apply to it exactly as to a study job. The campaign's own
            # extra_sbatch stays the per-request layer on top.
            qos=base.qos,
            constraints=list(base.constraints),
            reservation=base.reservation,
            required_headers=list(base.required_headers),
            extra_sbatch=list(self.extra_sbatch) or list(base.extra_sbatch),
        )


@dataclass(frozen=True)
class GridCondition:
    """One arm of the grid (plan §2: S0 shuffled-target null, S1 shift-only,
    S2 composite, S3 multiplicity). ``overrides`` is a PARTIAL
    ``OptVecTrainConfig`` payload deep-merged over the base — S1 is
    ``{"lambdaAnchor": 0, "lambdaCap": 0}``, S0 is
    ``{"shuffleTargetLabels": true}``. The condition contributes no code."""

    name: str
    overrides: dict = field(default_factory=dict)


#: Keys the GRID owns. A condition that also set them would silently
#: contradict the axis it is crossed with, and the cell id would lie.
#: ``itemFilter`` is owned unconditionally — including in campaigns that
#: declare no items axis — because a per-condition item filter crossed with a
#: grid whose ids say nothing about items is exactly the silent contradiction
#: this rule exists to prevent. A campaign that really trains every cell on
#: one item states it once, in ``baseConfig``.
_GRID_OWNED_KEYS = ("layer", "seed", "name", "itemFilter")


@dataclass(frozen=True)
class OptVecCampaignConfig:
    name: str
    base_config: dict
    layers: tuple[int, ...]
    conditions: tuple[GridCondition, ...]
    seeds: tuple[int, ...]
    slurm: CampaignSlurm = field(default_factory=CampaignSlurm)
    max_cells: int = DEFAULT_MAX_CELLS
    #: The optional per-item axis: RAW item ids, in declaration (or file) order.
    #: Empty is the historical campaign, unchanged in every respect.
    items: tuple[str, ...] = ()
    #: Provenance when the axis came from ``grid.itemsFile``: the path read and
    #: the SHA-256 of its bytes. Carried through ``to_dict`` so a materialized
    #: campaign records which file's ids it is a grid over, and so a re-read of
    #: ``campaign.json`` never has to touch that file again.
    items_file: dict | None = None

    @property
    def item_slugs(self) -> dict[str, str]:
        """Raw item id → the token that appears in cell ids (see
        :func:`item_slug_map`). Recomputed rather than stored: the slug rule is
        deterministic, so a stored map can only ever disagree."""
        return item_slug_map(self.items)

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecCampaignConfig":
        if not isinstance(payload, dict):
            raise CampaignConfigError("the campaign config must be a JSON object")
        known = {"name", "baseConfig", "grid", "slurm", "maxCells", "itemAxis"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise CampaignConfigError(
                "unknown campaign config key(s): " + ", ".join(unknown))
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise CampaignConfigError("'name' is required")
        if _slugify(name) != name.strip():
            raise CampaignConfigError(
                f"campaign name {name!r} must be a slug (lowercase letters, "
                "digits and hyphens): it names a run directory and every "
                "cell's job")
        base = payload.get("baseConfig")
        if not isinstance(base, dict):
            raise CampaignConfigError(
                "'baseConfig' must be a complete OptVecTrainConfig payload — "
                "the grid only varies layer, seed, and the condition's own "
                "overrides")
        grid = payload.get("grid")
        if not isinstance(grid, dict):
            raise CampaignConfigError("'grid' must be an object")
        grid_unknown = sorted(set(grid) - {"layers", "conditions", "seeds",
                                           "seedCount", "items", "itemsFile"})
        if grid_unknown:
            raise CampaignConfigError(
                "unknown grid key(s): " + ", ".join(grid_unknown))

        items, items_file = _read_item_axis(grid, payload.get("itemAxis"))
        if items and "itemFilter" in base:
            raise CampaignConfigError(
                "baseConfig sets 'itemFilter' while grid declares an items "
                "axis — the grid owns itemFilter there, and a base value "
                "would be silently overwritten in every cell")

        layers = grid.get("layers")
        if not isinstance(layers, list) or not layers:
            raise CampaignConfigError(
                "grid.layers must be a non-empty list of layer indices")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in layers):
            raise CampaignConfigError(
                "grid.layers must be non-negative integers")
        if len(set(layers)) != len(layers):
            raise CampaignConfigError(
                "grid.layers repeats a layer — a duplicated axis value would "
                "produce two cells with the same id")

        if "seeds" in grid and "seedCount" in grid:
            raise CampaignConfigError(
                "declare grid.seeds OR grid.seedCount, never both — two "
                "seed declarations cannot both be the record of what ran")
        if "seeds" in grid:
            seeds = grid["seeds"]
            if not isinstance(seeds, list) or not seeds:
                raise CampaignConfigError(
                    "grid.seeds must be a non-empty list of integers")
            if any(not isinstance(v, int) or isinstance(v, bool) for v in seeds):
                raise CampaignConfigError("grid.seeds must be integers")
            if len(set(seeds)) != len(seeds):
                raise CampaignConfigError("grid.seeds repeats a seed")
            seed_tuple = tuple(seeds)
        elif "seedCount" in grid:
            count = grid["seedCount"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise CampaignConfigError(
                    "grid.seedCount must be a positive integer (it expands to "
                    "0..N-1)")
            seed_tuple = tuple(range(count))
        else:
            raise CampaignConfigError(
                "grid needs seeds or seedCount — a campaign whose seeds are "
                "implicit cannot say what it ran")

        raw_conditions = grid.get("conditions")
        if not isinstance(raw_conditions, list) or not raw_conditions:
            raise CampaignConfigError(
                "grid.conditions must be a non-empty list of "
                "{name, overrides} objects")
        conditions: list[GridCondition] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_conditions):
            if not isinstance(raw, dict):
                raise CampaignConfigError(
                    f"grid.conditions[{index}] must be an object")
            extra = sorted(set(raw) - {"name", "overrides"})
            if extra:
                raise CampaignConfigError(
                    f"unknown condition key(s) in grid.conditions[{index}]: "
                    + ", ".join(extra))
            cond_name = raw.get("name")
            if not isinstance(cond_name, str) or not cond_name.strip():
                raise CampaignConfigError(
                    f"grid.conditions[{index}].name is required")
            overrides = raw.get("overrides", {})
            if not isinstance(overrides, dict):
                raise CampaignConfigError(
                    f"grid.conditions[{index}].overrides must be an object "
                    "(a partial OptVecTrainConfig payload)")
            owned = sorted(k for k in _GRID_OWNED_KEYS if k in overrides)
            if owned:
                raise CampaignConfigError(
                    f"grid.conditions[{index}].overrides sets grid-owned "
                    f"key(s): {', '.join(owned)} — layer, seed and itemFilter "
                    "are grid axes and the cell name is derived from them")
            slug = _slugify(cond_name)
            if not slug:
                raise CampaignConfigError(
                    f"condition name {cond_name!r} has no slug form")
            if slug in seen:
                raise CampaignConfigError(
                    f"two conditions slug to {slug!r} — cell ids would collide")
            seen.add(slug)
            conditions.append(GridCondition(name=slug, overrides=overrides))

        max_cells = payload.get("maxCells", DEFAULT_MAX_CELLS)
        if not isinstance(max_cells, int) or isinstance(max_cells, bool) \
                or max_cells < 1:
            raise CampaignConfigError("maxCells must be a positive integer")

        return cls(name=name.strip(), base_config=base,
                   layers=tuple(layers), conditions=tuple(conditions),
                   seeds=seed_tuple,
                   slurm=CampaignSlurm.from_dict(payload.get("slurm")),
                   max_cells=max_cells, items=items, items_file=items_file)

    def to_dict(self) -> dict:
        grid: dict[str, Any] = {
            "layers": list(self.layers),
            "seeds": list(self.seeds),
            "conditions": [{"name": c.name, "overrides": c.overrides}
                           for c in self.conditions],
        }
        payload = {"name": self.name, "baseConfig": self.base_config,
                   "grid": grid, "slurm": self.slurm.to_dict(),
                   "maxCells": self.max_cells}
        if self.items:
            # The canonical form of the axis is always the RESOLVED id list
            # (as ``seedCount`` canonicalizes to ``seeds``): re-reading a
            # campaign must never depend on a workspace file still being
            # there. ``itemAxis`` is the provenance beside it — the slug map a
            # reader needs to map a cell id back to an item, and the file the
            # ids came from.
            grid["items"] = list(self.items)
            payload["itemAxis"] = {"slugs": self.item_slugs,
                                   "file": self.items_file}
        return payload


def load_config(path: str) -> OptVecCampaignConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecCampaignConfig.from_dict(json.load(handle))


# -------------------------------------------------------------------- plan


@dataclass(frozen=True)
class GridPoint:
    """One cell's COORDINATES, before its train config is merged or validated.

    Split out from :class:`Cell` so the grid can be enumerated without
    importing the training driver (and therefore torch): the cap check, the
    cell ids and the item slugs are answerable on a login node, and are
    testable without a model stack.
    """

    cell_id: str
    condition: str
    layer: int
    seed: int
    #: The raw item id this cell optimizes on, and the token in its cell id.
    #: ``None`` on both when the campaign declares no items axis.
    item: str | None = None
    item_slug: str | None = None


@dataclass(frozen=True)
class Cell:
    """One independent training job: a condition at a layer with a seed (and,
    on a per-item campaign, an item)."""

    cell_id: str
    condition: str
    layer: int
    seed: int
    #: The fully merged, canonicalized ``OptVecTrainConfig`` payload.
    config: dict
    #: SHA-256 over that payload's canonical JSON — the cell's identity.
    config_hash: str
    item: str | None = None
    item_slug: str | None = None

    def to_dict(self) -> dict:
        payload = {"cellID": self.cell_id, "condition": self.condition,
                   "layer": self.layer, "seed": self.seed,
                   "configHash": self.config_hash}
        if self.item is not None:
            # Absent, not null, on a campaign without an items axis: a
            # non-item campaign's campaign.json is byte-identical to what it
            # was before the axis existed, and queued campaigns keep working.
            payload["item"] = self.item
            payload["itemSlug"] = self.item_slug
        return payload


def grid_points(config: OptVecCampaignConfig) -> list[GridPoint]:
    """The campaign's cell coordinates, in a deterministic order: conditions
    outer, then layers, then ITEMS, then seeds — declaration order throughout.

    Items sit OUTSIDE seeds deliberately. Seeds are restarts, and the per-item
    readout (how many distinct basins an item's optimization lands in) needs a
    COMPLETE restart set per item; a campaign that is drained early, or capped
    by the queue, then yields whole items rather than one restart of each.
    """
    conditions, layers = len(config.conditions), len(config.layers)
    items = config.items or ()
    total = conditions * layers * len(config.seeds) * max(1, len(items))
    if total > config.max_cells:
        axes = (f"{conditions} conditions × {layers} layers"
                + (f" × {len(items)} items" if items else "")
                + f" × {len(config.seeds)} seeds")
        raise CampaignConfigError(
            f"campaign {config.name!r} plans {total} cells "
            f"({axes}), over the cap of "
            f"{config.max_cells}. Raise 'maxCells' deliberately if that is "
            "really the campaign — the cap exists so a typo cannot "
            "mass-submit")

    slugs = config.item_slugs if items else {}
    points: list[GridPoint] = []
    for condition in config.conditions:
        for layer in config.layers:
            for item in (items or (None,)):
                slug = slugs.get(item) if item is not None else None
                for seed in config.seeds:
                    cell_id = (f"{condition.name}-L{layer}-s{seed}"
                               if item is None else
                               f"{condition.name}-L{layer}-i{slug}-s{seed}")
                    points.append(GridPoint(
                        cell_id=cell_id, condition=condition.name, layer=layer,
                        seed=seed, item=item, item_slug=slug))
    return points


def plan(config: OptVecCampaignConfig) -> list[Cell]:
    """The campaign's cells, in :func:`grid_points` order.

    Every cell payload is validated by constructing an ``OptVecTrainConfig``,
    which is what makes a broken base config a desk-time refusal. The stored
    payload is that object's own ``to_dict``, so the bytes are canonical
    (every key explicit) rather than however the author happened to write
    them, and the hash identifies the CONFIGURATION, not the authoring style.
    S3 priors that are resolvable on this machine are preflighted per cell
    layer as well (:func:`_preflight_prior_rows`).
    """
    from .optvec_train import OptVecConfigError, OptVecTrainConfig  # lazy: torch

    by_condition = {condition.name: condition for condition in config.conditions}
    cells: list[Cell] = []
    for point in grid_points(config):
        merged = _deep_merge(config.base_config,
                             by_condition[point.condition].overrides)
        merged["layer"] = point.layer
        merged["seed"] = point.seed
        merged["name"] = f"{config.name}-{point.cell_id}"
        if point.item is not None:
            # The grid owns this key (see ``_GRID_OWNED_KEYS``): one item per
            # cell, named by the cell id, filtered on the TRAIN pool only.
            merged["itemFilter"] = [point.item]
        try:
            canonical = OptVecTrainConfig.from_dict(merged).to_dict()
        except OptVecConfigError as exc:
            raise CampaignConfigError(
                f"cell {point.cell_id} would not run: {exc}") from exc
        cells.append(Cell(
            cell_id=point.cell_id, condition=point.condition, layer=point.layer,
            seed=point.seed, config=canonical,
            config_hash=_hash_payload(canonical),
            item=point.item, item_slug=point.item_slug))
    _preflight_prior_rows(cells)
    return cells


def _preflight_prior_rows(cells: list[Cell]) -> None:
    """Desk-time twin of the run-time zero-row gate in
    ``optvec_train._load_prior_vectors``.

    A campaign that crosses a fixed ``priorVectorPaths`` list with
    ``grid.layers`` plans cells whose prior is all zeros at the cell's layer
    (an OptVec artifact is nonzero only at its own optimization layer), and
    the run-time loader refuses those — but only after the cell has queued
    and started on the cluster. When the artifact bytes are HERE, that is
    answerable at planning time, so refuse before anything is submitted.

    When they are not here, stay silent: planning may happen on a machine
    that never holds the vectors (the Mac authors, the cluster executes;
    bundles ship them), so an artifact that cannot be resolved and loaded is
    NOT a desk-time refusal. Only a successful load may refuse — any load
    problem is left to the run-time gate, which reads the authoritative
    bytes next to the job.
    """
    from ..steering import vector_store  # lazy, like optvec_train above

    loaded: dict[str, Any] = {}
    for cell in cells:
        if not cell.config.get("lambdaOrth", 0) > 0:
            continue
        for reference in cell.config.get("priorVectorPaths", ()):
            if reference not in loaded:
                resolved = paths.resolve_artifact(reference)
                try:
                    vectors, _sidecar = vector_store.load(
                        os.path.dirname(resolved), os.path.basename(resolved))
                except Exception:
                    vectors = None   # absent or unreadable: run time decides
                loaded[reference] = vectors
            vectors = loaded[reference]
            if vectors is None:
                continue
            if cell.layer >= vectors.layer_count:
                raise CampaignConfigError(
                    f"cell {cell.cell_id} would not run: prior vector "
                    f"'{reference}' has {vectors.layer_count} layers; the "
                    f"cell's layer is {cell.layer}")
            if not any(vectors.per_layer[cell.layer]):
                raise CampaignConfigError(
                    f"cell {cell.cell_id} would not run: prior vector "
                    f"'{reference}' is all zeros at layer {cell.layer} — an "
                    "OptVec artifact is nonzero only at its own optimization "
                    "layer, so this prior would contribute exactly zero "
                    "orthogonality pressure while lambdaOrth stamps the cell "
                    "as S3. Use a prior trained at this cell's layer, or "
                    "drop it from that condition's priorVectorPaths")


# ------------------------------------------------------------- materialize


def materialize(config: OptVecCampaignConfig, root: str | None = None, *,
                campaign_dir: str | None = None) -> str:
    """Write (or re-write) the campaign directory; returns its path.

    Idempotent for the same directory: re-running after adding cells to the
    grid materializes the new ones and leaves the existing ones byte-for-byte
    alone. It REFUSES when an existing cell's ``train-config.json`` differs
    from what this config plans — a campaign whose cells changed underneath
    submitted jobs is a different campaign and must get its own directory.
    """
    cells = plan(config)
    if campaign_dir is None:
        campaign_dir = paths.make_unique_run_directory(
            f"optvec-campaign-{config.name}", root)
    os.makedirs(campaign_dir, exist_ok=True)

    profile = ServerProfile.from_env()
    for cell in cells:
        cell_dir = os.path.join(campaign_dir, CELLS_DIRNAME, cell.cell_id)
        os.makedirs(cell_dir, exist_ok=True)
        config_path = os.path.join(cell_dir, CELL_CONFIG_FILENAME)
        payload = _canonical_bytes(cell.config)
        if os.path.exists(config_path):
            with open(config_path, "rb") as handle:
                existing = handle.read()
            if existing != payload:
                raise CampaignError(
                    f"cell {cell.cell_id} already exists in {campaign_dir} "
                    "with a DIFFERENT train config. A drifted campaign is a "
                    "new campaign: materialize into a fresh directory rather "
                    "than editing one whose cells may already have run")
        else:
            with open(config_path, "wb") as handle:
                handle.write(payload)
        # The script is regenerated every time: it carries SITE facts
        # (partition, walltime, module/venv reconstruction) which may
        # legitimately change between top-ups, and it is not what identifies
        # the cell scientifically. The train config is.
        script = render_cell_script(config, cell, cell_dir, profile=profile)
        with open(os.path.join(cell_dir, CELL_SCRIPT_FILENAME), "w",
                  encoding="utf-8") as handle:
            handle.write(script)

    campaign_payload = {
        "schemaVersion": CAMPAIGN_SCHEMA,
        "name": config.name,
        "config": config.to_dict(),
        "cells": [cell.to_dict() for cell in cells],
        "completionMarker": COMPLETION_MARKER,
    }
    campaign_path = os.path.join(campaign_dir, CAMPAIGN_FILENAME)
    with open(campaign_path, "wb") as handle:
        handle.write(_canonical_bytes(campaign_payload))

    grid_notes: dict[str, Any] = {
        "layers": list(config.layers),
        "seeds": list(config.seeds),
        "conditions": [c.name for c in config.conditions]}
    if config.items:
        grid_notes["items"] = list(config.items)
        grid_notes["itemSlugs"] = config.item_slugs
        grid_notes["itemsFile"] = config.items_file
    write_run_config(
        campaign_dir, RUN_TYPE,
        model_id=config.base_config.get("modelID"),
        revision=config.base_config.get("revision"),
        notes={
            "campaign": config.name,
            "cellCount": len(cells),
            "grid": grid_notes,
            "submission": {"maxQueued": config.slurm.max_queued,
                           "maxResubmits": config.slurm.max_resubmits,
                           "jobNamePrefix": config.slurm.job_name_prefix},
            "cells": [cell.to_dict() for cell in cells],
            "claim": "sufficiency",
        })
    return campaign_dir


def render_cell_script(config: OptVecCampaignConfig, cell: Cell,
                       cell_dir: str, *,
                       profile: ServerProfile | None = None) -> str:
    """One cell's sbatch script, templated by the EXISTING site machinery
    (``SlurmResources`` + ``executors.render_slurm_script``) so the campaign
    inherits every hard-won cluster fact — ``--export=NONE`` plus the
    ``SLURM_EXPORT_ENV=ALL`` counter-trap, module/conda/venv reconstruction,
    the shared HF cache and node-staging propagation, the secret-shaped-env
    refusal, ``--ntasks=1``.

    The one campaign-specific thing is the completion marker: the job's
    command is ``<train> && touch <marker>``, so the marker exists if and
    only if the training process exited zero. That marker, not a state file,
    is what makes a cell "done" — and a dead cell WITHOUT one is exactly
    what a top-up should resubmit.
    """
    profile = profile or ServerProfile.from_env()
    python = os.environ.get("STEERLAB_PYTHON") or sys.executable or "python"
    train_command = [python, "-m", "steerlab_server.cli", "optvec", "train",
                     "--config", os.path.join(cell_dir, CELL_CONFIG_FILENAME)]
    marker = os.path.join(cell_dir, COMPLETION_MARKER)
    payload = (" ".join(shlex.quote(part) for part in train_command)
               + " && touch " + shlex.quote(marker))
    campaign_dir = os.path.dirname(os.path.dirname(os.path.normpath(cell_dir)))
    resources = config.slurm.resources(
        job_name=job_name_for(config, cell.cell_id, campaign_dir))
    env = _cell_env(profile)
    executors._refuse_secret_env(env)
    bundle = executors.JobBundle(
        bundle_dir=cell_dir,
        command=["bash", "-c", payload],
        env=env,
        resources=resources,
        stdout_path=os.path.join(cell_dir, "slurm-%j.out"),
        stderr_path=os.path.join(cell_dir, "slurm-%j.err"),
        script_path=os.path.join(cell_dir, CELL_SCRIPT_FILENAME),
        manifest_path=os.path.join(cell_dir, "bundle.json"),
    )
    return executors.render_slurm_script(bundle)


def campaign_identity(campaign_dir: str) -> str:
    """Eight hex chars binding scheduler job names to THIS materialized
    campaign directory. ``config.name`` is a human label two campaigns can
    legally share, and adoption-by-name would then let one campaign adopt —
    and count against its occupancy — the other's jobs (review finding
    2026-08-10). The run-directory basename is unique by construction
    (``make_unique_run_directory``); hashing it keeps the token fixed-width
    and scheduler-safe."""
    basename = os.path.basename(os.path.normpath(campaign_dir))
    return hashlib.sha256(basename.encode("utf-8")).hexdigest()[:8]


def job_name_for(config: OptVecCampaignConfig, cell_id: str,
                 campaign_dir: str) -> str:
    """``<prefix>-<campaign>-<identity8>-<cellID>`` — the scheduler-findable
    token. Passed as ``sbatch --job-name`` too, so a controller that died
    between sbatch and its own state write can still find the submission by
    name — and ONLY this campaign's submission, thanks to the identity
    token."""
    return (f"{config.slurm.job_name_prefix}-{config.name}-"
            f"{campaign_identity(campaign_dir)}-{cell_id}")


def _cell_env(profile: ServerProfile) -> dict[str, str]:
    """The bundle environment, mirroring ``SlurmExecutor.create_bundle``'s
    inheritance list (kept in step with it deliberately; the campaign builds
    its own JobBundle because its artifacts are named per cell)."""
    env: dict[str, str] = {"STEERLAB_ROOT": profile.root}
    if profile.asset_root:
        env.setdefault("STEERLAB_ASSET_ROOT", profile.asset_root)
    if profile.run_root:
        env.setdefault("STEERLAB_RUN_ROOT", profile.run_root)
    if profile.node_cache_root:
        env.setdefault("STEERLAB_NODE_CACHE_ROOT", profile.node_cache_root)
        env.setdefault("HF_HOME", profile.node_cache_root)
    # PYTORCH_CUDA_ALLOC_CONF rides along when the site sets it: long-prompt
    # training under gradient-checkpointing recompute churn fragments the
    # allocator (observed live 2026-08-11 on a hard-item campaign: ~1 GB
    # reserved-unallocated while a 180 MB request failed), and
    # expandable_segments is the documented remedy.
    for key in ("HF_HOME", "HF_HUB_OFFLINE", "STEERLAB_NODE_STAGE_DIR",
                "PYTORCH_CUDA_ALLOC_CONF"):
        if os.environ.get(key):
            env.setdefault(key, os.environ[key])
    return env


# ------------------------------------------------------------------ runner


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner:
    """The default command executor: the same ``subprocess.run(text=True,
    capture_output=True, check=False)`` call the Slurm executor makes, behind
    an injectable seam so every test runs with a fake and no test needs a
    scheduler on PATH."""

    def __call__(self, command: list[str], *,
                 cwd: str | None = None) -> CommandResult:
        proc = subprocess.run(command, text=True, capture_output=True,
                              check=False, cwd=cwd)
        return CommandResult(proc.returncode, proc.stdout or "",
                             proc.stderr or "")


Runner = Callable[..., CommandResult]


def parse_job_id(stdout: str) -> str | None:
    """The job id in ``Submitted batch job 12345``, or None.

    None is the ONLY thing that matters about a submission's success: the
    submission CLI has exited 0 on a fan-out failure before, so a zero exit
    with no id is a FAILED submission, and a non-zero exit that nevertheless
    printed an id is a real job that must be tracked (killing it later is a
    human decision; forgetting it is not an option)."""
    match = _JOB_ID_RE.search(stdout or "")
    return match.group(1) if match else None


# ------------------------------------------------------------------- state


def _state_path(campaign_dir: str) -> str:
    return os.path.join(campaign_dir, STATE_FILENAME)


def read_state(campaign_dir: str) -> dict:
    path = _state_path(campaign_dir)
    if not os.path.exists(path):
        return {"schemaVersion": CAMPAIGN_SCHEMA, "cells": {}}
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("cells", {})
    return state


def write_state(campaign_dir: str, state: dict) -> None:
    """Atomic rewrite: temp file in the same directory + ``os.replace``. A
    campaign top-up may be interrupted at any moment (cron, a killed login
    session); a half-written state file would either lose job ids or invent
    them."""
    path = _state_path(campaign_dir)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _record_attempt(campaign_dir: str, state: dict, cell_id: str,
                    attempt: dict) -> list:
    """Append one attempt to a cell's record and PERSIST IMMEDIATELY, before
    anything else can go out to the scheduler; returns the cell's attempts.

    This is the at-most-once seam. Batching these writes to the end of a
    submission cycle means a cycle killed mid-flight (cron kill, dropped
    login session, an OOM on the login node) erases job ids that exist on the
    cluster, and the next top-up resubmits every one of them — duplicate 27B
    training jobs on a billed allocation. The state entry is created HERE
    rather than when a cell is merely considered, so a cycle that submitted
    nothing leaves no record claiming otherwise.
    """
    entry = state.setdefault("cells", {}).setdefault(cell_id, {"attempts": []})
    attempts = entry.setdefault("attempts", [])
    attempts.append(attempt)
    if attempt.get("jobID"):
        entry["lastJobID"] = attempt["jobID"]
    write_state(campaign_dir, state)
    return attempts


def counted_attempts(attempts: list) -> int:
    """The attempts that count against the ``1 + maxResubmits`` budget: those
    that actually ENTERED the scheduler — sbatch returned a job id, or an
    existing submission was adopted by name.

    A submission that failed at sbatch time (recorded with ``jobID: None``)
    is kept in the state file as the audit trail of what was tried, but it is
    a NON-EVENT for budget purposes: it cannot have consumed cluster
    resources, so retrying it is always safe. Observed live 2026-08-12
    (a hard-item campaign): QOS submit-cap refusals burned two cells' entire
    budgets and the scheduler loop reported them "exhausted" although no
    Slurm job for them ever existed. This changes only the budget accounting
    for submission-time failures — the liveness/death classification of jobs
    that DID run (the unproven-death doctrine) is untouched."""
    return sum(1 for attempt in attempts if attempt.get("jobID"))


def read_campaign(campaign_dir: str) -> dict:
    path = os.path.join(campaign_dir, CAMPAIGN_FILENAME)
    if not os.path.exists(path):
        raise CampaignError(
            f"{campaign_dir} is not a materialized campaign "
            f"(no {CAMPAIGN_FILENAME})")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def cell_directory(campaign_dir: str, cell_id: str) -> str:
    return os.path.join(campaign_dir, CELLS_DIRNAME, cell_id)


def is_complete(campaign_dir: str, cell_id: str) -> bool:
    return os.path.exists(os.path.join(cell_directory(campaign_dir, cell_id),
                                       COMPLETION_MARKER))


# -------------------------------------------------------------- scheduler


def _queue_state(job_id: str, runner: Runner, squeue_command: str,
                 sacct_command: str | None = None) -> tuple[str | None, bool]:
    """``(mapped_state, query_ok)`` for one job, mirroring
    ``SlurmExecutor.poll_state_detailed``'s contract: ``(None, True)`` is
    POSITIVE absence (the scheduler answered and does not list the job),
    ``(None, False)`` means the query failed and says nothing at all.

    ``squeue`` first; on its failure, ``sacct``. This fallback is
    load-bearing (observed live 2026-08-11): squeue exits nonzero for a job
    it has already PURGED from the queue — "Invalid job id specified" — so a
    finished-and-forgotten job's death could never be proven by squeue
    alone. Under the unproven-death doctrine the two FAILED cells of the
    first OptVec campaign were treated as alive forever and the resubmit
    budget was unreachable. sacct is the accounting authority for exactly
    those jobs: a terminal answer there is positive knowledge, and an empty
    sacct answer after a squeue refusal is positive absence (a job neither
    in the queue nor in accounting is not running)."""
    result = runner([squeue_command, "-j", job_id, "-h", "-o", "%T"])
    if result.returncode == 0:
        text = (result.stdout or "").strip()
        if not text:
            return None, True
        return executors.map_slurm_state(text.splitlines()[0].strip()), True
    if sacct_command:
        acct = runner([sacct_command, "-j", job_id, "-X", "-n",
                       "-o", "State,ExitCode"])
        if acct.returncode == 0:
            text = (acct.stdout or "").strip()
            if not text:
                return None, True
            fields = text.splitlines()[0].split()
            exit_code = fields[1] if len(fields) > 1 else None
            return executors.map_slurm_state(fields[0], exit_code), True
    return None, False


def _find_by_name(job_name: str, runner: Runner,
                  squeue_command: str) -> tuple[str | None, str | None, bool]:
    """``(job_id, mapped_state, query_ok)`` for a submission looked up by its
    ``--job-name`` token — the recovery path for the residual crash window
    between one ``sbatch`` and the state write that records its id.

    Same three-valued contract as :func:`_queue_state`: a job listed, POSITIVE
    absence (the scheduler answered and lists nothing), or a failed query that
    proves nothing. ``squeue`` alone is enough here because the case being
    recovered is a job submitted seconds-to-minutes ago; a job already gone
    from the queue has either written its marker (completed) or died, and both
    are handled by the ordinary paths."""
    result = runner([squeue_command, "-h", "-o", "%i %T", "--name", job_name])
    if result.returncode != 0:
        return None, None, False
    lines = [line.strip() for line in (result.stdout or "").splitlines()
             if line.strip()]
    if not lines:
        return None, None, True
    fields = lines[0].split()
    job_id = fields[0]
    state = executors.map_slurm_state(fields[1]) if len(fields) > 1 else None
    return job_id, state, True


def _cell_liveness(campaign_dir: str, cell_id: str, entry: dict,
                   runner: Runner, squeue_command: str,
                   sacct_command: str | None = None) -> dict:
    """What the scheduler and the filesystem say about one cell.

    ``attempts`` is the BUDGET-COUNTED attempt count (submissions that
    entered the scheduler — see :func:`counted_attempts`);
    ``submitFailures`` is the number of recorded sbatch refusals, which
    consume no budget."""
    attempts = entry.get("attempts") or []
    last_job = None
    for attempt in reversed(attempts):
        if attempt.get("jobID"):
            last_job = attempt["jobID"]
            break
    counted = counted_attempts(attempts)
    out = {"attempts": counted, "jobID": last_job,
           "submitFailures": len(attempts) - counted,
           "completed": is_complete(campaign_dir, cell_id),
           "state": None, "queryOK": True}
    if last_job and not out["completed"]:
        state, ok = _queue_state(last_job, runner, squeue_command,
                                 sacct_command)
        out["state"] = state
        out["queryOK"] = ok
    return out


def _alive(liveness: dict) -> bool:
    """A cell occupies a queue slot when the scheduler says so — OR when the
    scheduler could not be asked. Unproven death never licenses a resubmit."""
    if liveness["completed"] or not liveness["jobID"]:
        return False
    if not liveness["queryOK"]:
        return True
    return liveness["state"] in ALIVE_STATES


# ------------------------------------------------------------------ submit


#: Held for the whole of a submit cycle. The per-attempt state persistence
#: and adoption-by-name close the CRASH windows, but two top-ups running
#: CONCURRENTLY could both pass the liveness scan inside one sbatch's flight
#: time and double-submit a 27B training job (review finding 2026-08-10).
SUBMIT_LOCK_FILENAME = "campaign-submit.lock"


@contextmanager
def _submission_lock(campaign_dir: str):
    """Non-blocking exclusive flock for the submit cycle; yields whether the
    lock is actually HELD. A lock already taken refuses (the other cycle
    will submit everything submittable — re-invocation costs nothing); a
    filesystem that cannot lock (some NFS/Lustre mounts) proceeds unheld so
    the guard never becomes a cluster-only blocker, and the caller says so
    in an advisory."""
    handle = open(os.path.join(campaign_dir, SUBMIT_LOCK_FILENAME), "a+")
    held = False
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise CampaignError(
                    "another submit cycle holds this campaign's submission "
                    "lock — concurrent top-ups can double-submit a cell; "
                    "let it finish and re-invoke (re-invocation costs "
                    "nothing)") from exc
            # ENOLCK/ENOSYS and kin: the filesystem cannot lock.
        yield held
    finally:
        if held:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def submit(campaign_dir: str, *, runner: Runner | None = None,
           sbatch_command: str = "sbatch") -> dict:
    """Top up the campaign's queue occupancy to ``maxQueued`` (serialized by
    :data:`SUBMIT_LOCK_FILENAME`) and return a machine-readable report.

    The campaign directory is absolutized FIRST (audit of the
    relative-path-through-the-renderer class, 2026-08-23). Every cell's
    rendered script carries this path twice — as its own ``cd`` target and
    inside ``optvec train --config <cell>/config.json`` — and the two resolve
    against DIFFERENT working directories once the ``cd`` has run, so a
    relative campaign directory typed on the command line renders a script
    that cannot find its own config. Same mechanism as the ``--source`` defect
    the same audit closed; louder failure, identical cause.
    """
    campaign_dir = os.path.abspath(campaign_dir)
    with _submission_lock(campaign_dir) as held:
        report = _submit_cycle(campaign_dir, runner=runner,
                               sbatch_command=sbatch_command)
        if not held:
            report["advisories"].append(
                "submission lock unavailable on this filesystem — concurrent "
                "top-ups are not guarded here; keep a single submitter per "
                "campaign")
        return report


def _submit_cycle(campaign_dir: str, *, runner: Runner | None = None,
                  sbatch_command: str = "sbatch") -> dict:
    """Top up the campaign's queue occupancy to ``maxQueued`` and return a
    machine-readable report. Re-invocation IS the mechanism: run it from a
    cron entry or a shell loop and it submits exactly the cells that can be
    submitted now.

    Never submits a cell that is completed (marker present) or alive
    (scheduler says so, or could not be asked). A dead-but-unmarked cell is
    resubmittable until its attempt budget (``1 + maxResubmits``) is spent,
    after which it is ``exhausted`` and waits for a human. Only attempts
    that entered the scheduler count against that budget
    (:func:`counted_attempts`): an sbatch refusal is recorded and reported
    loudly, but retrying it is always safe and never becomes ``exhausted``.

    At-most-once, in two layers. Every attempt is persisted to
    ``campaign-state.json`` the instant it happens, so a cycle killed
    mid-flight loses at most the one submission in flight. For that residual
    window, a cell with NO recorded job id is first looked up by its
    ``--job-name`` token: a job found there is ADOPTED (recorded, counted
    against occupancy, not resubmitted). The doctrine is deliberately
    asymmetric — for a cell with a KNOWN prior job a failed scheduler query
    means "treat as alive"; for adoption it means "skip this cell this
    cycle". Unproven absence licenses a submit no more than unproven death
    licenses a resubmit, and re-invocation costs nothing.
    """
    runner = runner or SubprocessRunner()
    campaign = read_campaign(campaign_dir)
    config = OptVecCampaignConfig.from_dict(campaign["config"])
    sacct_command, squeue_command = executors.scheduler_poll_commands()
    state = read_state(campaign_dir)
    cells_state = state.setdefault("cells", {})

    report: dict[str, Any] = {
        "campaign": config.name,
        "campaignDirectory": campaign_dir,
        "maxQueued": config.slurm.max_queued,
        "submitted": [], "adopted": [], "skipped": [], "failed": [],
        "advisories": [], "aliveBefore": 0, "capacity": 0,
    }

    window = executors.first_crossing_window(
        config.slurm.walltime or executors.SlurmResources.from_env().walltime,
        ServerProfile.from_env().maintenance_calendar_path)
    if window is not None:
        report["advisories"].append(
            "no submissions this cycle: a cell's walltime crosses the "
            f"maintenance window {window.get('start')}–{window.get('end')}")
        report["totals"] = _totals(campaign_dir, campaign, cells_state, runner,
                                   squeue_command,
                                   sacct_command=sacct_command)
        return report

    liveness = {
        entry["cellID"]: _cell_liveness(
            campaign_dir, entry["cellID"], cells_state.get(entry["cellID"], {}),
            runner, squeue_command, sacct_command)
        for entry in campaign["cells"]}
    for cell_id, live in liveness.items():
        if not live["queryOK"]:
            report["advisories"].append(
                f"{cell_id}: scheduler query failed for job {live['jobID']} — "
                "treated as ALIVE (unproven death never licenses a resubmit)")

    alive = [cid for cid, live in liveness.items() if _alive(live)]
    report["aliveBefore"] = len(alive)
    capacity = max(0, config.slurm.max_queued - len(alive))
    report["capacity"] = capacity

    budget = 1 + config.slurm.max_resubmits
    consecutive_failures = 0
    for entry in campaign["cells"]:
        if capacity <= 0:
            break
        cell_id = entry["cellID"]
        live = liveness[cell_id]
        if live["completed"] or _alive(live):
            continue
        if live["attempts"] >= budget:
            continue        # exhausted: reported by status(), never retried
        cell_dir = cell_directory(campaign_dir, cell_id)
        script = os.path.join(cell_dir, CELL_SCRIPT_FILENAME)
        if not os.path.exists(script):
            raise CampaignError(
                f"cell {cell_id} has no {CELL_SCRIPT_FILENAME} — materialize "
                "the campaign before submitting it")
        job_name = job_name_for(config, cell_id, campaign_dir)

        # The crash-window recovery, and the reason `job_name_for`'s promise
        # is true: a cell with no recorded job id may nevertheless HAVE a
        # submission — one whose state write never landed.
        if live["jobID"] is None:
            found_id, found_state, query_ok = _find_by_name(
                job_name, runner, squeue_command)
            if not query_ok:
                report["skipped"].append({"cellID": cell_id,
                                          "reason": "schedulerQueryFailed"})
                report["advisories"].append(
                    f"{cell_id}: scheduler query failed while checking for an "
                    "existing submission by name — skipping this cycle; "
                    "unproven absence never licenses a submit")
                continue
            if found_id is not None:
                attempts = _record_attempt(
                    campaign_dir, state, cell_id,
                    {"jobID": found_id, "outcome": "adopted",
                     "jobName": job_name})
                live = {**live, "jobID": found_id, "state": found_state,
                        "queryOK": True,
                        "attempts": counted_attempts(attempts)}
                liveness[cell_id] = live
                report["adopted"].append({"cellID": cell_id,
                                          "jobID": found_id,
                                          "schedulerState": found_state})
                report["advisories"].append(
                    f"{cell_id}: ADOPTED existing job {found_id} found under "
                    f"job name {job_name} — a previous cycle submitted it and "
                    "died before recording the id; not resubmitting")
                if _alive(live):
                    capacity -= 1
                continue

        result = runner([sbatch_command, f"--job-name={job_name}", script],
                        cwd=cell_dir)
        job_id = parse_job_id(result.stdout)
        if job_id is None:
            detail = ((result.stderr or "").strip()
                      or (result.stdout or "").strip()
                      or "sbatch produced no output")
            attempts = _record_attempt(
                campaign_dir, state, cell_id,
                {"jobID": None, "outcome": "failed",
                 "exitCode": result.returncode, "detail": detail})
            counted = counted_attempts(attempts)
            report["failed"].append({
                "cellID": cell_id, "exitCode": result.returncode,
                "detail": detail,
                "attempts": counted, "attemptBudget": budget,
                "submitFailures": len(attempts) - counted})
            report["advisories"].append(
                f"SUBMISSION FAILED for {cell_id}: sbatch exited "
                f"{result.returncode} and printed no job id — {detail}; "
                "not counted against the attempt budget (no job entered "
                "the scheduler)")
            liveness[cell_id] = {**live,
                                 "submitFailures": live["submitFailures"] + 1}
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_SUBMIT_FAILURES:
                report["advisories"].append(
                    f"stopped after {consecutive_failures} consecutive "
                    "submission failures — the usual cause (queue limit, QoS "
                    "refusal) applies to every further sbatch this cycle; "
                    "re-invoke once it clears")
                break
            continue
        consecutive_failures = 0
        attempts = _record_attempt(
            campaign_dir, state, cell_id,
            {"jobID": job_id, "outcome": "submitted",
             "exitCode": result.returncode, "jobName": job_name})
        counted = counted_attempts(attempts)
        report["submitted"].append({"cellID": cell_id, "jobID": job_id,
                                    "attempt": counted})
        capacity -= 1
        liveness[cell_id] = {**live, "jobID": job_id, "state": "submitted",
                             "queryOK": True, "attempts": counted}

    write_state(campaign_dir, state)
    report["totals"] = _totals(campaign_dir, campaign, cells_state, runner,
                               squeue_command, liveness=liveness,
                               sacct_command=sacct_command)
    return report


# ------------------------------------------------------------------ status


#: Cell status vocabulary. ``unknown`` is the honest answer when the scheduler
#: query itself failed: reporting "queued" would assert liveness we cannot
#: prove, and reporting "failed" would assert death we cannot prove.
STATUSES = ("planned", "queued", "running", "completed", "failed",
            "exhausted", "unknown")


def _cell_status(live: dict, budget: int) -> str:
    if live["completed"]:
        return "completed"
    if not live["jobID"]:
        # No job id ever came back. Untried is "planned"; tried-and-refused
        # (the exit-0-with-no-id fan-out failure, the QOS submit-cap crunch)
        # is a FAILURE that must not hide among the not-yet-submitted cells —
        # but it consumed no budget: a submission that never entered the
        # scheduler is a non-event for budget purposes, so a cell whose
        # failures were all submission-time is retryable, never "exhausted".
        return "failed" if live.get("submitFailures", 0) > 0 else "planned"
    if not live["queryOK"]:
        return "unknown"
    if live["state"] in _RUNNING_STATES:
        return "running"
    if live["state"] in _QUEUED_STATES:
        return "queued"
    # "exhausted" requires >= budget REAL attempts (jobs that entered the
    # scheduler); sbatch refusals in the record do not bring a cell closer.
    if live["attempts"] >= budget:
        return "exhausted"
    return "failed"


def status(campaign_dir: str, *, runner: Runner | None = None) -> dict:
    """Per-cell status plus totals — a machine-readable dict, no printing.

    Sources, in order of authority: the completion MARKER (a job wrote it and
    only on success), then ``squeue`` for anything with a job id, then the
    campaign state's attempt count for the failed/exhausted distinction.
    """
    runner = runner or SubprocessRunner()
    campaign = read_campaign(campaign_dir)
    config = OptVecCampaignConfig.from_dict(campaign["config"])
    sacct_command, squeue_command = executors.scheduler_poll_commands()
    cells_state = read_state(campaign_dir).get("cells", {})
    budget = 1 + config.slurm.max_resubmits

    rows = []
    advisories: list[str] = []
    totals = {name: 0 for name in STATUSES}
    for entry in campaign["cells"]:
        cell_id = entry["cellID"]
        live = _cell_liveness(campaign_dir, cell_id,
                              cells_state.get(cell_id, {}), runner,
                              squeue_command, sacct_command)
        cell_status = _cell_status(live, budget)
        totals[cell_status] += 1
        if not live["queryOK"]:
            advisories.append(
                f"{cell_id}: scheduler query failed for job {live['jobID']} — "
                "state unknown, not death")
        row = {
            "cellID": cell_id,
            "condition": entry.get("condition"),
            "layer": entry.get("layer"),
            "seed": entry.get("seed"),
            "configHash": entry.get("configHash"),
            "status": cell_status,
            "jobID": live["jobID"],
            "schedulerState": live["state"],
            "attempts": live["attempts"],
            "attemptBudget": budget,
            "submitFailures": live["submitFailures"],
            "schedulerQueryFailed": not live["queryOK"],
        }
        if entry.get("item") is not None:
            # Present only on a per-item campaign, so a classic campaign's
            # status table is exactly the table it was.
            row["item"] = entry["item"]
            row["itemSlug"] = entry.get("itemSlug")
        rows.append(row)
    return {"campaign": config.name, "campaignDirectory": campaign_dir,
            "cells": rows, "totals": totals,
            "cellCount": len(rows),
            "alive": totals["queued"] + totals["running"] + totals["unknown"],
            "maxQueued": config.slurm.max_queued,
            "advisories": advisories}


def _totals(campaign_dir: str, campaign: dict, cells_state: dict,
            runner: Runner, squeue_command: str,
            liveness: dict | None = None,
            sacct_command: str | None = None) -> dict:
    """Status totals for a submit report, reusing the liveness already polled
    this cycle so a top-up never double-queries the scheduler."""
    config = OptVecCampaignConfig.from_dict(campaign["config"])
    budget = 1 + config.slurm.max_resubmits
    totals = {name: 0 for name in STATUSES}
    for entry in campaign["cells"]:
        cell_id = entry["cellID"]
        live = (liveness or {}).get(cell_id) or _cell_liveness(
            campaign_dir, cell_id, cells_state.get(cell_id, {}), runner,
            squeue_command, sacct_command)
        totals[_cell_status(live, budget)] += 1
    return totals


# ------------------------------------------------------------------ helpers


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower())
    return text.strip("-")


def item_slug_map(items) -> dict[str, str]:
    """Raw item id → the token that appears in cell ids, directory names and
    scheduler job names.

    THE SLUG RULE (stated here because a cell id is a durable identifier and a
    reader must be able to run it backwards):

    1. ``base = slugify(id)`` — lowercased, every run of characters outside
       ``[a-z0-9]`` collapsed to ``-``, leading/trailing ``-`` stripped.
    2. The base is used AS IS only when it is **lossless and unique**: it
       equals ``id.strip().lower()``, it is at most :data:`ITEM_SLUG_MAX`
       characters, and no other item in this campaign produced the same base.
       Ordinary ids (``t-14``, ``c-004``, ``T-14``) take this path, so cell
       ids stay readable.
    3. Otherwise — the id contained characters a filename cannot carry, or was
       longer than the cap, or two ids slugged alike — the token becomes
       ``<base truncated to ITEM_SLUG_MAX>-<sha256(id)[:8]>``, and
       ``item-<sha256(id)[:8]>`` when the base is empty. The hash is over the
       RAW id, so two ids that truncate or sanitize to the same base still get
       different tokens.

    Uniqueness is then verified over the finished set rather than assumed: the
    disambiguated form of one id could in principle equal the clean base of
    another, and a cell id that names two items would corrupt the campaign
    silently. A residual collision refuses.
    """
    items = list(items)
    bases: dict[str, str] = {}
    base_counts: dict[str, int] = {}
    for item in items:
        base = _slugify(item)
        bases[item] = base
        base_counts[base] = base_counts.get(base, 0) + 1

    slugs: dict[str, str] = {}
    for item in items:
        base = bases[item]
        lossless = (base == item.strip().lower() and 0 < len(base)
                    <= ITEM_SLUG_MAX and base_counts[base] == 1)
        if lossless:
            slugs[item] = base
            continue
        digest = hashlib.sha256(item.encode("utf-8")).hexdigest()[
            :ITEM_SLUG_HASH_CHARS]
        truncated = base[:ITEM_SLUG_MAX].strip("-")
        slugs[item] = f"{truncated}-{digest}" if truncated else f"item-{digest}"

    seen: dict[str, str] = {}
    for item, slug in slugs.items():
        if slug in seen:
            raise CampaignConfigError(
                f"item ids {seen[slug]!r} and {item!r} both slug to {slug!r} — "
                "cell ids would name two different items; rename one")
        seen[slug] = item
    return slugs


def _read_item_axis(grid: dict, item_axis: Any) -> tuple[tuple[str, ...],
                                                         dict | None]:
    """``(items, itemsFile provenance)`` from the grid's optional item axis.

    Two authored forms, never both: ``grid.items`` (the ids inline) or
    ``grid.itemsFile`` (a choice-row JSONL whose ids define the axis, read
    through the strict cross-engine loader — the same one the sweep's
    ``logprobShift`` objective uses, so a file that loads for a campaign loads
    for a measurement). Two declarations of one axis cannot both be the record
    of what ran.

    ``itemAxis`` is the CANONICAL form's provenance block, written by
    :meth:`OptVecCampaignConfig.to_dict` and read back when a materialized
    campaign is re-opened. It never declares the axis and never re-reads the
    items file; its slug map is checked against the recomputed one so a
    hand-edited campaign.json cannot make cell ids lie.
    """
    if "items" in grid and "itemsFile" in grid:
        raise CampaignConfigError(
            "declare grid.items OR grid.itemsFile, never both — two "
            "declarations of the item axis cannot both be the record of what "
            "ran")

    items: tuple[str, ...] = ()
    items_file: dict | None = None
    if "items" in grid:
        raw = grid["items"]
        if not isinstance(raw, list) or not raw:
            raise CampaignConfigError(
                "grid.items must be a non-empty list of item-id strings")
        if any(not isinstance(v, str) or not v.strip() for v in raw):
            raise CampaignConfigError(
                "grid.items must be non-empty item-id strings")
        if len(set(raw)) != len(raw):
            raise CampaignConfigError(
                "grid.items repeats an item — a duplicated axis value would "
                "produce two cells with the same id")
        items = tuple(raw)
    elif "itemsFile" in grid:
        items, items_file = _load_items_file(grid["itemsFile"])

    if item_axis is not None:
        if not isinstance(item_axis, dict):
            raise CampaignConfigError("'itemAxis' must be a JSON object")
        unknown = sorted(set(item_axis) - {"slugs", "file"})
        if unknown:
            raise CampaignConfigError(
                "unknown itemAxis key(s): " + ", ".join(unknown))
        if not items:
            raise CampaignConfigError(
                "'itemAxis' is the item axis's provenance, not its "
                "declaration — declare grid.items or grid.itemsFile")
        declared_slugs = item_axis.get("slugs")
        if declared_slugs is not None:
            if declared_slugs != item_slug_map(items):
                raise CampaignConfigError(
                    "itemAxis.slugs disagrees with the slug rule applied to "
                    "grid.items — a hand-edited slug map would make every "
                    "cell id lie about which item it trained")
        declared_file = item_axis.get("file")
        if declared_file is not None:
            if not isinstance(declared_file, dict) or not isinstance(
                    declared_file.get("path"), str) or not isinstance(
                    declared_file.get("sha256"), str):
                raise CampaignConfigError(
                    "itemAxis.file must be {'path': …, 'sha256': …} or null")
            items_file = {"path": declared_file["path"],
                          "sha256": declared_file["sha256"]}

    if items:
        item_slug_map(items)         # surface a slug collision at config time
    return items, items_file


def _load_items_file(declared: Any) -> tuple[tuple[str, ...], dict]:
    """The item ids in a choice-row JSONL, plus the SHA-256 of its bytes.

    ``declared`` is either the path, or ``{"path": …, "sha256": …}`` — the
    object form PINS the file: a declared hash that does not match the bytes
    read refuses, because a campaign whose item axis changed underneath it is
    a different campaign.
    """
    pinned: str | None = None
    if isinstance(declared, dict):
        unknown = sorted(set(declared) - {"path", "sha256"})
        if unknown:
            raise CampaignConfigError(
                "unknown grid.itemsFile key(s): " + ", ".join(unknown))
        path = declared.get("path")
        pinned = declared.get("sha256")
        if pinned is not None and (not isinstance(pinned, str)
                                   or not pinned.strip()):
            raise CampaignConfigError(
                "grid.itemsFile.sha256 must be the SHA-256 of the file's raw "
                "bytes")
    else:
        path = declared
    if not isinstance(path, str) or not path.strip():
        raise CampaignConfigError(
            "grid.itemsFile must be a path to a choice-row JSONL (or "
            "{'path': …, 'sha256': …})")

    from .sweep_selection import load_choice_rows  # stdlib-only, no torch

    try:
        rows, digest = load_choice_rows(path, path)
    except ValueError as exc:
        raise CampaignConfigError(
            f"grid.itemsFile {path!r} is not a loadable choice-row file: "
            f"{exc}") from exc
    if not rows:
        raise CampaignConfigError(
            f"grid.itemsFile {path!r} declares no items — an empty item axis "
            "would silently collapse the grid")
    if pinned is not None and pinned.strip().lower() != digest:
        raise CampaignConfigError(
            f"grid.itemsFile {path!r} hashes to {digest}, not the declared "
            f"{pinned.strip().lower()} — the item axis drifted; a campaign "
            "over different items is a different campaign")
    ids = [row.id for row in rows]
    if len(set(ids)) != len(ids):            # the loader already refuses this
        raise CampaignConfigError(
            f"grid.itemsFile {path!r} repeats an item id")
    return tuple(ids), {"path": path, "sha256": digest}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """``overrides`` over ``base``: nested objects merge key-by-key, anything
    else REPLACES. A list is a value, not a set to be extended — a condition
    that names ``priorVectorPaths`` means exactly those paths."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")).hexdigest()
