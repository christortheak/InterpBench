"""Resident model registry for the server API.

The native Swift app can load models on demand inside one process. The server
needs one extra layer: a model requested by an agent, judge, or endpoint may be
different from the current UI-selected model. This registry keeps independently
loaded model slots on available devices when possible, and falls back to LRU
eviction/reload when the serving budget is one model.

Locking contract (live 2026-07-17): a model load takes MINUTES, and the
registry lock must never be held across one. Holding it starved every reader —
/api/capabilities (the GPU-session controller's health probe) and /api/state
blocked behind the load, so the controller read a hard-working worker as
unreachable and demoted the session to "Starting" for the whole load. Loads now
reserve a placeholder slot under the registry lock, run ``model_loader.load``
outside it (serialized against OTHER loads by a dedicated load lock — one big
weight copy at a time), and publish the model under the lock afterward.
Readers (``snapshots``, ``any_busy``, ``residency``) answer instantly
throughout; concurrent requests for the same model wait on the slot's ready
event and share the one load.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

from ..steering import model_loader


def _identity(item) -> tuple[str, str | None, str | None]:
    """Coerce one release request into a ``(modelID, revision, canonical
    dtype)`` triple.

    A bare slug is REFUSED rather than silently widened (external review
    round 12, finding 3): "release org/model" cannot say WHICH of two
    pinned revisions it means, and guessing either way is a wrong answer —
    drop the container the next column needs, or leave the finished one
    resident. Callers name the identity.
    """
    if isinstance(item, str):
        raise TypeError(
            "release_models takes (modelID, revision, dtype) identities, not "
            f"the bare slug {item!r} — a slug cannot tell two pinned "
            "revisions of one model apart, and the release has to")
    model_id, revision, dtype = (tuple(item) + (None, None))[:3]
    return (str(model_id), revision or None,
            model_loader.normalize_dtype(dtype))


@dataclass
class ModelSlot:
    key: tuple[str, str | None, str, str]
    # None while the load is in flight (a reserved "loading" placeholder);
    # published under the registry lock once model_loader.load returns.
    model: model_loader.SteeredModel | None
    device: str
    dtype: str
    loaded_at: float
    last_used: float
    lock: threading.Lock
    # Set once the slot holds a usable model OR its load failed (in which case
    # the slot has been removed from the registry) — waiters re-check and retry.
    ready: threading.Event = field(default_factory=threading.Event)

    @property
    def loading(self) -> bool:
        return self.model is None

    def snapshot(self) -> dict:
        if self.model is None:
            return {
                "modelID": self.key[0],
                "revision": self.key[1],
                "device": self.device,
                "dtype": self.dtype,
                "numLayers": None,
                "hiddenSize": None,
                "contextWindow": None,
                # A loading slot IS busy: generations against it will queue.
                "busy": True,
                "loading": True,
                "loadedAt": None,
                "lastUsed": self.last_used,
            }
        return {
            "modelID": self.model.model_id,
            "revision": self.model.revision,
            "device": self.device,
            # Prefer the dtype the model ACTUALLY runs in (stamped at load);
            # fall back to the requested string (often "auto") for wrappers
            # that predate the stamp.
            "dtype": getattr(self.model, "dtype", None) or self.dtype,
            "numLayers": self.model.num_layers,
            "hiddenSize": self.model.hidden_size,
            "contextWindow": self.model.context_window,
            "busy": self.lock.locked(),
            "loading": False,
            "loadedAt": self.loaded_at,
            "lastUsed": self.last_used,
        }


class ModelRegistry:
    """Load/reuse models across devices with conservative eviction.

    Defaults:
    - CUDA server: one resident model per CUDA device.
    - MPS/CPU server: one resident model total.

    Override with ``STEERLAB_MAX_LOADED_MODELS`` when you deliberately want more
    or fewer resident slots.
    """

    def __init__(self):
        self._slots: dict[tuple[str, str | None, str, str], ModelSlot] = {}
        self._lock = threading.RLock()
        # Serializes the loads themselves (one multi-GB weight copy at a
        # time), NEVER readers — held only outside self._lock.
        self._load_lock = threading.Lock()
        self._devices = model_loader.available_devices()
        cuda_devices = [d for d in self._devices if d.startswith("cuda")]
        default_max = len(cuda_devices) if cuda_devices else 1
        self.max_loaded = max(1, int(os.environ.get("STEERLAB_MAX_LOADED_MODELS", default_max)))
        # Eviction listener (external review round 12, finding 2b). The
        # registry can drop its OWN reference to a container, but it cannot
        # reclaim weights a caller outside it still holds — ``ServiceState``
        # keeps the ``/api/load``-ed model in ``state.model``, and an
        # eviction that only nulled ``slot.model`` left the real owner
        # untouched: the trim ran, freed nothing, and the "released" GiB
        # never came back. Set by the owner; called with
        # ``(key, container)`` after the slot is unregistered and BEFORE the
        # allocator trim, so every strong reference is gone by the time the
        # device is asked to give the memory back. It must not raise.
        self.on_evict: Callable[[tuple, object], None] | None = None

    @property
    def devices(self) -> list[str]:
        return list(self._devices)

    def snapshots(self) -> list[dict]:
        with self._lock:
            return [slot.snapshot() for slot in self._slots.values()]

    def any_busy(self) -> bool:
        with self._lock:
            return any(slot.lock.locked() or slot.loading
                       for slot in self._slots.values())

    def residency(self, model_id: str, revision: str | None = None, *,
                  dtype: str = "auto", device: str | None = None) -> str | None:
        """``"ready"`` / ``"loading"`` / None for the slot this request would
        reuse. Never blocks, never loads — the honesty seam the streaming chat
        routes use to warn the client BEFORE a silent in-stream cold load."""
        with self._lock:
            slot = self._matching_slot(model_id, revision, dtype, device)
        if slot is None:
            return None
        return "loading" if slot.loading else "ready"

    def unload(self, model_id: str, revision: str | None = None) -> int:
        with self._lock:
            keys = [
                key for key, slot in self._slots.items()
                # A loading placeholder is never evicted mid-copy: the load
                # thread would publish into a dangling slot.
                if not slot.loading
                and slot.model.model_id == model_id
                and (revision is None or slot.model.revision == revision)
                and not slot.lock.locked()
            ]
            for key in keys:
                self._evict(key)
            return len(keys)

    def release_models(self, identities) -> list[dict]:
        """Drop the resident container of every named IDENTITY, reporting what
        went (the explicit release the judged run seams call).

        ``identities`` are ``(modelID, revision | None, canonical dtype |
        None)`` triples — never bare slugs (external review round 12,
        finding 3). A judge panel can pin one slug at two revisions, and a
        release that spoke slugs would either drop the container the next
        column is about to use or, matching by slug alone, leave the
        finished OLD-revision container resident as dead weight — the
        co-residency OOM the seam exists to prevent. The triple mirrors the
        container key ``_matching_slot`` reuses by: revision None means "the
        revisionless slot", not "any revision", and dtype is compared
        canonically so ``auto``/unset and a resolved pin agree. ``device``
        is not part of the identity — it is the registry's choice, not
        something a judge pins — and rides only in the returned record.

        The maintainer's ruling, 2026-08-28: "any runs that require two
        models will need to unload and load models in order not to OOM. We
        need to ensure this happens." A run that needs its models
        SEQUENTIALLY must not pay the SUM of their weights: the caller
        computes which models the remainder of the run still needs and
        names the rest here, BEFORE the next load.

        This is not a new eviction heuristic and it does not touch the cache
        POLICY — ``_make_room`` still declines to evict while slots are
        free, which is what interactive serving wants. It is an explicit
        call at a run seam, reusing ``_evict`` (the one eviction
        implementation: drop the model reference, collect, trim the
        allocator) so there is never a second way to free a container.

        Busy slots are skipped exactly as ``unload``/``_make_room`` skip
        them — a slot is locked for the duration of any in-flight
        generation or load, and the pipeline's chain-held study model is
        locked for the whole chain, so a still-working model is never
        released out from under its worker. Each returned record carries
        ``modelID``/``revision``/``dtype``/``device`` and the estimated
        ``bytes`` freed (the cached snapshot size, None when unknown) so the
        caller can write the memory story into the run log.
        """
        wanted = [_identity(item) for item in (identities or ()) if item]
        if not wanted:
            return []
        released: list[dict] = []
        with self._lock:
            victims = [
                (key, slot) for key, slot in self._slots.items()
                if not slot.loading and not slot.lock.locked()
                and any(self._slot_is(slot, ident) for ident in wanted)]
            for key, slot in victims:
                released.append({
                    "modelID": slot.model.model_id,
                    "revision": slot.model.revision,
                    "dtype": model_loader.normalize_dtype(slot.dtype),
                    "device": slot.device,
                    "bytes": model_loader.snapshot_size_bytes(
                        slot.model.model_id, slot.model.revision),
                })
                self._evict(key)
        return released

    @staticmethod
    def _slot_is(slot: ModelSlot, identity: tuple) -> bool:
        """Does this READY slot hold exactly the named identity?

        The same three comparisons ``_matching_slot`` reuses a slot by, so a
        release can never name a container a load would not have matched:
        the slug; the revision (a pinned one against the revision the load
        RESOLVED to, an unpinned one only against a slot that was itself
        requested revisionless); and the dtype, canonically — ``auto`` and
        an unset pin are the same "let the device decide", and bf16 and fp16
        are different containers.
        """
        model_id, revision, dtype = identity
        if slot.model.model_id != model_id:
            return False
        if revision is not None:
            if slot.model.revision != revision and slot.key[1] != revision:
                return False
        elif slot.key[1] is not None:
            return False
        return model_loader.normalize_dtype(slot.dtype) == dtype

    def unload_all(self) -> int:
        with self._lock:
            keys = [key for key, slot in self._slots.items()
                    if not slot.loading and not slot.lock.locked()]
            for key in keys:
                self._evict(key)
            return len(keys)

    def _matching_slot(self, model_id: str, revision: str | None,
                       dtype: str, device: str | None) -> ModelSlot | None:
        """Reuse scan (registry lock held by the caller). Ready slots match a
        revisionless request only when they were themselves loaded
        revisionless; loading placeholders match on their requested key, so a
        second request for an in-flight model waits instead of double-loading."""
        for slot in self._slots.values():
            if slot.loading:
                revision_matches = slot.key[1] == revision
                slot_id = slot.key[0]
            else:
                slot_id = slot.model.model_id
                revision_matches = (
                    slot.model.revision == revision if revision is not None
                    else slot.key[1] is None)
            if (slot_id == model_id and revision_matches
                    and slot.dtype == (dtype or "auto")
                    and (not device or device == "auto" or slot.device == device)):
                return slot
        return None

    def get_or_load(self, model_id: str, revision: str | None = None, *,
                    dtype: str = "auto", device: str | None = None) -> ModelSlot:
        while True:
            created: ModelSlot | None = None
            with self._lock:
                slot = self._matching_slot(model_id, revision, dtype, device)
                if slot is None:
                    target_device = self._choose_device(device)
                    key = (model_id, revision, dtype or "auto", target_device)
                    slot = self._slots.get(key)
                    if slot is None:
                        self._make_room(target_device)
                        now = time.time()
                        created = ModelSlot(
                            key=key, model=None, device=target_device,
                            dtype=dtype or "auto", loaded_at=now,
                            last_used=now, lock=threading.Lock())
                        self._slots[key] = created

            if created is None:
                # Wait for a concurrent load OUTSIDE the registry lock (a
                # ready slot's event is already set — this returns at once).
                slot.ready.wait()
                if slot.model is None:
                    # That load failed and the placeholder was removed; retry
                    # from the top (this caller may become the new loader).
                    continue
                with self._lock:
                    slot.last_used = time.time()
                return slot

            try:
                with self._load_lock:
                    model = model_loader.load(
                        model_id, revision=revision, dtype=dtype,
                        device=created.device)
            except BaseException:
                with self._lock:
                    self._slots.pop(created.key, None)
                created.ready.set()  # waiters re-check, see the removal, retry
                # A failed CUDA load can strand partially-moved weights on
                # the device (live 2026-07-18: a 12B OOM left ~21 GiB of
                # debris that OOM'd the NEXT model's first generation) —
                # sweep before surfacing the failure.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            with self._lock:
                created.model = model
                now = time.time()
                created.loaded_at = now
                created.last_used = now
            created.ready.set()
            return created

    @contextmanager
    def acquire(self, model_id: str, revision: str | None = None, *,
                dtype: str = "auto", device: str | None = None) -> Iterator[model_loader.SteeredModel]:
        while True:
            slot = self.get_or_load(model_id, revision, dtype=dtype, device=device)
            slot.lock.acquire()
            # Eviction race (engineer review 2026-07-17): unload/_make_room
            # skip only LOCKED slots, so between get_or_load returning and
            # our lock landing the slot can be evicted (model dropped). Verify
            # it is still the registered slot before using it; reload if not.
            with self._lock:
                current = (self._slots.get(slot.key) is slot
                           and slot.model is not None)
            if current:
                break
            slot.lock.release()
        try:
            slot.last_used = time.time()
            yield slot.model
        finally:
            slot.last_used = time.time()
            slot.lock.release()

    def _choose_device(self, requested: str | None) -> str:
        if requested and requested != "auto":
            return requested
        cuda_devices = [d for d in self._devices if d.startswith("cuda")]
        if cuda_devices:
            loaded_by_device = {d: 0 for d in cuda_devices}
            for slot in self._slots.values():
                if slot.device in loaded_by_device:
                    loaded_by_device[slot.device] += 1
            return min(cuda_devices, key=lambda d: loaded_by_device[d])
        return self._devices[0] if self._devices else "cpu"

    def _make_room(self, target_device: str) -> None:
        if len(self._slots) < self.max_loaded:
            return
        candidates = [slot for slot in self._slots.values()
                      if not slot.lock.locked() and not slot.loading]
        if not candidates:
            busy = ", ".join(sorted(
                s.key[0] if s.loading else s.model.model_id
                for s in self._slots.values()))
            raise model_loader.ModelLoadError(
                f"cannot load another model: every resident model slot is busy "
                f"({busy}). A slot is held for the duration of any in-flight "
                "generation, running job, or model load (chat, extraction, "
                "sweep, study, judging); retry when it finishes or cancel the "
                f"work holding it. This server keeps up to {self.max_loaded} "
                "resident model(s) — raise STEERLAB_MAX_LOADED_MODELS for "
                "more capacity.")
        # Prefer evicting a model from the target device; otherwise evict global LRU.
        same_device = [slot for slot in candidates if slot.device == target_device]
        victim = min(same_device or candidates, key=lambda s: s.last_used)
        self._evict(victim.key)

    def _evict(self, key: tuple[str, str | None, str, str]) -> None:
        slot = self._slots.pop(key, None)
        if slot is None:
            return
        container = slot.model
        # Drop the reference by ASSIGNMENT, not `del`: stale slot handles held
        # by a racing acquire() must read a clean None (→ retry), never raise
        # AttributeError on a deleted dataclass field.
        slot.model = None
        # Every OTHER owner drops it too, before the trim — a service that
        # still holds this container keeps its weights alive and the trim
        # below would reclaim nothing.
        if container is not None and self.on_evict is not None:
            self.on_evict(key, container)
        # The one reclamation implementation, shared with the CLI/bundle
        # release seam (which has no registry to evict from).
        model_loader.free_device_memory(slot.device)
