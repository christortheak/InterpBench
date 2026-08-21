"""Env-gated per-record host-memory diagnostic (open-issues §15, hunt 2).

The §15 leak is CLUSTER-ONLY: ~230 KB of host RAM per decode step on
vector-injection study jobs, ~10.3 GB/h, invisible to two Mac harnesses that
drove the identical per-step code (hunt 1). Hunt 2 mapped the cluster path and
found that a study job is a plain CLI subprocess (``bundle execute``) whose
per-decode-step code is byte-identical to what hunt 1 measured flat — so the
retention is either below Python, or per-record with a token-proportional
magnitude, or specific to the armed-hook branch under glibc's allocator. No
local reproduction can separate those; a cluster measurement can.

This module is that measurement. It appends ONE JSON line per emitted record to
``<runDir>/memory-diagnostic.jsonl``, carrying the three quantities the field
evidence never had:

* **host RSS vs torch-visible device memory** — whether the growth is
  torch-accounted at all, which no OOM report so far has established;
* **cumulative decode steps** beside cumulative records — which separates
  token-linear from record-linear retention directly in the file, instead of
  inferring it from two jobs' records-per-hour;
* **armed hook fires** beside decode steps — which separates "per decode step"
  from "per armed hook fire"; and beside them the corrected allocation
  counters (``interventionDispatches`` / ``effectiveInjections`` /
  ``replacementTensors`` / ``outputTuplesRebuilt`` / ``injectionCloneBytes``),
  which say how much the armed arm actually allocates per step rather than how
  many times it was dispatched. Hunt 2's hypothesis was stated as "the armed
  branch returns a FRESH tensor for every layer of every pass"; that was wrong
  — fresh tensors appeared only at layers an intervention modified, though a
  fresh output TUPLE was built at every armed layer (fixed 2026-08-18, along
  with the injector's addition temporary and the per-cell dispatch fan-out).
  The counters are what let a cluster run see which of those cadences the RSS
  slope tracks, on the post-fix code.

**Baselines are per CONDITION.** A run walks its arms in sequence, so a single
run-global baseline attributes every earlier arm's growth to the current one —
fatal for the one comparison the §15 matrix exists to make. The unprefixed
ratios (``hostBytesPerDecodeStep`` …) are measured from the current condition's
first record; the cumulative run-global pair lives under ``run*`` names; and
``intervalHostBytesPerDecodeStep`` is the marginal record-to-record slope, which
needs no baseline at all.

**Zero overhead when unset.** ``STEERLAB_MEMORY_DIAGNOSTIC`` unset means
:func:`observe` returns before constructing anything, and
``steering.hooks.HookedModel`` installs its historical hook closure — the
per-layer-per-token hot path does not gain so much as a branch.

Tiers:

* ``STEERLAB_MEMORY_DIAGNOSTIC=1`` — the cheap tier: RSS, CUDA allocator
  counters, device identity, hook counters, structure lengths. Microseconds per
  record.
* ``STEERLAB_MEMORY_DIAGNOSTIC=2`` — adds a ``gc`` object census and a live
  ``torch.Tensor`` count/bytes (hunt 1's "is anything Python-reachable?"
  measurement). Seconds per record on a large process; use it on a SHORT job.

Nothing here may ever break a run: every collection step is individually
guarded and :func:`observe` swallows its own failures. A diagnostic that can
kill a 7-hour cluster job is worse than no diagnostic.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings

#: Set to 1 (cheap) or 2 (adds the gc/tensor census) to arm the diagnostic.
ENV_VAR = "STEERLAB_MEMORY_DIAGNOSTIC"

#: Written inside the run directory, beside generations.jsonl.
FILENAME = "memory-diagnostic.jsonl"

#: 2 (2026-08-18): per-CONDITION re-baselining — the ratios named without a
#: prefix are now measured from the current condition's first record, and the
#: run-global ones moved to ``run*`` names. Version 1 files carry only the
#: run-global reading under the unprefixed names; for a single-condition file
#: the two are identical. Also added: the corrected hook counters
#: (``interventionDispatches`` / ``effectiveInjections`` / ``replacementTensors``
#: / ``outputTuplesRebuilt`` / ``injectionCloneBytes``) and the per-record
#: interval fields.
SCHEMA_VERSION = 2

_DISABLED_VALUES = {"", "0", "false", "no", "off"}

#: One collector per run directory, created lazily on the first observation.
_ACTIVE: dict[str, "MemoryDiagnostic"] = {}

_warned = False


def level() -> int:
    """0 = off, 1 = cheap tier, 2 = deep tier. Unrecognized truthy text reads
    as the cheap tier, so ``=true`` behaves like ``=1``."""
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    if raw in _DISABLED_VALUES:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def enabled() -> bool:
    return level() > 0


def new_counters() -> dict[str, int]:
    """The counter block ``steering.hooks.HookedModel`` fills when armed.

    Defined here rather than in the hook manager so the diagnostic owns its own
    schema: the reader of memory-diagnostic.jsonl and the writer of the counts
    agree by construction.

    The allocation-side counters, and what each one settles:

    ``interventionApplications``
        ``armed objects × armed hook fires`` — dispatches ATTEMPTED under the
        pre-consolidation accounting. Historical name, historical meaning, so
        files from either side of 2026-08-18 compare.
    ``interventionDispatches``
        ``apply`` calls actually made. With one injector per cell this equals
        ``interventionApplications``; with ``steering.plan``'s distinct-layer
        consolidation an 11-cell band on 62 layers reads 62 per decode step
        where it used to read 682. Reading 682 in a fresh run means the
        condition shares layers between cells (legitimate) or the
        consolidation did not engage (a regression).
    ``effectiveInjections``
        ``apply`` calls that returned a NEW tensor — i.e. that actually
        modified the residual stream. ~11 per decode step for an 11-cell band,
        whatever the dispatch count. This is the true clone cadence.
    ``replacementTensors``
        Hook fires whose outgoing hidden state was not the one the block
        produced. ≤ ``effectiveInjections``: a layer carrying two chained edits
        clones twice but hands back one tensor.
    ``outputTuplesRebuilt``
        Hook fires that had to rebuild the block's output tuple. Before
        2026-08-18 this was every armed fire (62/step at 27B); now it tracks
        ``replacementTensors``.
    ``injectionCloneBytes``
        Cumulative bytes of every tensor an ``apply`` returned fresh —
        ``numel × element_size``, so prefill's ``[1, P, hidden]`` clone and a
        decode step's ``[1, 1, hidden]`` are both counted at their real size.
        The device-side churn figure to hold beside host RSS growth.
    """
    return {
        "hookFires": 0,
        "armedHookFires": 0,
        "interventionApplications": 0,
        "interventionDispatches": 0,
        "effectiveInjections": 0,
        "replacementTensors": 0,
        "outputTuplesRebuilt": 0,
        "injectionCloneBytes": 0,
        "forwardPasses": 0,
        "decodeSteps": 0,
        "prefillTokens": 0,
    }


def reset() -> None:
    """Drop the per-run collectors (tests; and any in-process run boundary)."""
    _ACTIVE.clear()


# --- collection helpers -------------------------------------------------------


def _process_memory() -> dict:
    """Host RSS/VMS, by whichever mechanism this platform offers.

    psutil is in both platform locks, so it is the normal answer; procfs is the
    zero-dependency Linux fallback and matters because the cluster is the only
    place this diagnostic is meant to run.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {"rssBytes": int(info.rss), "vmsBytes": int(info.vms),
                "memorySource": "psutil"}
    except Exception:  # noqa: BLE001 - advisory only
        pass
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        page = os.sysconf("SC_PAGE_SIZE")
        return {"rssBytes": int(fields[1]) * page,
                "vmsBytes": int(fields[0]) * page,
                "memorySource": "procfs"}
    except Exception:  # noqa: BLE001 - advisory only
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on Darwin, kilobytes on Linux — and it is a PEAK,
        # not a current reading, so it is the last resort only.
        scale = 1 if sys.platform == "darwin" else 1024
        return {"rssBytes": int(peak) * scale, "vmsBytes": None,
                "memorySource": "rusage-peak"}
    except Exception:  # noqa: BLE001 - advisory only
        return {"rssBytes": None, "vmsBytes": None, "memorySource": None}


def _accelerator_memory() -> dict:
    """Torch-visible device memory + device identity.

    The RSS-vs-``cudaReserved`` split is the whole point: host growth while the
    CUDA allocator stays flat proves the retention is host-side and
    torch-invisible, which is what every §15 OOM report so far has been unable
    to say. ``deviceName`` settles the A100/H100 attribution for the record.
    """
    out = {"cudaAllocatedBytes": None, "cudaReservedBytes": None,
           "cudaMaxReservedBytes": None, "mpsAllocatedBytes": None,
           "deviceName": None, "deviceCount": 0}
    try:
        import torch
    except Exception:  # noqa: BLE001 - advisory only
        return out
    try:
        if torch.cuda.is_available():
            out["cudaAllocatedBytes"] = int(torch.cuda.memory_allocated())
            out["cudaReservedBytes"] = int(torch.cuda.memory_reserved())
            out["cudaMaxReservedBytes"] = int(torch.cuda.max_memory_reserved())
            out["deviceCount"] = int(torch.cuda.device_count())
            out["deviceName"] = torch.cuda.get_device_name(
                torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - advisory only
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            out["mpsAllocatedBytes"] = int(torch.mps.current_allocated_memory())
            out["deviceName"] = out["deviceName"] or "mps"
    except Exception:  # noqa: BLE001 - advisory only
        pass
    return out


def _hook_facts(model) -> dict:
    """Counters and structure lengths from the hook manager.

    ``armedInterventions`` is the stale-injector probe (hunt 1's by-catch: a
    suspended ``_stream_rendered`` generator leaves a dead record's injectors
    restored and permanently armed). Between records it must read 0; anything
    else is that hazard firing live.
    """
    out = {"armedInterventions": None, "forwardHookCount": None,
           "hookFires": None, "armedHookFires": None,
           "interventionApplications": None, "interventionDispatches": None,
           "effectiveInjections": None, "replacementTensors": None,
           "outputTuplesRebuilt": None, "injectionCloneBytes": None,
           "cumulativeForwardPasses": None,
           "cumulativeDecodeSteps": None, "cumulativePrefillTokens": None}
    hooked = getattr(model, "hooked", None)
    if hooked is None:
        return out
    try:
        out["armedInterventions"] = len(hooked.interventions)
    except Exception:  # noqa: BLE001 - advisory only
        pass
    try:
        out["forwardHookCount"] = sum(
            len(layer._forward_hooks) for layer in hooked.layers)
    except Exception:  # noqa: BLE001 - advisory only
        pass
    counters = getattr(hooked, "counters", None)
    if isinstance(counters, dict):
        for key in ("hookFires", "armedHookFires", "interventionApplications",
                    "interventionDispatches", "effectiveInjections",
                    "replacementTensors", "outputTuplesRebuilt",
                    "injectionCloneBytes"):
            out[key] = counters.get(key)
        out["cumulativeForwardPasses"] = counters.get("forwardPasses")
        out["cumulativeDecodeSteps"] = counters.get("decodeSteps")
        out["cumulativePrefillTokens"] = counters.get("prefillTokens")
    return out


def _record_facts(writer) -> dict:
    """Token counts for the record just emitted, READ off that record.

    Nothing is computed from text: the run loop's records carry what they carry,
    and a diagnostic that re-tokenized to fill a field would be measuring its own
    work. Today's sampled-generation record has no token-count key at all (the
    counts live in the generator, not the record), so these usually read null
    and ``decodeStepsThisRecord`` — the exact difference in the hook counter
    between two consecutive observations — is the reliable per-record token
    denominator. They are read anyway because ``recordTokenIDs`` studies DO
    carry the sequence, and because the key may be added upstream later.
    """
    out = {"promptTokens": None, "generatedTokens": None}
    try:
        records = getattr(writer, "records", None) or []
        record = records[-1]
    except Exception:  # noqa: BLE001 - advisory only
        return out
    if not isinstance(record, dict):
        return out
    for key in ("promptTokens", "promptTokenCount"):
        value = record.get(key)
        if isinstance(value, int):
            out["promptTokens"] = value
            break
    for key in ("generatedTokens", "outputTokens", "outputTokenCount"):
        value = record.get(key)
        if isinstance(value, int):
            out["generatedTokens"] = value
            break
    if out["generatedTokens"] is None:
        ids = record.get("outputTokenIDs")
        if isinstance(ids, list):
            out["generatedTokens"] = len(ids)
    return out


def _condition_facts(condition) -> dict:
    """Name, intervention state and ARMED CELL COUNT for this record.

    The cell count is what lets the operator correlate the RSS slope with the
    size of the armed band: a per-armed-layer-per-step retention scales with
    it, a per-step one does not.
    """
    out = {"condition": None, "interventionState": None,
           "armedCells": None, "armedLatentEdits": None}
    if condition is None:
        return out
    out["condition"] = getattr(condition, "name", None)
    out["interventionState"] = getattr(condition, "intervention_state", None)
    try:
        out["armedCells"] = len(getattr(condition, "injections", None) or [])
    except Exception:  # noqa: BLE001 - advisory only
        pass
    try:
        out["armedLatentEdits"] = len(getattr(condition, "latent_edits", None) or [])
    except Exception:  # noqa: BLE001 - advisory only
        pass
    return out


def _deep_facts() -> dict:
    """gc object census + live-tensor census (tier 2 only) — hunt 1's
    "is anything Python-reachable?" question, asked on the cluster."""
    out = {"gcObjects": None, "liveTensorCount": None, "liveTensorBytes": None}
    try:
        objects = gc.get_objects()
        out["gcObjects"] = len(objects)
    except Exception:  # noqa: BLE001 - advisory only
        return out
    try:
        import torch

        count = 0
        total = 0
        # The census walks EVERY live object, which includes torch's own
        # deprecation shims — merely isinstance-testing one emits a
        # FutureWarning. Suppressed here so a tier-2 run's log stays readable;
        # the warning says nothing about this process's memory.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for obj in objects:
                try:
                    if isinstance(obj, torch.Tensor):
                        count += 1
                        total += obj.element_size() * obj.nelement()
                except Exception:  # noqa: BLE001 - a broken __class__ is not our problem
                    continue
        out["liveTensorCount"] = count
        out["liveTensorBytes"] = total
    except Exception:  # noqa: BLE001 - advisory only
        pass
    return out


def _ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator


#: Distinguishes "no condition observed yet" from a condition legitimately
#: named ``None`` (an unnamed condition object), so record 0 baselines once.
_UNSET = object()


# --- the collector ------------------------------------------------------------


class MemoryDiagnostic:
    """One run directory's diagnostic stream."""

    def __init__(self, run_directory: str, *, tier: int = 1):
        self.run_directory = run_directory
        self.tier = tier
        self.path = os.path.join(run_directory, FILENAME)
        self.record_index = 0
        self.started_at = time.monotonic()
        #: First observation's RSS + counters — every RUN-GLOBAL delta is
        #: measured from there, so model-load memory never pollutes the slope.
        self.baseline: dict | None = None
        #: The same snapshot, retaken whenever the observed condition CHANGES.
        #: A run walks its conditions in sequence (baseline, then each steered
        #: cell band), so a single run-global baseline mixes arms: the slope
        #: attributed to condition N carries every earlier condition's growth,
        #: which is exactly the comparison the §15 matrix needs to make. The
        #: unprefixed ratios are therefore per-condition and the ``run*`` ones
        #: cumulative. Conditions are identified by NAME; a run that revisits a
        #: name (it does not today) would re-baseline on the same key, which is
        #: the conservative reading.
        self.condition_baseline: dict | None = None
        self.condition_key: object = _UNSET
        self.condition_index = -1
        self.condition_record_index = 0
        #: Previous observation's RSS + decode steps, for the marginal (record
        #: to record) reading — the one slope that is immune to both a stale
        #: baseline and a mid-run condition change.
        self.previous: dict | None = None

    def observe(self, *, model=None, writer=None, condition=None) -> dict:
        now = time.monotonic()
        line = {
            "schemaVersion": SCHEMA_VERSION,
            "recordIndex": self.record_index,
            "elapsedSeconds": round(now - self.started_at, 3),
            "tier": self.tier,
            # Echoed so a MALLOC_ARENA_MAX=2 control run is self-describing:
            # the arena hypothesis is tested by comparing two files, and the
            # file has to say which variant produced it.
            "mallocArenaMax": os.environ.get("MALLOC_ARENA_MAX"),
        }
        line.update(_condition_facts(condition))
        line.update(_process_memory())
        line.update(_accelerator_memory())
        line.update(_hook_facts(model))
        line.update(_record_facts(writer))
        try:
            line["writerRecords"] = len(getattr(writer, "records", None) or [])
        except Exception:  # noqa: BLE001 - advisory only
            line["writerRecords"] = None
        if self.tier >= 2:
            line.update(_deep_facts())

        snapshot = {
            "rssBytes": line.get("rssBytes"),
            "decodeSteps": line.get("cumulativeDecodeSteps"),
            "armedHookFires": line.get("armedHookFires"),
            "recordIndex": self.record_index,
        }
        if self.baseline is None:
            self.baseline = dict(snapshot)

        condition = line.get("condition")
        if self.condition_baseline is None or condition != self.condition_key:
            self.condition_baseline = dict(snapshot)
            self.condition_key = condition
            self.condition_index += 1
            self.condition_record_index = 0
        line["conditionIndex"] = self.condition_index
        line["conditionRecordIndex"] = self.condition_record_index
        line["conditionBaselineRecordIndex"] = self.condition_baseline.get(
            "recordIndex")

        # Per-CONDITION ratios keep the unprefixed names: they are the ones an
        # operator reads to attribute a slope to an arm, and the arm is what
        # the §15 matrix varies. For a single-condition file they equal the
        # run-global ``run*`` pair exactly.
        self._emit_deltas(line, self.condition_baseline, prefix="",
                          records=self.condition_record_index)
        self._emit_deltas(line, self.baseline, prefix="run",
                          records=self.record_index)

        # The marginal reading: growth since the PREVIOUS record, over the
        # decode steps of that interval. It answers "is it still leaking, right
        # now, at this rate" without any baseline at all — the reading that
        # survives a mid-run condition change and a plateau alike.
        previous = self.previous
        interval_rss = None
        interval_steps = None
        if previous is not None:
            interval_rss = _delta(line.get("rssBytes"), previous.get("rssBytes"))
            interval_steps = _delta(line.get("cumulativeDecodeSteps"),
                                    previous.get("decodeSteps"))
        line["rssDeltaSincePreviousRecord"] = interval_rss
        line["decodeStepsThisRecord"] = interval_steps
        line["intervalHostBytesPerDecodeStep"] = _ratio(interval_rss,
                                                        interval_steps)
        self.previous = dict(snapshot)

        self._append(line)
        self.record_index += 1
        self.condition_record_index += 1
        return line

    def _emit_deltas(self, line: dict, base: dict, *, prefix: str,
                     records: int) -> None:
        """Write one baseline's deltas and the three competing denominators.

        The denominators, and what each one would mean (open-issues §15):
          per decode step  ~230 KB  → the field-evidence signature
          per armed fire            → the armed-branch-churn hypothesis
          per record                → a record-linear leak the rate evidence
                                      was misread as token-linear
        """
        rss_delta = None
        if line.get("rssBytes") is not None and base.get("rssBytes") is not None:
            rss_delta = line["rssBytes"] - base["rssBytes"]
        steps_delta = _delta(line.get("cumulativeDecodeSteps"),
                             base.get("decodeSteps"))
        fires_delta = _delta(line.get("armedHookFires"),
                             base.get("armedHookFires"))

        def key(name: str) -> str:
            if not prefix:
                return name
            return prefix + name[0].upper() + name[1:]

        line[key("rssDeltaBytes")] = rss_delta
        line[key("decodeStepsSinceBaseline")] = steps_delta
        line[key("armedHookFiresSinceBaseline")] = fires_delta
        line[key("hostBytesPerDecodeStep")] = _ratio(rss_delta, steps_delta)
        line[key("hostBytesPerArmedHookFire")] = _ratio(rss_delta, fires_delta)
        line[key("hostBytesPerRecord")] = _ratio(rss_delta, records)

    def _append(self, line: dict) -> None:
        # flush() but no fsync: the point is surviving an OOM SIGKILL, and a
        # write() that reached the page cache survives the process dying. fsync
        # per record would only buy survival of a NODE crash, at a real cost.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return current - baseline


def observe(writer, model=None, condition=None) -> dict | None:
    """Record one line for the record just emitted, or do nothing.

    Returns ``None`` when the diagnostic is unset — WITHOUT constructing a
    collector, opening a file, importing torch, or reading any process state.
    """
    if level() <= 0:
        return None
    try:
        run_directory = getattr(writer, "run_directory", None)
        if not run_directory:
            return None
        diagnostic = _ACTIVE.get(run_directory)
        if diagnostic is None:
            diagnostic = MemoryDiagnostic(run_directory, tier=level())
            _ACTIVE[run_directory] = diagnostic
        return diagnostic.observe(model=model, writer=writer,
                                  condition=condition)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a run
        global _warned
        if not _warned:
            _warned = True
            sys.stderr.write(
                f"memory diagnostic disabled after an error "
                f"({type(exc).__name__}: {exc})\n")
        return None
