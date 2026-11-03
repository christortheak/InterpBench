"""Peak device memory a study run used: per generated record, and the run's
maximum.

A long prompt can fit a card's finished cache and still run out of memory
while it is read, and until now a run did not say how close it came: the
headroom on a 217,000-token item on an 80 GB card was unknown, and nothing
could calibrate the submission's memory check. On CUDA the allocator keeps
its own peak counters, so each sampled generation is measured exactly as
``instrumentation_cost`` measures a response: ``reset_peak_memory_stats``
before it, ``max_memory_allocated`` and ``max_memory_reserved`` after it.

Two run-level files, beside (never inside) the generation records, so no
records reader on either engine or the results explorer meets a new field:

- ``device-memory.jsonl``: one line per measured generation: condition,
  promptID, promptIndex, sampleIndex, seed, the card (``deviceName``,
  ``totalBytes``), and the two peaks in bytes. A generation that failed is
  measured too and says so. The file only grows: a record generated again
  after a crash (measured, then lost before its record was written) gets a
  second line.
- ``device-memory.json``: the summary, rewritten after each line, from the
  LAST line of each record: how many records, the run's maximum of each peak
  (null until something is measured), and every card the run measured on
  (a resumed run may have moved to another).

On any other device the summary says the peaks were not measured, and why;
no records file is written. A panel's turns are not measured.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

from .prompt_reading import device_kind

SUMMARY_FILENAME = "device-memory.json"
RECORDS_FILENAME = "device-memory.jsonl"
SCHEMA_VERSION = 1
METHOD = ("torch.cuda.max_memory_allocated and torch.cuda.max_memory_reserved, "
          "read after each sampled generation, with "
          "torch.cuda.reset_peak_memory_stats before it: the allocator's own "
          "peaks for that generation, on the device that holds the model")
#: What names one record's line; the last line for a record wins.
RECORD_KEYS = ("condition", "promptID", "promptIndex", "sampleIndex", "seed")


def _write_json(path: str, document: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_summary(run_directory: str) -> dict | None:
    try:
        with open(os.path.join(run_directory, SUMMARY_FILENAME),
                  encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_lines(run_directory: str) -> list[dict]:
    """The records file's lines, skipping a torn last line."""
    lines = []
    try:
        with open(os.path.join(run_directory, RECORDS_FILENAME),
                  encoding="utf-8") as handle:
            for text in handle:
                try:
                    line = json.loads(text)
                except ValueError:
                    continue
                if isinstance(line, dict):
                    lines.append(line)
    except OSError:
        return []
    return lines


def not_measured(kind: str) -> dict:
    reason = ("the model names a CUDA device, but this process cannot read CUDA "
              "memory statistics" if kind == "cuda" else
              "peak device memory is measured on CUDA only, where the allocator "
              f"keeps peak counters; this run ran on {kind}")
    return {"schemaVersion": SCHEMA_VERSION, "measured": False, "device": kind,
            "reason": reason}


def summarise(lines: list[dict]) -> dict:
    """The run's figures from the last line of each record."""
    latest: dict[str, dict] = {}
    for line in lines:
        latest[json.dumps([line.get(k) for k in RECORD_KEYS], sort_keys=True)] = line
    kept = list(latest.values())

    def peak(key):
        values = [line[key] for line in kept if isinstance(line.get(key), int)]
        return max(values) if values else None

    devices = []
    for line in kept:
        card = {"deviceName": line.get("deviceName"), "totalBytes": line.get("totalBytes")}
        if card not in devices:
            devices.append(card)
    return {"recordCount": len(kept), "peakAllocatedBytes": peak("peakAllocatedBytes"),
            "peakReservedBytes": peak("peakReservedBytes"), "devices": devices}


def _measured_summary(lines: list[dict]) -> dict:
    return {"schemaVersion": SCHEMA_VERSION, "measured": True, "device": "cuda",
            "method": METHOD, "recordsFile": RECORDS_FILENAME, **summarise(lines)}


class DeviceMemoryLog:
    """The run's measurements. ``measure`` is a no-op off CUDA."""

    def __init__(self, run_directory: str, model, *, resuming: bool = False):
        self.run_directory = run_directory
        self.kind = device_kind(str(getattr(model, "device", "") or ""))
        self.torch = None
        self.device = None
        if self.kind == "cuda":
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.device = torch.device(str(model.device))
        path = os.path.join(run_directory, SUMMARY_FILENAME)
        if self.torch is None:
            if not resuming or read_summary(run_directory) is None:
                _write_json(path, not_measured(self.kind))
            return
        properties = self.torch.cuda.get_device_properties(self.device)
        self.card = {"deviceName": str(properties.name),
                     "totalBytes": int(properties.total_memory)}
        # A resumed run continues its own lines; the summary is rebuilt from
        # them, so nothing it says rests on a figure this session cannot see.
        self.lines = read_lines(run_directory) if resuming else []
        _write_json(path, _measured_summary(self.lines))

    @contextmanager
    def measure(self, **key):
        """Measure one generation; ``key`` names its record."""
        if self.torch is None:
            yield
            return
        torch, device = self.torch, self.device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        failed = True
        try:
            yield
            failed = False
        finally:
            # Best effort: a reading that cannot be taken (a faulted device
            # after an out-of-memory failure) never replaces the run's own
            # outcome.
            try:
                torch.cuda.synchronize(device)
                self._add({**key, **self.card,
                           "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                           "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
                           **({"generationFailed": True} if failed else {})})
            except Exception:  # noqa: BLE001 - see the comment above
                pass

    def _add(self, line: dict) -> None:
        with open(os.path.join(self.run_directory, RECORDS_FILENAME), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
        self.lines.append(line)
        _write_json(os.path.join(self.run_directory, SUMMARY_FILENAME),
                    _measured_summary(self.lines))


def merge_shards(shard_directories: list[str], into: str) -> None:
    """The merged run's files: every shard's lines in shard order, a summary
    rebuilt from them (last line per record), and each shard's own figures."""
    summaries = [read_summary(directory) for directory in shard_directories]
    if all(summary is None for summary in summaries):
        return
    measured = [summary for summary in summaries
                if summary is not None and summary.get("measured")]
    if not measured:
        first = next(summary for summary in summaries if summary is not None)
        _write_json(os.path.join(into, SUMMARY_FILENAME), first)
        return
    lines: list[dict] = []
    with open(os.path.join(into, RECORDS_FILENAME), "wb") as out:
        for directory in shard_directories:
            path = os.path.join(directory, RECORDS_FILENAME)
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    out.write(handle.read())
            lines.extend(read_lines(directory))
    merged = _measured_summary(lines)
    merged["shards"] = [
        {"shardIndex": index, "measured": bool(summary and summary.get("measured")),
         **(summarise(read_lines(directory))
            if summary and summary.get("measured") else {})}
        for index, (directory, summary) in enumerate(zip(shard_directories, summaries))]
    _write_json(os.path.join(into, SUMMARY_FILENAME), merged)
