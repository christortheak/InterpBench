"""Peak device memory per generated record and for the run, beside the
records. CUDA is faked: the allocator's peak counters are the only thing
read, and the expected values are written by hand."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import device_memory, sharding, tasks

import test_prompt_reading as reading_fixture

GIB = 1024 ** 3


class _FakeAllocator:
    """Stands in for torch.cuda's peak counters on one 80 GB card."""

    def __init__(self):
        self.allocated = self.reserved = 0
        self.resets = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self.reset)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                            lambda device=None: self.allocated)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                            lambda device=None: self.reserved)
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda device=None: SimpleNamespace(
                                name="Example 80GB", total_memory=80 * GIB))
        return self

    def reset(self, device=None):
        self.allocated = self.reserved = 0
        self.resets += 1

    def use(self, allocated_gib, reserved_gib):
        self.allocated = int(allocated_gib * GIB)
        self.reserved = int(reserved_gib * GIB)


def _summary(run_dir):
    with open(os.path.join(run_dir, device_memory.SUMMARY_FILENAME),
              encoding="utf-8") as handle:
        return json.load(handle)


def _lines(run_dir):
    path = os.path.join(run_dir, device_memory.RECORDS_FILENAME)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_each_generation_is_measured_and_the_run_keeps_its_maximum(
        tmp_path, monkeypatch):
    allocator = _FakeAllocator().install(monkeypatch)
    model = SimpleNamespace(device="cuda")
    log = device_memory.DeviceMemoryLog(str(tmp_path), model)
    with log.measure(condition="baseline", promptID="a", promptIndex=0,
                     sampleIndex=0, seed=0):
        allocator.use(71, 78)
    with log.measure(condition="baseline", promptID="b", promptIndex=1,
                     sampleIndex=0, seed=0):
        allocator.use(56, 60)
    assert allocator.resets == 2  # reset before each generation
    assert [(line["promptID"], line["peakAllocatedBytes"], line["peakReservedBytes"])
            for line in _lines(tmp_path)] == [
        ("a", 71 * GIB, 78 * GIB), ("b", 56 * GIB, 60 * GIB)]
    summary = _summary(tmp_path)
    assert (summary["measured"], summary["device"], summary["deviceName"],
            summary["totalBytes"], summary["recordCount"],
            summary["peakAllocatedBytes"], summary["peakReservedBytes"]) == (
        True, "cuda", "Example 80GB", 80 * GIB, 2, 71 * GIB, 78 * GIB)
    # A resumed run continues the same files and keeps the earlier maximum.
    resumed = device_memory.DeviceMemoryLog(str(tmp_path), model, resuming=True)
    with resumed.measure(condition="steered", promptID="a", promptIndex=0,
                         sampleIndex=0, seed=0):
        allocator.use(40, 41)
    summary = _summary(tmp_path)
    assert (summary["recordCount"], summary["peakAllocatedBytes"]) == (3, 71 * GIB)
    assert len(_lines(tmp_path)) == 3


def test_a_failed_generation_is_measured_and_still_fails(tmp_path, monkeypatch):
    allocator = _FakeAllocator().install(monkeypatch)
    log = device_memory.DeviceMemoryLog(str(tmp_path), SimpleNamespace(device="cuda"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with log.measure(condition="baseline", promptID="a", promptIndex=0,
                         sampleIndex=0, seed=0):
            allocator.use(79, 79.2)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    [line] = _lines(tmp_path)
    assert line["generationFailed"] is True
    assert line["peakAllocatedBytes"] == 79 * GIB


def test_off_cuda_the_file_says_the_peaks_were_not_measured(tmp_path):
    log = device_memory.DeviceMemoryLog(str(tmp_path), SimpleNamespace(device="mps"))
    with log.measure(condition="baseline", promptID="a", promptIndex=0,
                     sampleIndex=0, seed=0):
        pass
    assert _summary(tmp_path) == {
        "schemaVersion": 1, "measured": False, "device": "mps",
        "reason": ("peak device memory is measured on CUDA only, where the "
                   "allocator keeps peak counters; this run ran on mps")}
    assert _lines(tmp_path) == []


def _peaks_by_prompt(monkeypatch, allocator):
    """The fake generation's peak follows the prompt it reads."""
    import steerlab_server.experiment.generate as generate_mod
    peaks = {"long-packet": (71, 78), "short-note": (52, 53)}

    def generate(model, prompt, *, token_ids_out=None, **_):
        allocator.use(*peaks["long-packet" if "long-packet" in prompt else "short-note"])
        return f"an answer to {prompt}"
    monkeypatch.setattr(generate_mod, "generate", generate)


def test_a_study_run_on_cuda_records_each_records_peak(tmp_path, monkeypatch):
    root = str(tmp_path)
    allocator = _FakeAllocator().install(monkeypatch)
    prompts = reading_fixture._study(root, "memory-run")
    reading_fixture._patch(monkeypatch)
    _peaks_by_prompt(monkeypatch, allocator)
    run_dir = tasks.run("memory-run", prompts, root,
                        model_provider=reading_fixture._cuda_model,
                        log=lambda *_: None)
    lines = _lines(run_dir)
    assert sorted((line["condition"], line["promptID"], line["peakAllocatedBytes"])
                  for line in lines) == [
        ("baseline", "long-packet", 71 * GIB), ("baseline", "short-note", 52 * GIB),
        ("steered", "long-packet", 71 * GIB), ("steered", "short-note", 52 * GIB)]
    summary = _summary(run_dir)
    assert (summary["recordCount"], summary["peakAllocatedBytes"],
            summary["peakReservedBytes"]) == (4, 71 * GIB, 78 * GIB)
    # The generation records themselves are unchanged in shape.
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        assert not [json.loads(line) for line in h
                    if "peakAllocatedBytes" in line]


def test_a_shard_merge_keeps_every_shards_peaks(tmp_path, monkeypatch):
    root = str(tmp_path)
    allocator = _FakeAllocator().install(monkeypatch)
    prompts = reading_fixture._study(root, "memory-shards")
    dirs = []
    for index in range(2):
        reading_fixture._patch(monkeypatch)
        _peaks_by_prompt(monkeypatch, allocator)
        dirs.append(tasks.run("memory-shards", prompts, root,
                              model_provider=reading_fixture._cuda_model,
                              log=lambda *_: None,
                              shard=sharding.ShardSpec(index, 2)))
    merged = sharding.merge_shard_runs("memory-shards", dirs, root=root)
    assert len(_lines(merged)) == 4
    summary = _summary(merged)
    assert (summary["recordCount"], summary["peakAllocatedBytes"]) == (4, 71 * GIB)
    assert [shard["recordCount"] for shard in summary["shards"]] == [2, 2]
