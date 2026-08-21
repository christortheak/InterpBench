"""``make_unique_run_directory`` hands every caller its OWN directory, even when
several callers resolve the same millisecond stamp.

Run directories are immutable (CLAUDE.md › Data & reproducibility): two callers
must never be handed one directory, and a caller must never die because another
won the name first. The old check-then-create loop (``while os.path.exists`` →
``os.makedirs``) had a TOCTOU window between the two — reachable in production,
where Slurm generation shards create run directories concurrently on a shared
filesystem. The fix makes creation itself the exclusivity test; these tests pin
that by (a) racing a competitor into the window deterministically, (b) racing
real threads, and (c) checking the retry budget fails loudly rather than
spinning.
"""

import os
import threading
from datetime import datetime, timezone

import pytest

from steerlab_server.experiment import paths


FIXED = datetime(2026, 7, 24, 12, 0, 0, 123_456, tzinfo=timezone.utc)
STAMP = "20260724T120000123"
SLUG = "exp-shard-run"


class _FrozenClock:
    """``datetime`` stand-in whose ``now()`` always lands on one millisecond —
    the condition that makes shards collide on a real cluster."""

    @staticmethod
    def now(tz=None):  # noqa: ARG004 - signature parity with datetime.now
        return FIXED


@pytest.fixture
def frozen_stamp(monkeypatch):
    monkeypatch.setattr(paths, "datetime", _FrozenClock)


def test_same_stamp_yields_distinct_directories(tmp_path, frozen_stamp):
    """Sequential callers on one stamp get the documented suffix ladder."""
    made = [paths.make_unique_run_directory(SLUG, root=str(tmp_path))
            for _ in range(4)]
    assert len(set(made)) == 4
    assert all(os.path.isdir(path) for path in made)
    runs_root = os.path.join(str(tmp_path), "runs")
    assert made == [
        os.path.join(runs_root, f"{STAMP}-{SLUG}"),
        os.path.join(runs_root, f"{STAMP}-{SLUG}-2"),
        os.path.join(runs_root, f"{STAMP}-{SLUG}-3"),
        os.path.join(runs_root, f"{STAMP}-{SLUG}-4"),
    ]


def test_competitor_winning_the_toctou_window_does_not_raise(
        tmp_path, monkeypatch, frozen_stamp):
    """A competitor that claims the name *inside* the check→create window must
    cost us a suffix, not the run. Deterministic stand-in for the second shard:
    it creates the un-suffixed path immediately before our own ``mkdir`` for it
    lands, which is exactly the interleaving the old code died on."""
    real_makedirs = os.makedirs
    target = os.path.join(str(tmp_path), "runs", f"{STAMP}-{SLUG}")
    fired = []

    def racing_makedirs(path, *args, **kwargs):
        if path == target and not fired:
            fired.append(path)
            real_makedirs(path)  # the other shard wins the name
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(paths.os, "makedirs", racing_makedirs)
    made = paths.make_unique_run_directory(SLUG, root=str(tmp_path))
    assert fired, "the racing competitor never ran — test no longer simulates the race"
    assert made == f"{target}-2"
    assert os.path.isdir(made)


def test_concurrent_callers_each_get_their_own_directory(tmp_path, frozen_stamp):
    """Real threads, one stamp: every caller returns a distinct directory and
    none raises FileExistsError."""
    workers = 16
    barrier = threading.Barrier(workers)
    results: list[str] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        try:
            barrier.wait()
            path = paths.make_unique_run_directory(SLUG, root=str(tmp_path))
        except BaseException as exc:  # noqa: BLE001 - the point is to report it
            with lock:
                failures.append(exc)
            return
        with lock:
            results.append(path)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"concurrent creation raised: {failures!r}"
    assert len(results) == workers
    assert len(set(results)) == workers, "two callers were handed one run directory"
    assert all(os.path.isdir(path) for path in results)


def test_exhausted_retry_budget_raises_loudly(tmp_path, monkeypatch, frozen_stamp):
    monkeypatch.setattr(paths, "MAX_RUN_DIRECTORY_ATTEMPTS", 3)
    runs_root = os.path.join(str(tmp_path), "runs")
    os.makedirs(runs_root, exist_ok=True)
    for name in (f"{STAMP}-{SLUG}", f"{STAMP}-{SLUG}-2", f"{STAMP}-{SLUG}-3"):
        os.makedirs(os.path.join(runs_root, name))
    with pytest.raises(RuntimeError, match="unique run directory"):
        paths.make_unique_run_directory(SLUG, root=str(tmp_path))
