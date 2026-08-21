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


# --------------------------------------------------------------------------
# Containment: a slug is a NAME, never a path (review finding, 2026-08-21)
# --------------------------------------------------------------------------
# Most call sites build the slug from a verb plus an already-safe-named value,
# but several interpolate data straight out of a request body, a bundle's
# metadata, or a scenario file. One separator turned `runs/<stamp>-<slug>`
# into an arbitrary filesystem path.

@pytest.mark.parametrize("slug", [
    "gemmascope-../../../escaped",
    "../escaped",
    "..",
    ".",
    "nested/slug",
    "back\\slash",
    "/absolute",
    os.path.join(os.sep, "tmp", "absolute"),
    "nul\0byte",
    "",
])
def test_a_slug_that_could_leave_runs_is_refused(tmp_path, slug):
    with pytest.raises(paths.UnsafeRunSlug):
        paths.make_unique_run_directory(slug, root=str(tmp_path))
    # Nothing was created on the way to the refusal.
    runs_root = os.path.join(str(tmp_path), "runs")
    assert not os.path.exists(runs_root) or not os.listdir(runs_root)
    assert not os.path.exists(os.path.join(str(tmp_path), "escaped"))


def test_the_refusal_is_a_value_error_callers_can_catch(tmp_path):
    # `UnsafeRunSlug` subclasses ValueError so the routes that already map
    # ValueError to a 400 keep doing the right thing without a new handler.
    assert issubclass(paths.UnsafeRunSlug, ValueError)
    with pytest.raises(ValueError, match="path separator"):
        paths.make_unique_run_directory("a/b", root=str(tmp_path))


@pytest.mark.parametrize("slug", [
    SLUG,
    "exp-case1-run",
    "submit-my.study-run",
    "sae-feature-62389",
    "gemmascope-joy",
    "multi-agent-prisoners-dilemma-baseline",
    "reader-fit-concept_name",
    "vec 2026",
    "concept-café",
    "a" * 120,
])
def test_legitimate_slugs_are_untouched(tmp_path, frozen_stamp, slug):
    """Naming is unchanged for every ordinary slug — an existing run tree
    must stay resolvable by the names it already has on disk."""
    made = paths.make_unique_run_directory(slug, root=str(tmp_path))
    assert made == os.path.join(str(tmp_path), "runs", f"{STAMP}-{slug}")
    assert os.path.isdir(made)


def test_an_existing_run_directory_still_resolves(tmp_path, frozen_stamp):
    """The check guards CREATION only: nothing about reading a run tree that
    was written before it existed changes."""
    runs_root = os.path.join(str(tmp_path), "runs")
    legacy = os.path.join(runs_root, f"{STAMP}-{SLUG}")
    os.makedirs(legacy)
    with open(os.path.join(legacy, "report.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    assert paths.runs_directory(str(tmp_path)) == runs_root
    assert os.path.isfile(
        paths.resolve(os.path.join("runs", f"{STAMP}-{SLUG}", "report.json"),
                      str(tmp_path)))
    # …and a second call still ladders off the existing name.
    assert paths.make_unique_run_directory(SLUG, root=str(tmp_path)) == \
        os.path.join(runs_root, f"{STAMP}-{SLUG}-2")
