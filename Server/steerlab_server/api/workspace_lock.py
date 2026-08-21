"""The one lock that makes a runtime workspace switch atomic against job
submission.

``POST /api/workspace/switch`` mutates a PROCESS GLOBAL (``STEERLAB_ROOT``)
that every path helper, profile, and resolver reads per call. It refuses while
any job is non-terminal — but that check and the mutation used to be two
unsynchronized steps, so a submission arriving between them was neither
refused nor contained: it resolved its early paths (a run directory, a
packaged bundle) against the OLD root and everything after the switch against
the NEW one, splitting one submission across two workspaces.

The two sides are asymmetric, so the lock is a reader/writer pair rather than
a mutex:

- **Submissions take the SHARED side** (:func:`submitting`). Any number run at
  once — they do not conflict with each other, only with a root that moves
  underneath them — so a slow bundle packaging never serializes its siblings.
- **The switch takes the EXCLUSIVE side** (:func:`switching`) and re-checks
  the active-job set INSIDE that critical section. Anything already registered
  is seen and refused; anything still arriving blocks until the root has
  moved, then resolves every path against the new one.

Writer-preferring: once a switch is waiting, new shared acquisitions queue
behind it, so a busy submit loop cannot starve it indefinitely.

The shared side is REENTRANT per thread. A submission entry point
(``submissions.submit_study``) holds it across the whole prepare-and-register
sequence and calls ``JobManager.submit``/``record_external``, which take it
again; without reentrancy that inner acquisition would deadlock the moment a
switch started waiting.

Lock ordering, everywhere: this lock is always acquired OUTSIDE
``JobManager._lock``, never the reverse.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class WorkspaceRootLock:
    """A small writer-preferring RW lock with a reentrant reader side."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0
        self._local = threading.local()

    @contextmanager
    def submitting(self):
        """Shared: hold while a submission resolves roots and registers."""
        depth = getattr(self._local, "depth", 0)
        if depth:  # already ours on this thread — nest without re-acquiring
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return
        with self._condition:
            while self._writer or self._writers_waiting:
                self._condition.wait()
            self._readers += 1
        self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._condition:
                self._readers -= 1
                if not self._readers:
                    self._condition.notify_all()

    @contextmanager
    def switching(self):
        """Exclusive: hold across the active-job re-check AND the mutation."""
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()

    # Diagnostics for tests; never a control flow input.
    @property
    def readers(self) -> int:
        return self._readers


#: Process-wide, because the root it guards is process-wide.
WORKSPACE_ROOT_LOCK = WorkspaceRootLock()


def submitting():
    return WORKSPACE_ROOT_LOCK.submitting()


def switching():
    return WORKSPACE_ROOT_LOCK.switching()


__all__ = ["WORKSPACE_ROOT_LOCK", "WorkspaceRootLock", "submitting", "switching"]
