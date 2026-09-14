"""Bounded, dedicated executors for bulk transfers, so a burst of downloads or
one long export can never starve the rest of the API.

The 2026-09-13 incident: a Mac app launched for an eight-second build check
began fetching every succeeded job's evidence bundle it had never seen from a
live controller (hundreds of ``GET /api/bundles/download``) and requested a
science export that tarred and hashed 12 GB of output; the app was then killed
mid-transfer. The controller stopped answering even ``GET /api/capabilities``
for more than half an hour. Reading the stack shows why that is possible:

- every sync ``def`` route runs on anyio's DEFAULT thread limiter (40 tokens,
  shared process-wide) via Starlette's ``run_in_threadpool``;
- ``FileResponse`` streams a file by reading 64 KiB chunks through
  ``anyio.open_file`` — each read is ``to_thread.run_sync`` on that SAME
  default limiter;
- uvicorn advertises ASGI spec 2.3 and its ``send()`` silently returns once
  the client is gone, so a download whose client vanished keeps reading its
  whole file into the void, chunk after chunk, competing for the tokens the
  capabilities probe needs;
- the science export tars and hashes its output inside the sync handler, so
  it holds one of those tokens for its entire duration.

This module gives transfers their own bounded pools:

``TransferLimits``
    One per ``ServiceState``. A download pool (``max_downloads`` slots, env
    ``STEERLAB_DOWNLOAD_CONCURRENCY``, default 8) whose file I/O runs on a
    dedicated ``ThreadPoolExecutor`` sized to the slots, and an export pool of
    exactly one worker, so exports serialize and never touch the request
    threadpool. A download that finds every slot taken is refused at once with
    ``503`` and ``Retry-After`` instead of queueing; ``/api/capabilities``,
    ``/api/jobs``, and the science plan/submit routes keep answering.

``BoundedFileResponse``
    Serves a file with ``FileResponse``-identical headers and bytes, reading
    chunks on the download executor and checking for a client disconnect
    between chunks, so an abandoned transfer stops within one chunk and
    releases its slot.

Nothing here changes a route path, a success body, or the transfer-policy
gate (``require_http_transfer`` still runs first, unchanged).
"""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.requests import Request

#: Advisory pause a saturated client should observe before retrying.
RETRY_AFTER_SECONDS = 5
#: Same chunk size as Starlette's FileResponse, so throughput is unchanged.
CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_DOWNLOADS = 8


def download_concurrency() -> int:
    """Concurrent ``/api/bundles/download`` transfers this process serves."""
    raw = os.environ.get("STEERLAB_DOWNLOAD_CONCURRENCY", "")
    try:
        value = int(raw) if raw else DEFAULT_MAX_DOWNLOADS
    except ValueError:
        value = DEFAULT_MAX_DOWNLOADS
    return max(1, value)


class TransferLimits:
    """Slot accounting plus the two dedicated executors."""

    def __init__(self, max_downloads: int | None = None) -> None:
        self.max_downloads = max(1, max_downloads or download_concurrency())
        self._lock = threading.Lock()
        self.active_downloads = 0
        self.peak_downloads = 0
        self.refused_downloads = 0
        self.active_exports = 0
        self.peak_exports = 0
        self._download_executor = ThreadPoolExecutor(
            max_workers=self.max_downloads, thread_name_prefix="steerlab-download")
        self._export_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="steerlab-export")

    # -- download slots -----------------------------------------------------

    def try_acquire_download(self) -> bool:
        """Take a slot now, or report saturation (never waits)."""
        with self._lock:
            if self.active_downloads >= self.max_downloads:
                self.refused_downloads += 1
                return False
            self.active_downloads += 1
            self.peak_downloads = max(self.peak_downloads, self.active_downloads)
            return True

    def release_download(self) -> None:
        with self._lock:
            if self.active_downloads > 0:
                self.active_downloads -= 1

    def saturated_response(self) -> JSONResponse:
        """The refusal a caller gets when every download slot is busy."""
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            content={"detail": {
                "code": "downloadCapacity",
                "message": (
                    f"this controller is already streaming {self.max_downloads} "
                    "artifacts; the request was not queued"),
                "repairAction": (
                    f"retry after {RETRY_AFTER_SECONDS} seconds (Retry-After); "
                    "raise STEERLAB_DOWNLOAD_CONCURRENCY on the controller "
                    "only after confirming the shared filesystem keeps up"),
            }})

    async def run_download_io(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run blocking file I/O for a download on the download executor —
        never on the request threadpool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._download_executor, func, *args)

    # -- exports -------------------------------------------------------------

    async def run_export(self, func: Callable[[], Any]) -> Any:
        """Run one export to completion on the single export worker. Callers
        queue behind each other there, holding no request-threadpool token
        while they wait or work. A client that disconnects mid-export does not
        cancel it: the archive it produces is keyed by the job context, so the
        researcher's retry finds it ready instead of packaging 12 GB twice."""
        loop = asyncio.get_running_loop()
        with self._lock:
            self.active_exports += 1
            self.peak_exports = max(self.peak_exports, self.active_exports)
        try:
            return await loop.run_in_executor(self._export_executor, func)
        finally:
            with self._lock:
                self.active_exports -= 1

    def snapshot(self) -> dict:
        """Counters for tests and operator diagnostics."""
        with self._lock:
            return {
                "maxDownloads": self.max_downloads,
                "activeDownloads": self.active_downloads,
                "peakDownloads": self.peak_downloads,
                "refusedDownloads": self.refused_downloads,
                "activeExports": self.active_exports,
                "peakExports": self.peak_exports,
            }

    def shutdown(self) -> None:
        self._download_executor.shutdown(wait=False, cancel_futures=True)
        self._export_executor.shutdown(wait=False, cancel_futures=True)


class BoundedFileResponse(Response):
    """A file download that owns one download slot for its whole life.

    Headers are exactly ``FileResponse``'s (media type guessed from the name,
    ``content-disposition: attachment; filename=…``, ``accept-ranges``,
    ``content-length``, ``last-modified``, ``etag``); the body is the file's
    bytes in 64 KiB chunks. Every ``stat``/``open``/``read`` runs on the
    download executor, and between chunks the response asks the ASGI
    ``receive`` channel whether the client is still there — an abandoned
    transfer stops within one chunk. The slot is released on every exit.

    Range requests are answered with the full body (``200``), as no SteerLab
    client issues them; ``HEAD`` returns the headers only.
    """

    def __init__(self, path: str, *, filename: str, limits: TransferLimits) -> None:
        self.path = path
        self.limits = limits
        self.bytes_sent = 0
        self._slot_released = False
        template = FileResponse(path, filename=filename)
        super().__init__(content=None, status_code=200, media_type=template.media_type)
        # Take the template's headers wholesale: same media type, disposition,
        # and accept-ranges a stock FileResponse would have sent.
        self.raw_headers = list(template.raw_headers)
        self._template = template

    def release_slot(self) -> None:
        if not self._slot_released:
            self._slot_released = True
            self.limits.release_download()

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        request = Request(scope, receive)
        handle = None
        try:
            stat_result = await self.limits.run_download_io(os.stat, self.path)
            self._template.set_stat_headers(stat_result)
            self.raw_headers = list(self._template.raw_headers)
            await send({"type": "http.response.start", "status": self.status_code,
                        "headers": self.raw_headers})
            if scope.get("method", "GET").upper() == "HEAD":
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            handle = await self.limits.run_download_io(open, self.path, "rb")
            more_body = True
            while more_body:
                chunk = await self.limits.run_download_io(handle.read, CHUNK_SIZE)
                more_body = len(chunk) == CHUNK_SIZE
                if more_body and await request.is_disconnected():
                    # The client is gone: stop reading, stop sending. Nothing
                    # downstream will ever see the rest of this body.
                    return
                await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
                self.bytes_sent += len(chunk)
        finally:
            if handle is not None:
                handle.close()
            self.release_slot()


_FALLBACK_LIMITS: TransferLimits | None = None
_FALLBACK_LOCK = threading.Lock()


def limits_for(state: Any) -> TransferLimits:
    """The ``TransferLimits`` a router's state carries (``ServiceState``
    constructs one), or a process-wide fallback for the bare namespaces the
    test suite hands routers — attached to the state when it accepts
    attributes, so one test's counters stay its own."""
    existing = getattr(state, "transfers", None)
    if isinstance(existing, TransferLimits):
        return existing
    global _FALLBACK_LIMITS
    with _FALLBACK_LOCK:
        if _FALLBACK_LIMITS is None:
            _FALLBACK_LIMITS = TransferLimits()
        limits = _FALLBACK_LIMITS
    try:
        setattr(state, "transfers", TransferLimits())
        return state.transfers
    except (AttributeError, TypeError):
        return limits


async def serve_download(limits: TransferLimits, resolve: Callable[[], str]) -> Response:
    """Handler body for ``GET /api/bundles/download`` after the transfer
    policy gate: take a slot (or refuse with 503), resolve and validate the
    path on the download executor while holding it, and hand back a response
    that releases the slot when the transfer ends however it ends."""
    if not limits.try_acquire_download():
        return limits.saturated_response()
    try:
        safe_path = await limits.run_download_io(resolve)
        return BoundedFileResponse(safe_path, filename=os.path.basename(safe_path), limits=limits)
    except BaseException:
        limits.release_download()
        raise


__all__ = [
    "BoundedFileResponse",
    "CHUNK_SIZE",
    "RETRY_AFTER_SECONDS",
    "TransferLimits",
    "download_concurrency",
    "limits_for",
    "serve_download",
]
