"""The lens library over HTTP (public issues #13 and #14).

A site keeps each converted lens file once, under its SHA-256, outside any
workspace (``jlens/lens_library.py``). These routes ask what it holds, send a
lens in checked parts, add a file that is already on the site, and put a lens
record in the serving workspace. They are thin: every rule lives in the
library module, and every refusal carries its code and a repair.

A part is streamed to disk and never held in memory. Checking a whole lens
(``complete``, ``adopt``) reads gigabytes, so it runs off the request thread
pool and the rest of the API keeps answering.
"""
import asyncio
import hashlib

from fastapi import APIRouter, HTTPException, Request

from . import diagnostic_transport
from .profile import ServerProfile
from .transfer_policy import require_http_transfer
from .workspace_lock import submitting
from ..jlens import lens_library as library


def refusal(exc):
    """A declined request as the house refusal: ``reason`` as the science
    routes give it, and the same sentence as ``message``, which the Python
    client prints."""
    reason = str(exc)
    known = isinstance(exc, library.LensLibraryRefusal)
    return HTTPException(exc.status if known else 409, detail={
        'code': exc.code if known else 'lensLibraryRefused', 'reason': reason, 'message': reason,
        'repairAction': exc.repair_action if known else
        "Check the request against the lens record (`lens.json`), then try again. If the engine's storage is the "
        "cause, the reason above names it for the site's operator."})


def build_router(state):
    router = APIRouter()

    def answer(work):
        try:
            return work()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    def exactly(body, wanted):
        if not isinstance(body, dict) or set(body) != set(wanted):
            raise library.LensLibraryRefusal(
                'lensLibraryRequestInvalid',
                'Supply exactly ' + (', '.join(wanted) if wanted else 'an empty object') + '.',
                'Send the lens with `lens-push`, which makes these requests.', status=400)

    @router.get('/api/lens-library/{sha}')
    def status(sha: str):
        return answer(lambda: library.status(sha))

    @router.post('/api/lens-library/register')
    def register(body: dict):
        def work():
            exactly(body, ['record'])
            with submitting():
                return library.register(body['record'], ServerProfile.from_env().root)
        return answer(work)

    @router.post('/api/lens-library/adopt')
    async def adopt(body: dict):
        def work():
            exactly(body, ['path', 'sha256'])
            return library.adopt(body['path'], body['sha256'], diagnostic_transport.storage(ServerProfile.from_env()))
        try:
            return await asyncio.to_thread(work)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    @router.post('/api/lens-library/{sha}/begin')
    def begin(sha: str, body: dict):
        return answer(lambda: library.begin(sha, body))

    @router.put('/api/lens-library/{sha}/parts/{index}')
    async def part(sha: str, index: str, request: Request):
        require_http_transfer()
        try:
            expected = await asyncio.to_thread(library.expect_part, sha, index, request.headers.get('x-content-sha256'))
            if expected is None:
                return {'received': True}   # the library holds this lens already; nothing to keep
            try:
                digest, total = hashlib.sha256(), 0
                with library.spool(sha) as path:
                    with open(path, 'xb') as stream:
                        async for chunk in request.stream():
                            total += len(chunk)
                            if total > expected['bytes']:
                                break       # never spool more than the plan allows
                            digest.update(chunk)
                            await asyncio.to_thread(stream.write, chunk)
                    await asyncio.to_thread(library.commit_part, sha, expected, path, digest.hexdigest(), total)
            except library.LensLibraryRefusal as exc:
                # Another sender finished this lens while the part was in
                # flight: its upload folder is gone because the lens is held.
                if exc.code == 'lensUploadNotBegun' and library.status(sha)['held']:
                    return {'received': True}
                raise
            return {'received': True}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    @router.post('/api/lens-library/{sha}/complete')
    async def complete(sha: str, body: dict | None = None):
        def work():
            exactly(body or {}, [])
            return library.complete(sha)
        try:
            return await asyncio.to_thread(work)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    return router
