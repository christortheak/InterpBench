"""Workbench-only file staging and custom instrument import adapters."""
import asyncio
import hashlib
from pathlib import Path
import re
import tempfile

from fastapi import APIRouter, HTTPException, Request

from .profile import ServerProfile
from .workspace_lock import submitting
from .transfer_policy import require_http_transfer
from ..experiment import artifact_imports, artifact_sources, diagnostic_archives


def description_path(reference, root):
    if not isinstance(reference, str) or not reference:
        raise artifact_sources.ImportRefusal('Choose a description file staged in this workbench workspace.')
    base = Path(root).resolve()
    target = Path(reference)
    if target.is_absolute():
        try: reference = target.relative_to(base).as_posix()
        except ValueError as exc:
            raise artifact_sources.ImportRefusal('Stage the source files in this workbench workspace first.') from exc
    return diagnostic_archives.ordinary(base, reference)


def refusal(exc):
    return HTTPException(400, detail={'code': 'artifactImportRefused', 'reason': str(exc),
                                     'repairAction': artifact_sources.ImportRefusal.repair_action})


def build_router(state):
    router = APIRouter()

    @router.post('/api/artifact-imports/stage/{source_id}/{file_path:path}')
    async def stage(source_id: str, file_path: str, request: Request):
        """Stream one source file; never buffer a multi-gigabyte lens in RAM."""
        require_http_transfer()
        try:
            if not re.fullmatch('[0-9a-f]{32}', source_id):
                raise artifact_sources.ImportRefusal('Use a fresh source ID (32 lowercase hex characters).')
            expected = request.headers.get('x-content-sha256', '')
            if not re.fullmatch('[0-9a-f]{64}', expected):
                raise artifact_sources.ImportRefusal('Supply X-Content-SHA256 for the exact file being uploaded.')
            root = Path(ServerProfile.from_env().root).resolve()
            relative = '.steerlab/artifact-inputs/' + source_id + '/' + file_path
            target = diagnostic_archives.ordinary(root, relative, missing=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Only this temporary file is removed on failure. Existing sources
            # are never replaced, even if another upload wins publication.
            with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
                incoming = Path(temporary) / 'incoming'
                digest = hashlib.sha256(); total = 0
                with incoming.open('xb') as stream:
                    async for chunk in request.stream():
                        total += len(chunk)
                        if total > 32 * 1024**3:
                            raise artifact_sources.ImportRefusal('This file exceeds the 32 GiB HTTP staging limit; stage it through the cluster file-transfer tools.')
                        digest.update(chunk)
                        await asyncio.to_thread(stream.write, chunk)
                if digest.hexdigest() != expected:
                    raise artifact_sources.ImportRefusal('Uploaded bytes do not match the supplied hash; no source file was published.')
                diagnostic_archives.publish_file(incoming, target)
            return {'path': relative, 'sha256': expected, 'bytes': total}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    @router.post('/api/artifact-imports/plan')
    def plan(body: dict):
        try:
            if set(body) != {'descriptionFile'}:
                raise artifact_sources.ImportRefusal('Supply descriptionFile.')
            root = ServerProfile.from_env().root
            path = description_path(body['descriptionFile'], root)
            return artifact_imports.inspect_source(path, root)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    @router.post('/api/artifact-imports/import')
    def publish(body: dict):
        try:
            if set(body) != {'descriptionFile', 'planSHA256'}:
                raise artifact_sources.ImportRefusal('Supply descriptionFile and the reviewed planSHA256.')
            with submitting():
                root = ServerProfile.from_env().root
                path = description_path(body['descriptionFile'], root)
                # Cheap admission; the owner rechecks all bytes in the worker.
                if not re.fullmatch('[0-9a-f]{64}', body['planSHA256']):
                    raise artifact_sources.ImportRefusal('Supply the hash returned by source review.')
                def work(job):
                    job.log('Importing reviewed source files into a fresh library destination.')
                    return artifact_imports.publish(path, root, body['planSHA256'])
                return {'jobId': state.jobs.submit('artifact-import', work).id}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise refusal(exc) from exc

    return router
