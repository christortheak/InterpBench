"""Enforce the deployment's bulk-transfer policy before opening a stream."""
from fastapi import HTTPException
import os
from .profile import ServerProfile


def max_upload_bytes() -> int:
    """Per-request deployment limit shared by artifact upload endpoints."""
    return int(os.environ.get("STEERLAB_MAX_UPLOAD_BYTES", str(4 * 1024**3)))


def require_http_transfer() -> None:
    method = ServerProfile.from_env().transfer_method
    if method not in (None, "", "http"):
        raise HTTPException(status_code=403, detail={
            "code": "external_transfer_required",
            "message": "This deployment does not permit HTTP artifact transfer.",
            "repairAction": (
                "Use the site's configured external transfer to its staging root, "
                "then inspect and submit the staged bundle; retrieve evidence with "
                "that transport and verify its reported SHA-256 before local import."),
        })
