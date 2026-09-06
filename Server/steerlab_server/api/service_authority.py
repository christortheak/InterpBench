"""Service-role authorization, independent of tokens and compute topology."""
from __future__ import annotations

import os
from starlette.routing import compile_path
from .route_roles import CENSUS, Role

# Static spellings take precedence over parameterized spellings, as in routing.
_ROUTES = tuple((entry, compile_path(entry.path)[0]) for entry in
                sorted(CENSUS, key=lambda entry: entry.path.count("{")))


def service_role() -> str:
    role = os.environ.get("STEERLAB_SERVICE_ROLE", "workbench")
    if role not in {"runner", "workbench"}:
        raise ValueError("STEERLAB_SERVICE_ROLE must be runner or workbench")
    return role


def role_for_request(method: str, path: str) -> Role | None:
    for entry, pattern in _ROUTES:
        if entry.method == method.upper() and pattern.fullmatch(path):
            return entry.role
    return None


def refusal(method: str, path: str) -> dict | None:
    try:
        role = service_role()
    except ValueError as exc:
        return {"code": "invalid_service_role", "message": str(exc),
                "repairAction": "Configure STEERLAB_SERVICE_ROLE as runner or workbench and restart."}
    if role == "workbench":
        return None
    declared = role_for_request(method, path)
    if declared in {Role.RUNNER, Role.BOTH}:
        return None
    return {
        "code": "workbench_required" if declared == Role.WORKBENCH else "undeclared_runner_operation",
        "message": "This runner does not serve independent workspace authoring or interactive workbench operations.",
        "repairAction": "Author in the client workspace or connect a workbench service. "
                        "Submit pinned inputs to this runner through the bundle protocol.",
        "serviceRole": "runner",
    }
