"""FastAPI application entry point.

Run with ``steerlab-server serve`` or ``uvicorn steerlab_server.api.app:app``.
The default bind remains localhost. Cluster deployments should run behind Open
OnDemand or an SSH tunnel and enable token/external auth for API routes.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from . import gpu_session
from .posture import LOOPBACK_HOSTS
from .profile import ServerProfile, server_role
from .routes import ServiceState, build_router

# One process-wide service: a single loaded model + slot locks + the job manager.
state = ServiceState()


def _controller_jobs() -> list[dict]:
    """The controller's own jobs, as dicts, for the /api/state overlay: a
    proxied worker /api/state always reports an empty jobs array (the worker
    runs no job subsystem — single-writer rule), which made the jobs badge
    read zero during a live session. The controller's list is the truth."""
    manager = state.job_manager_or_none
    if manager is None:
        return []
    return [j.to_dict() for j in manager.list()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if server_role() == "gpu-session":
        # Same refusal the controller job template enforces in bash, here in
        # Python so NO launch path can skip it: a non-loopback bind without a
        # bearer token would be an open instrument on the cluster network.
        # (cli._serve refuses earlier with a friendlier message and hydrates
        # the token from STEERLAB_AUTH_TOKEN_FILE; this guard covers direct
        # `uvicorn steerlab_server.api.app:app` launches.)
        profile = ServerProfile.from_env()
        if profile.bind not in _LOOPBACK and not os.environ.get("STEERLAB_AUTH_TOKEN"):
            raise RuntimeError(
                "refusing to serve role=gpu-session on a non-loopback bind "
                "without STEERLAB_AUTH_TOKEN — set the token (or "
                "STEERLAB_AUTH_TOKEN_FILE) before starting the worker")
        # Discovery record + idle timer up-front, so the controller can find
        # this worker before the first request arrives.
        gpu_session.ensure_worker()
    # Only the Slurm executor needs a background reconciler; the local executor
    # tracks jobs in-process, and a gpu-session worker has no job subsystem at
    # all (single-writer rule). Keeps sacct out of dev/test runs.
    manager = state.job_manager_or_none
    if manager is not None and ServerProfile.from_env().executor == "slurm":
        manager.start_monitor()
    yield
    if manager is not None:
        manager.stop_monitor()


app = FastAPI(title="SteerLab cluster engine", version="0.1.0", lifespan=lifespan)
app.include_router(build_router(state))

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

_LOOPBACK = LOOPBACK_HOSTS
_DEFAULT_PORTS = {"http": "80", "https": "443"}

# Routes that accept an arbitrary command/env or read caller-named files and so
# must never be reachable without a token off a plain local box. They stay open
# on a local+localhost dev process (still CSRF-guarded below) but require a
# bearer token as soon as the server runs a Slurm executor, a non-local profile,
# or a non-loopback bind.
_PRIVILEGED_PREFIXES = (
    "/api/slurm/submit",
    "/api/slurm/bundle",
    "/api/bundles/upload",
    "/api/bundles/download",
    "/api/studies/submit",
    "/api/studies/submit-bundle",
    # The WHOLE fine-tune family: /train runs GPU compute, /submit runs
    # sbatch, and /plan reads caller-named workspace files. All three join the
    # gate the moment the deployment is anything but a local loopback dev
    # process (there is no read-only GET in this family to keep open).
    "/api/finetune/",
    "/api/bundles/import",
    "/api/variants/upload",
    "/api/jobs/reconcile",
    "/api/models/install",
    # Writes the chat-template capability record under prompts/models/;
    # the read-only GET /api/models/capabilities stays open.
    "/api/models/capabilities/probe",
    # J-lens lifecycle: /acquire goes ONLINE (the whole point of the verb) and
    # /import writes the workspace lens store. Both are token-gated the moment
    # the deployment is anything but a local loopback dev process. The trailing
    # segment names the mutating verbs exactly, so the read-only catalog
    # (GET /api/jlens/lenses) stays open like every other listing.
    "/api/jlens/lenses/acquire",
    "/api/jlens/lenses/import",
    # Writes a vector artifact into runs/. The read-only token-options verb
    # stays open: it loads a tokenizer, reads no caller-named path, and its
    # whole purpose is to be consulted BEFORE anyone commits to a token.
    "/api/jlens/directions/derive",
    # Reads a caller-named vector artifact and the gain-scaled head (GBs), and
    # writes a readout run. Same containment and same gate as its peers.
    "/api/jlens/support",
    # Stage 4 + the G0 gate: both load the model and generate, so they are GPU
    # compute on a shared node, and both write into the workspace (a lens
    # record's qualifications; a G0 run directory).
    "/api/jlens/qualify",
    "/api/jlens/g0",
    "/api/jlens/probe",
    # Reads and writes a caller-named run directory.
    "/api/jlens/report",
    # Runtime workspace switch: repoints every subsequent read/write at a
    # caller-named directory — a write-anywhere primitive on shared nodes, so
    # it is token-gated exactly like its peers the moment the deployment is
    # anything but a local loopback dev process. The prefix names the switch
    # verb exactly, so the read-only GET /api/workspace stays open.
    "/api/workspace/switch",
    # Reader verbs: /score reads caller-named files and runs model compute;
    # /fit writes files under prompts/ + runs/ and runs model compute. The
    # trailing slash keeps read-only GET /api/readers (catalog listing) open.
    "/api/reader/",
    # Norm backfill reads caller-named artifact/corpus files, runs model
    # compute, and writes under runs/. The exact path keeps the read-only
    # GET /api/vectors catalog listing open.
    "/api/vectors/backfill-norms",
    # Housekeeping (WS3): refresh forces filesystem scans; maintenance writes
    # the calendar the executor's submit-time refusal enforces. The read-only
    # GET /api/housekeeping/status stays open (exact prefixes, not the family).
    "/api/housekeeping/refresh",
    "/api/housekeeping/maintenance",
    # GPU session lifecycle (GPU-SESSION-PLAN Wave 1): start submits a Slurm
    # job, DELETE scancels one, GET runs scheduler polls — the whole family
    # is privileged (the controller runs token mode anyway; this covers a
    # misconfigured one).
    "/api/session",
    # Panel scripts: /save writes a caller-named file under prompts/panels/,
    # and /run reads a caller-named path AND runs model compute — the same
    # combination that put /api/reader/ on this list. The trailing slash is
    # load-bearing: it gates /api/scenario/save and /api/scenario/run while
    # leaving the read-only GET /api/scenario and GET /api/scenarios open,
    # exactly as /api/reader/ leaves GET /api/readers open.
    "/api/scenario/",
    # Direct Gemma Scope feature-ID import: goes ONLINE to Hugging Face (the
    # same reason /api/jlens/lenses/acquire is here), reads a caller-named
    # calibration-donor artifact, and writes a vector artifact under runs/.
    # The exact path keeps the older report-based /api/gemmascope/import and
    # the read-only /api/gemmascope/info exactly as they were.
    "/api/gemmascope/import-id",
)

#: Methods that change server-side state, spend compute, or spend quota.
_MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

# Work Package S (2026-08-18): classification is MUTATING-BY-DEFAULT. Every
# POST/PUT/DELETE/PATCH under /api/ is privileged unless it appears here, so a
# newly added mutating route joins the gate by default and an author who wants
# it open must say so in the diff, with a reason, next to a test that checks
# this list has no stale entries (``test_wp_s_hardening.py``).
#
# Entries are ROUTE TEMPLATES exactly as the router declares them — a
# ``{param}`` segment matches one non-empty path segment. Nothing here may also
# match ``_PRIVILEGED_PREFIXES``: the allowlist can only decline to ADD a gate,
# never remove one (asserted in the same test).
_OPEN_MUTATING_PATHS = (
    # Loads a tokenizer and returns candidate token ids. Reads no caller-named
    # path, writes nothing, runs no model — and its whole purpose is to be
    # consulted BEFORE anyone commits to a token id, which is why its sibling
    # /api/jlens/directions/derive is gated and this is not.
    "/api/jlens/token-options",
    # Tokenizer-only in the other direction: ids -> text. Same reasoning.
    "/api/jlens/decode-tokens",
    # Template render: substitutes into a FIXED template file chosen by an enum
    # key (never a caller-named path) and returns the string. No model, no
    # writes, no state.
    "/api/generation-prompt",
    # Parse-only preview of pasted stimulus text: returns the parsed pairs for
    # the UI to show. The WRITING sibling is /api/concept/{name}/save.
    "/api/concept/import",
    # Parse-only preview of a pasted probe file — returns the parsed items for
    # the UI to show. The WRITING sibling is /api/concept/{name}/probe-items.
    "/api/concept/{name}/probe-import",
)


def _path_matches_template(path: str, template: str) -> bool:
    """Exact segment match, ``{param}`` matching one non-empty segment.

    Deliberately strict about trailing slashes and segment count: anything that
    is not an exact shape match falls through to the gate, so no spelling of a
    URL can talk its way OUT of the privileged set.
    """
    parts = path.split("/")
    wanted = template.split("/")
    if len(parts) != len(wanted):
        return False
    for got, want in zip(parts, wanted):
        if want.startswith("{") and want.endswith("}"):
            if not got:
                return False
        elif got != want:
            return False
    return True


def request_is_privileged(method: str, path: str) -> bool:
    """Whether this request belongs to the token-gated set.

    Pure and exported so the completeness test can classify the real route
    table without driving HTTP.
    """
    if any(path.startswith(prefix) for prefix in _PRIVILEGED_PREFIXES):
        return True
    # The two historical method-aware cases. Both are now subsumed by the
    # mutating-by-default rule below; they stay as the explicit record of WHY
    # these families are privileged, so a future narrowing of the default rule
    # cannot silently reopen them. Experiment verbs launch GPU jobs, and the
    # PUT manifest-sync route writes a caller-supplied manifest into the
    # workspace, while GETs on the family stay open reads.
    if method in ("POST", "PUT") and path.startswith("/api/experiment/"):
        return True
    # Manual resume (POST /api/jobs/{id}/resubmit) runs sbatch on the job's own
    # script — privileged like /api/jobs/reconcile.
    if (method == "POST" and path.startswith("/api/jobs/")
            and path.endswith("/resubmit")):
        return True
    if method in _MUTATING_METHODS and path.startswith("/api/"):
        return not any(_path_matches_template(path, template)
                       for template in _OPEN_MUTATING_PATHS)
    return False


def peer_is_loopback(request: Request) -> bool:
    """Whether the CONNECTING PEER is on this machine.

    ``ServerProfile.bind`` reports what ``STEERLAB_BIND`` *says*; it is a
    declaration, not an observation, and a direct
    ``uvicorn steerlab_server.api.app:app --host 0.0.0.0`` never sets it. The
    peer address is what the socket actually accepted, so every gate that
    means "this machine only" reads THIS, not the configured bind.

    Fail-closed by construction:

    - No peer in the ASGI scope at all (``request.client is None``) counts as
      NON-loopback. That covers unix-domain-socket serving and any ASGI server
      that omits ``client`` — both then need an auth mode declared, which is
      the honest answer for a socket whose other end this process cannot see.
    - A hostname that is not an IP literal is only loopback when it is
      literally in :data:`LOOPBACK_HOSTS` (``localhost``). A real network
      server always puts an ADDRESS here, never a name, so nothing a remote
      peer controls can spell its way into this branch.
    """
    client = request.client
    host = (getattr(client, "host", "") or "").strip().lower() if client else ""
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host in _LOOPBACK
    # ::ffff:127.0.0.1 is 127.0.0.1 arriving on a dual-stack socket, and
    # ipaddress does not call the mapped form loopback on its own.
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def _server_authenticates(profile: ServerProfile) -> bool:
    """Whether this deployment performs ANY authentication of its own.

    ``token`` checks a bearer token below; ``external`` declares that a
    fronting proxy (Open OnDemand, an SSO gateway) has already authenticated
    the caller. ``none`` means every /api route is open to whoever can reach
    the socket — legal on loopback, never off it.
    """
    return profile.auth_mode in ("token", "external")


#: What to tell a caller whose request was refused for arriving from off-box
#: with no auth configured. Names the supported launch path first: ``serve``
#: resolves token mode by default and mints/reads the token file, so the
#: overwhelmingly common cause is a hand-rolled uvicorn line.
_OPEN_INSTRUMENT_REFUSAL = (
    "refusing a non-loopback request: this server has no authentication "
    "configured (STEERLAB_AUTH_MODE=none), so serving it off this machine "
    "would leave every mutating and compute route open to the network. "
    "Start the server with `steerlab-server serve` (token mode is the "
    "default; it reads or mints STEERLAB_AUTH_TOKEN_FILE), or set "
    "STEERLAB_AUTH_MODE=token with STEERLAB_AUTH_TOKEN — or keep the socket "
    "on loopback and reach it through an SSH tunnel."
)


def _host_only(value: str) -> str:
    """Bare hostname from a Host header value (strip port and IPv6 brackets)."""
    v = value.strip()
    if v.startswith("["):  # [::1]:8080 or [::1]
        return v[1:].split("]", 1)[0].lower()
    if v.count(":") == 1:  # host:port
        return v.split(":", 1)[0].lower()
    return v.lower()


def _port_only(value: str) -> str | None:
    """Port from a Host header value, or None when absent."""
    v = value.strip()
    if v.startswith("["):  # [::1]:8080 or [::1]
        rest = v.split("]", 1)[1] if "]" in v else ""
        return rest[1:] or None if rest.startswith(":") else None
    if v.count(":") == 1:
        return v.split(":", 1)[1] or None
    return None


def _effective_port(scheme: str, port: str | None) -> str | None:
    return port or _DEFAULT_PORTS.get(scheme.lower())


def _origin_parts(value: str) -> tuple[str, str, str | None] | None:
    """Return normalized (scheme, host, effective-port) for an Origin value."""
    parsed = urlparse(value)
    if parsed.scheme.lower() not in _DEFAULT_PORTS or not parsed.hostname:
        return None
    return (
        parsed.scheme.lower(),
        parsed.hostname.lower(),
        _effective_port(parsed.scheme, str(parsed.port) if parsed.port else None),
    )


def _allowed_origins() -> set[tuple[str, str, str | None]]:
    """Explicit browser origins allowed for reverse-proxy/OOD deployments.

    Values are comma-separated exact origins, e.g.
    ``https://ondemand.example.edu`` or ``https://ood.example.edu:8443``.
    Wildcards are intentionally unsupported.
    """
    raw = os.environ.get("STEERLAB_ALLOWED_ORIGINS", "")
    out: set[tuple[str, str, str | None]] = set()
    for item in raw.split(","):
        parts = _origin_parts(item.strip())
        if parts is not None:
            out.add(parts)
    return out


def _origin_matches_host(origin: tuple[str, str, str | None], host_header: str) -> bool:
    scheme, origin_host, origin_port = origin
    host = _host_only(host_header)
    host_port = _effective_port(scheme, _port_only(host_header))
    return host == origin_host and host_port == origin_port


# Defined BEFORE auth_middleware, which makes it the INNER middleware
# (Starlette wraps in registration order, last = outermost): every request
# below has already passed the CSRF and token checks, so the controller never
# forwards an unauthenticated request and the worker never counts one as
# activity.
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """GPU-session plumbing (GPU-SESSION-PLAN Wave 1), keyed on role.

    Controller: reverse-proxy the interactive route families to a live,
    discovered worker (streaming — SSE chunks pass through as they arrive).
    With no session, every route falls through and behaves exactly as today.

    Worker: idle-activity accounting. Only the activity allowlist touches the
    timer — the app's ~15 s capability polls must never hold a GPU — and the
    in-flight bracket spans the FULL response stream, so an active generation
    blocks expiry until its last chunk.
    """
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)
    role = server_role()
    if role == "controller":
        proxied = await gpu_session.maybe_proxy(
            request, controller_jobs=_controller_jobs)
        if proxied is not None:
            return proxied
        return await call_next(request)
    if role != "gpu-session" or not gpu_session.is_activity(path):
        return await call_next(request)
    worker = gpu_session.ensure_worker()
    worker.timer.begin()
    try:
        response = await call_next(request)
    except BaseException:
        worker.timer.end()
        raise
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:  # non-streaming response: the work is done
        worker.timer.end()
        return response

    async def counted():
        try:
            async for chunk in body_iterator:
                yield chunk
        finally:
            worker.timer.end()

    response.body_iterator = counted()
    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    profile = ServerProfile.from_env()
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    bind_is_loopback = profile.bind in _LOOPBACK

    # (0) Fail-closed on the OBSERVED peer, before anything else. Every gate
    # below keys on the CONFIGURED bind, which is a claim the process cannot
    # verify: `uvicorn steerlab_server.api.app:app --host 0.0.0.0` never sets
    # STEERLAB_BIND, so `profile.bind` reads 127.0.0.1 and the whole API was
    # open to the network with auth_mode=none. The socket knows better — if
    # the request came from off this machine and this deployment authenticates
    # nothing, refuse it. A request carrying valid auth (token/external mode)
    # is unaffected, and so is every loopback caller.
    if not _server_authenticates(profile) and not peer_is_loopback(request):
        return JSONResponse({"detail": _OPEN_INSTRUMENT_REFUSAL},
                            status_code=403)

    # (1) CSRF / DNS-rebinding defense. Browsers stamp Origin / Sec-Fetch-* on
    # every request; curl, the SwiftUI client, and SSH-tunnel users do not.
    # Only browser-originated requests are constrained to loopback origin/host,
    # so non-browser clients (and the test client) are unaffected.
    is_browser = ("origin" in request.headers) or ("sec-fetch-site" in request.headers)
    if is_browser:
        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site", "").lower()
        if not origin and fetch_site and fetch_site not in {"same-origin", "same-site", "none"}:
            return JSONResponse({"detail": "cross-site browser request refused"}, status_code=403)
        if origin:
            parsed = urlparse(origin)
            origin_host = (parsed.hostname or "").lower()
            origin_parts = _origin_parts(origin)
            allowed_proxy_origin = (
                origin_parts in _allowed_origins()
                and _origin_matches_host(origin_parts, request.headers.get("host", ""))
            ) if origin_parts is not None else False
            if origin_host not in _LOOPBACK and not allowed_proxy_origin:
                return JSONResponse(
                    {"detail": "cross-origin API request refused"}, status_code=403)
            # Same-origin means same port too: a hostile page served from a
            # different local port (another user's process on a shared node)
            # must not pass just because its host is loopback.
            origin_port = _effective_port(parsed.scheme, str(parsed.port) if parsed.port else None)
            host_port = _port_only(request.headers.get("host", ""))
            if origin_host in _LOOPBACK and origin_port != _effective_port(parsed.scheme, host_port):
                return JSONResponse(
                    {"detail": "loopback origin on a different port refused"},
                    status_code=403)
        if bind_is_loopback:
            host = _host_only(request.headers.get("host", ""))
            allowed_hosts = {origin[1] for origin in _allowed_origins()}
            if host and host not in _LOOPBACK and host not in allowed_hosts:
                return JSONResponse({"detail": "non-loopback Host header refused"}, status_code=403)

    # (2) Token auth: always in token mode; also for privileged routes once the
    # deployment is anything other than a local+localhost dev process. The
    # privileged set is mutating-by-default plus the read-side prefixes — see
    # ``request_is_privileged``.
    privileged = request_is_privileged(request.method, path)
    require_token = profile.auth_mode == "token" or (
        privileged
        and (profile.executor == "slurm" or profile.profile != "local" or not bind_is_loopback)
    )
    if require_token:
        token = os.environ.get("STEERLAB_AUTH_TOKEN")
        if not token:
            detail = (
                "STEERLAB_AUTH_TOKEN is not configured"
                if profile.auth_mode == "token"
                else "this endpoint requires STEERLAB_AUTH_TOKEN on non-local deployments"
            )
            return JSONResponse({"detail": detail}, status_code=503)
        header = request.headers.get("authorization", "")
        if not hmac.compare_digest(header, f"Bearer {token}"):
            return JSONResponse({"detail": "missing or invalid bearer token"}, status_code=401)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"service": "steerlab-server", "note": "UI not found; use /api"})
