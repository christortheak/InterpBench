"""The runner HTTP adapter — Phase 2 of the portability program.

A **runner** is an engine serving ``steerlab_server.api`` over HTTP: it
executes what this client authored. :class:`RunnerClient` is the whole surface
the client uses to reach one, and it is deliberately small — six concerns, in
the order a study travels:

===========================  ==================================================
concern                      route(s) it speaks
===========================  ==================================================
capabilities / identity      ``GET  /api/info``, ``GET /api/capabilities``
upload a run bundle          ``POST /api/bundles/upload``
submit an uploaded bundle    ``POST /api/bundles/inspect`` (the pre-check),
                             ``POST /api/studies/submit-bundle``
job get / list / cancel      ``GET  /api/jobs/{id}``, ``GET /api/jobs``,
                             ``POST /api/jobs/{id}/cancel``
job logs                     ``GET  /api/jobs/{id}`` (``logTail``),
                             ``GET  /api/jobs/{id}/stream`` (follow)
evidence bundle download     ``GET  /api/bundles/download?path=…``
===========================  ==================================================

Every route above **already exists**. Nothing here asks the engine for a new
endpoint, a ``v2`` of an old one, or a shape it does not already return: the
payload keys below were transcribed out of ``api/routes.py`` and
``api/submissions.py``, not designed. That is the same discipline Phase 1b
applied to the authoring verbs, and for the same reason — an adapter that
invents a wire format makes the engine's tests stop being the client's tests.

Two integrity checks are the point of this module
--------------------------------------------------

Everything else here is plumbing; these two are why the adapter exists rather
than a `curl` in a shell script.

1. **Upload is verified against the runner's own digest.** The client hashes
   the file it sends; the runner hashes the bytes it received
   (``upload_bundle`` in ``api/routes.py`` streams into a SHA-256 as it
   writes). :meth:`RunnerClient.upload_run_bundle` refuses when the two
   disagree — a truncated or mangled upload must never become a submitted
   study, because everything downstream (the run's manifest snapshot, the
   evidence, the citation) inherits that bundle's identity.

2. **Evidence is verified before the caller can import it.**
   :meth:`RunnerClient.download_bundle` streams to a caller-supplied TEMP path,
   hashing as it goes, and only moves the file to its destination once the
   digest matches what the runner reported. A mismatch leaves the destination
   untouched and the temp file deleted. This is the client half of contract
   ``evidence-outer-hash-pre-extraction`` (§3, Phase 1a's G3): the outer digest
   travels out of band and the recipient checks it BEFORE opening the archive.
   The adapter deliberately stops there — it does not import. Importing is
   ``steerlab bundle import --sha256 <digest>``, a separate, explicit act
   against a named workspace.

Retries and idempotency
-----------------------

**No retry policy in this phase beyond idempotent GETs**, and no automatic
retry at all — the adapter makes one attempt and reports what happened. What a
CALLER may safely retry, stated so nobody has to guess:

* ``info`` / ``capabilities`` / ``job`` / ``jobs`` / ``job_logs`` / ``inspect``
  — plain reads. Retry freely.
* ``upload_run_bundle`` — **idempotent to retry.** Each upload lands in its own
  server-minted staging directory (``make_unique_run_directory``), so a retry
  costs disk and produces a second path, but never a second *effect*. The sha
  check makes a partial first attempt visible rather than silent.
* ``download_bundle`` — **idempotent to retry.** It is a GET, and the write is
  temp-then-verify-then-move.
* ``cancel_job`` — idempotent in effect (cancelling a cancelled job is a
  no-op), though the route answers 502 when ``scancel`` does not confirm; that
  is a real state to report, not a state to retry blindly.
* ``submit_uploaded_bundle`` — **NOT idempotent. Never retry it automatically.**
  Each call creates a job and may spend a scheduler allocation. A timeout on
  submit is genuinely ambiguous (the job may exist), and the repair is to LOOK
  — ``runner jobs`` — not to submit again. That is why the sha pre-check
  happens on a separate, idempotent ``inspect`` call *before* the submit: the
  one thing worth verifying is verified while verifying is still free.

The token
---------

The bearer token is an explicit constructor input and is never written
anywhere. It is not in the envelope, not in a log line, not in an exception
message, not in a ``repr``, and not in a URL (it rides an ``Authorization``
header). :meth:`RunnerClient.scrub` is applied to every message this module
raises, so even an upstream string that happened to contain the token comes
out redacted. The CLI never accepts a bare ``--token`` flag — argv is world
readable in a process listing — only ``$STEERLAB_RUNNER_TOKEN`` or
``--token-file``.

Why httpx and not stdlib ``urllib``
-----------------------------------

``httpx`` is a declared client dependency (``pyproject.toml``
``[project] dependencies``) and it is imported LAZILY, inside the adapter's
methods' module, never by ``client_cli``. Four reasons it earns the line:

1. **It costs the install nothing new.** ``httpx==0.28.1`` and its whole
   closure are already pinned in both committed platform locks, pulled in by
   ``huggingface_hub`` — so ``uv pip compile --extra all`` resolves the same
   package set it resolved before. Pure wheels, no compiler, no CUDA.
2. **The in-process test and the TCP test run the SAME adapter code.**
   ``starlette.testclient.TestClient`` *is* an ``httpx.Client``, so the
   ``http_client`` seam below drives the real ASGI app in one test and a real
   socket in another with no second implementation and no fake. With urllib
   the in-process half would be testing a mock of the adapter.
3. **Streaming both ways, with the hashing in our hands.** A chunked request
   body from a file (uploads are archives, not strings) and ``iter_bytes()``
   on the response (so the download can hash and size-cap as it arrives rather
   than after). ``urllib.request`` gives neither without dropping to
   ``http.client``.
4. **Timeouts and TLS as parameters.** One ``timeout=`` covering connect/read/
   write, and ``verify=`` taking a bool or a CA-bundle path; the urllib
   equivalent is a hand-built ``ssl.SSLContext`` plus a per-call socket
   timeout, which is exactly the sort of security-relevant plumbing a client
   should not be reimplementing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from urllib.parse import quote, urlsplit, urlunsplit

#: Read/connect/write budget for one request, in seconds. Generous: a runner
#: on a cluster login node can be slow to answer while its filesystem is busy,
#: and an adapter that gives up early turns a slow answer into a phantom
#: failure the caller cannot distinguish from a real one.
DEFAULT_TIMEOUT = 60.0

#: How much of an evidence bundle the client will accept by default before
#: refusing. A cap the CALLER can raise, never a silent truncation: the
#: download refuses and deletes the partial rather than handing back a short
#: file that would fail its digest check for a reason nobody could read.
DEFAULT_MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024      # 4 GiB

#: Upload chunk. Matches nothing on the server side on purpose — the route
#: reads ``request.stream()`` and hashes whatever arrives.
UPLOAD_CHUNK_BYTES = 1024 * 1024

#: The header ``POST /api/bundles/upload`` reads the destination filename from.
#: The route accepts a RAW body precisely so the core server needs no
#: ``python-multipart``; this header is how the name travels instead.
FILENAME_HEADER = "X-SteerLab-Filename"

#: What the runner reports as its own identity, for the submit envelope. Read
#: off ``GET /api/info`` (``routes._info_payload``).
IDENTITY_KEYS = ("service", "engineVersion", "root",
                 "rootLooksLikeSourceCheckout")

#: Where an evidence bundle reference hides in a job's ``result``, in the three
#: shapes the engine really writes it — a direct in-process job
#: (``jobs._retain_partial_evidence``), a bundle-execute child record folded in
#: as ``runResult`` (``submissions._read_child_record``), and the reconciler's
#: nested ``result`` (``jobs._continuation_evidence``). Transcribed rather than
#: chosen: a client that knew only one shape would report "no evidence" for a
#: run that produced some.
EVIDENCE_RESULT_PATHS = (("evidenceBundle",),
                         ("runResult", "evidenceBundle"),
                         ("result", "evidenceBundle"))


# --- errors ---------------------------------------------------------------
#
# Three kinds, because they answer to three different envelope states and a
# caller repairs them differently. All carry a `repair_action`: this client's
# whole refusal contract is that a decline names its own fix.


class RunnerError(Exception):
    """Base: anything this adapter refuses or fails at.

    Carries the four fields ``client_cli._envelope_for_exception`` needs to
    build a document without re-deriving them from the message text.
    """

    #: Envelope state (``cli_envelope.STATE_EXIT_CODES``).
    state = "failed"
    #: Envelope ``error.code``.
    code = "runnerFailed"

    def __init__(self, reason: str, *, repair_action: str,
                 code: str | None = None, state: str | None = None,
                 detail: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.repair_action = repair_action
        if code is not None:
            self.code = code
        if state is not None:
            self.state = state
        #: Structured extras worth putting in the envelope's ``result`` — the
        #: two digests of a mismatch, the status of an HTTP error. Never a
        #: credential.
        self.detail = dict(detail or {})


class RunnerRefusal(RunnerError):
    """An INTEGRITY check declined: a digest that did not match, a body over
    the cap, a destination that already exists.

    Nothing broke — the instrument worked. 65, like every other refusal on
    this surface.
    """

    state = "refused"
    code = "runnerRefused"


class RunnerHTTPError(RunnerError):
    """The runner answered, and its answer was not a success.

    The status is the useful part and the ``detail`` body is the runner's own
    sentence; both travel. The state is chosen from the status because 401 and
    500 are not the same kind of problem: one is a token the caller can fix,
    the other is a runner that broke.
    """

    def __init__(self, status: int, detail: str, *, url: str,
                 repair_action: str, code: str, state: str) -> None:
        super().__init__(
            f"runner answered {status}: {detail}" if detail
            else f"runner answered {status}",
            repair_action=repair_action, code=code, state=state,
            detail={"status": status, "url": url, "runnerDetail": detail})
        self.status = status


class RunnerUnreachable(RunnerError):
    """The request never got an answer — connect refused, DNS, TLS, timeout.

    Always a FAILURE, never a refusal: nothing declined anything, the client
    simply could not talk to the runner.
    """

    state = "failed"
    code = "runnerUnreachable"


# --- the address ----------------------------------------------------------


#: The envelope code every URL refusal carries. Its own code, not the generic
#: ``runnerRefused``: the repair is "fix the locator you typed", which is
#: nothing like "the runner declined your request".
URL_REFUSED_CODE = "runnerURLRefused"


def _redacted(parts) -> str:
    """A locator safe to print: same shape, with any userinfo replaced.

    A refusal that echoed the URL back verbatim would put the very credential
    it is refusing into the envelope, the log line, and the shell history —
    which is precisely the leak the refusal exists to prevent.
    """
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    if parts.username or parts.password:
        host = f"<credentials redacted>@{host}"
    return urlunsplit((parts.scheme.lower(), host, parts.path, "", ""))


def normalize_base_url(raw: str) -> str:
    """The ONE spelling of a runner's address, or a typed refusal.

    Every surface downstream — the ``runner`` key of every envelope, the
    ``remote-execution.json`` provenance record, every error sentence, the
    ``repr`` of the client — reproduces whatever string arrived here. Until
    this function existed the constructor accepted any non-empty text, so
    ``https://user:hunter2@runner.example.edu/`` flowed straight into a
    provenance document that is deliberately written once, kept forever, and
    read by anyone who reads the run (external review, 2026-08-24).

    So the boundary NORMALIZES, and refuses what it cannot normalize safely:

    - the scheme must be ``http`` or ``https`` — this adapter speaks HTTP, and
      a ``file:``/``ssh:``/typo locator that reached httpx would fail somewhere
      much less legible;
    - **userinfo is refused**, not stripped. Silently dropping a password
      would send an unauthenticated request that fails in a way nobody can
      trace back to the credential they thought they were using; this surface
      carries its credential in one place, an ``Authorization`` header built
      from ``--token-file`` or ``$STEERLAB_RUNNER_TOKEN``;
    - a **query string** and a **fragment** are refused: a base URL is an
      origin plus an optional proxy prefix, both of these are silently lost
      the moment a path is appended to it, and a caller who put a token in one
      would never learn it was ignored.

    Every refusal here names the CATEGORY and never the value (third-round
    review, 2026-08-24). Echoing the rejected component back is the same leak
    the userinfo rule exists to prevent: ``?token=…`` and ``#token=…`` are
    exactly where a caller puts a secret by mistake, so a message reading "the
    URL carries a query string (``token=hunter2``)" would print the credential
    into the envelope, the log line and the shell history — while warning about
    it. The repair sentences carry the whole explanation instead.

    Returns the canonical form: lowercased scheme and host, port preserved,
    path preserved (a runner behind a proxy prefix is a real deployment) with
    any trailing slash removed.
    """
    text = (raw or "").strip()
    if not text:
        raise RunnerRefusal(
            "no runner URL was supplied", code=URL_REFUSED_CODE,
            repair_action=(
                "--runner http://127.0.0.1:8080  (the port every `serve` "
                "surface defaults to)"))
    # The parse, and the PORT, before anything else reads either. `urlsplit`
    # itself raises on a malformed authority (an unclosed IPv6 bracket), and
    # `parts.port` raises only when the attribute is READ — non-numeric
    # ("http://host:notaport") or out of range ("http://host:99999"). Every
    # branch below reads it, `_redacted` included, so an uncaught ValueError
    # used to escape this boundary as a generic malformed-invocation 64 with
    # the raw locator in its traceback (third-round review, 2026-08-24). It is
    # a typed refusal like every other thing wrong with a locator — and it
    # names the CATEGORY only: a mistyped authority is precisely where a
    # credential ends up.
    try:
        parts = urlsplit(text)
        parts.port                     # noqa: B018 — reading it IS the check
    except ValueError:
        raise RunnerRefusal(
            "the runner URL's port is not a number in 1–65535",
            code=URL_REFUSED_CODE,
            repair_action=(
                "write the authority out as scheme://host:port — e.g. "
                "--runner http://127.0.0.1:8080. Nothing of what you typed is "
                "echoed here, deliberately: a malformed authority is where a "
                "mistyped credential lands, and this refusal is recorded.")
        ) from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise RunnerRefusal(
            f"a runner URL must be http:// or https:// — "
            f"{_redacted(parts)!r} is not",
            code=URL_REFUSED_CODE,
            repair_action=(
                "write the scheme out: --runner http://127.0.0.1:8080, or "
                "--runner https://runner.example.edu:8443. A bare host is "
                "ambiguous and this client will not guess TLS on or off."))
    if parts.username or parts.password:
        raise RunnerRefusal(
            f"the runner URL carries credentials in its userinfo "
            f"({_redacted(parts)}) — this client will not accept a secret in "
            "a locator it records and prints",
            code=URL_REFUSED_CODE,
            repair_action=(
                "drop the user:password@ prefix and present the bearer token "
                "the way this surface carries one: $STEERLAB_RUNNER_TOKEN, or "
                "--token-file <path>. A URL is written into the run's "
                "remote-execution.json provenance record and into every "
                "envelope and diagnostic; a header is not."))
    if not parts.hostname:
        raise RunnerRefusal(
            f"the runner URL names no host: {_redacted(parts)!r}",
            code=URL_REFUSED_CODE,
            repair_action=(
                "--runner http://127.0.0.1:8080 — scheme, host, and port"))
    # The CATEGORY, never the value: `?token=…` is the commonest thing to find
    # here, and a refusal that quoted it would put the secret in the envelope.
    if parts.query:
        raise RunnerRefusal(
            "the runner URL carries a query string — a base URL is an origin "
            "plus an optional proxy prefix, and a query on it is silently "
            "lost the moment a route path is appended",
            code=URL_REFUSED_CODE,
            repair_action=(
                "pass only scheme://host[:port][/prefix]. If that query was "
                "meant as authentication, it is a token, and tokens travel in "
                "the Authorization header this client builds from "
                "--token-file / $STEERLAB_RUNNER_TOKEN. The query itself is "
                "not repeated back here — if it held a secret, treat it as "
                "having been typed into a shell history and rotate it."))
    if parts.fragment:
        raise RunnerRefusal(
            "the runner URL carries a fragment — a fragment never reaches a "
            "server at all",
            code=URL_REFUSED_CODE,
            repair_action=(
                "pass only scheme://host[:port][/prefix]. The fragment is not "
                "repeated back here, for the same reason userinfo is not."))
    host = parts.hostname
    if ":" in host:                    # an IPv6 literal, unbracketed by parse
        host = f"[{host}]"
    if parts.port:
        host = f"{host}:{parts.port}"
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, host, path, "", ""))


# --- the adapter ----------------------------------------------------------


class RunnerClient:
    """One runner, addressed over HTTP.

    Every input is EXPLICIT — there is no ambient configuration, no env-var
    fallback and no config file inside this class. The client CLI resolves the
    token (from ``$STEERLAB_RUNNER_TOKEN`` or ``--token-file``) and hands it
    in; a library caller passes whatever it has. A class that read the
    environment itself would make "which credential did this use?" unanswerable
    from the call site, which is the wrong property for the object that carries
    a bearer token.

    :param base_url: e.g. ``https://runner.example.edu:8080``. NORMALIZED by
        :func:`normalize_base_url` — a trailing slash is tolerated, a path
        component is preserved (a runner behind a proxy prefix is a real
        deployment), and userinfo, a query string or a fragment is a typed
        refusal rather than something this object would go on to print.
    :param token: the bearer token, or ``None`` for an unauthenticated runner
        (a loopback dev process). Never logged, never echoed, never in a URL.
    :param timeout: seconds, applied to connect/read/write alike. Set on the
        ``httpx.Client`` this adapter builds, NOT passed per request: an
        injected client carries its own budget, and re-stating one per call
        would silently override it (``TestClient`` warns about exactly that).
    :param verify: TLS verification policy, passed to httpx: ``True`` (the
        default — verify against the system trust store), a path to a CA
        bundle, or an ``ssl.SSLContext``. ``False`` disables verification and
        is deliberately NOT reachable from the CLI: turning off certificate
        checking should take more than one word.
    :param http_client: an existing ``httpx.Client`` to use instead of building
        one. This is the test seam — ``TestClient`` is an ``httpx.Client``, so
        the adapter runs unchanged against the in-process ASGI app. When
        supplied, the adapter does NOT own it and will not close it.
    :param max_download_bytes: the download cap (see
        :data:`DEFAULT_MAX_DOWNLOAD_BYTES`).
    """

    def __init__(self, *, base_url: str, token: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 verify=True, http_client=None,
                 max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES) -> None:
        # NORMALIZED at the boundary, never merely accepted: `self.base_url`
        # is what every envelope, provenance record and error sentence
        # downstream reproduces, so this is the one place that can guarantee
        # none of them ever carries a credential (:func:`normalize_base_url`).
        self.base_url = normalize_base_url(base_url)
        self.timeout = float(timeout)
        self.verify = verify
        self.max_download_bytes = int(max_download_bytes)
        # Single underscore, and never rendered: see `__repr__` and `scrub`.
        self._token = (token or "").strip() or None
        self._owns_client = http_client is None
        if http_client is None:
            import httpx           # lazy: the light-import contract, §7/§8.6
            http_client = httpx.Client(timeout=self.timeout, verify=verify,
                                       follow_redirects=False)
        self._client = http_client

    # -- identity of the object itself -------------------------------------

    @property
    def has_token(self) -> bool:
        """Whether a token was supplied. A PRESENCE boolean is the only thing
        about the credential that may appear in a document — the envelope
        secret rule."""
        return self._token is not None

    def __repr__(self) -> str:
        # A `repr` that printed the token would put it in every traceback frame
        # and every debugger transcript. This one says the same useful thing.
        return (f"{type(self).__name__}(base_url={self.base_url!r}, "
                f"token={'<present>' if self.has_token else '<absent>'})")

    def scrub(self, text: str) -> str:
        """``text`` with the token replaced, if it somehow appears.

        Belt and braces: nothing in this module puts the token into a message,
        and httpx does not put headers into its exception strings. But the one
        failure mode a secret rule cannot tolerate is the one nobody predicted,
        and the cost of running every raised sentence through here is a string
        scan.
        """
        if self._token and self._token in text:
            return text.replace(self._token, "<token redacted>")
        return text

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RunnerClient":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    # -- the wire ----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self, extra: dict | None = None) -> dict:
        headers = dict(extra or {})
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _detail_of(self, response) -> str:
        """The runner's own sentence for a failed request.

        FastAPI puts it in ``{"detail": …}``; ``detail`` is sometimes a dict
        (the preflight rejection carries ``{"message", "preflight"}``). Both
        are rendered, and a non-JSON body falls back to the text, capped —
        an HTML error page from a misconfigured proxy must not become the
        whole envelope.

        ``read()`` first: on a STREAMING response (the download, the log
        follow) the body has not been consumed yet, and ``.json()`` would
        raise ``ResponseNotRead`` — which would replace the runner's actual
        sentence with an httpx implementation detail on exactly the paths
        where the sentence matters most.
        """
        try:
            response.read()
        except Exception:      # noqa: BLE001 - a body we cannot read is a
            pass               # body we report as empty, never a new error.
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            return self.scrub((response.text or "").strip()[:500])
        if isinstance(body, dict) and "detail" in body:
            detail = body["detail"]
            if isinstance(detail, str):
                return self.scrub(detail.strip()[:1000])
            if isinstance(detail, dict) and isinstance(detail.get("message"),
                                                       str):
                return self.scrub(detail["message"].strip()[:1000])
            return self.scrub(json.dumps(detail, sort_keys=True)[:1000])
        return self.scrub(json.dumps(body, sort_keys=True)[:500]
                          if body is not None else "")

    def _raise_for_status(self, response, *, url: str) -> None:
        if response.status_code < 400:
            return
        status = response.status_code
        detail = self._detail_of(response)
        if status in (401, 403):
            raise RunnerHTTPError(
                status, detail, url=url, code="runnerUnauthorized",
                state="blocked",
                repair_action=(
                    "this runner requires a bearer token this client did not "
                    "present (or presented a stale one). Put the token in "
                    "$STEERLAB_RUNNER_TOKEN, or pass --token-file <path> — "
                    "never on the command line, where argv is readable by "
                    "every process on the machine. The runner writes its "
                    "token to STEERLAB_AUTH_TOKEN_FILE (default "
                    "~/.steerlab-token) on the SERVER; copy it over a channel "
                    "that is not this one."))
        if status == 404:
            raise RunnerHTTPError(
                status, detail, url=url, code="notFound", state="notFound",
                repair_action=(
                    "the runner has no such job or path — `steerlab runner "
                    "jobs --runner <url>` lists what it does have. A job id "
                    "from a runner that has since restarted with a different "
                    "STEERLAB_JOBS_DB is gone, not hidden."))
        if status == 413:
            raise RunnerHTTPError(
                status, detail, url=url, code="runnerRefused", state="refused",
                repair_action=(
                    "the runner refused the body as oversize — raise its "
                    "STEERLAB_MAX_UPLOAD_BYTES, or package a smaller bundle. "
                    "Nothing was staged."))
        if status == 503:
            raise RunnerHTTPError(
                status, detail, url=url, code="runnerUnavailable",
                state="degraded",
                repair_action=(
                    "the runner is up but declined to serve this route — the "
                    "usual cause is token mode with no STEERLAB_AUTH_TOKEN "
                    "configured on the SERVER. Nothing the client can send "
                    "fixes it; the runner needs its token."))
        if status < 500:
            raise RunnerHTTPError(
                status, detail, url=url, code="runnerRefused", state="refused",
                repair_action=(
                    "the runner declined this request — read its reason "
                    "above; it is the engine's own sentence, not this "
                    "client's paraphrase of one."))
        raise RunnerHTTPError(
            status, detail, url=url, code="runnerFailed", state="failed",
            repair_action=(
                "the runner failed internally — its own logs hold the "
                "traceback; nothing was necessarily left undone, so check "
                "`steerlab runner jobs` before resubmitting anything."))

    def _request(self, method: str, path: str, *, params=None, json_body=None,
                 content=None, headers=None):
        url = self._url(path)
        import httpx
        try:
            response = self._client.request(
                method, url, params=params, json=json_body, content=content,
                headers=self._headers(headers))
        except httpx.HTTPError as exc:
            raise RunnerUnreachable(
                self.scrub(f"{type(exc).__name__} reaching {url}: {exc}"),
                repair_action=(
                    f"check that a runner is serving {self.base_url} and that "
                    "this machine can reach it — an SSH tunnel that has "
                    "dropped looks exactly like this. `steerlab runner "
                    f"capabilities --runner {self.base_url}` is the smallest "
                    "request that answers the question.")) from exc
        self._raise_for_status(response, url=url)
        return response

    def _json(self, method: str, path: str, **kwargs) -> dict:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RunnerError(
                self.scrub(f"{self._url(path)} answered a non-JSON body "
                           f"({exc})"),
                repair_action=(
                    "the URL answered, but not as a SteerLab engine — the "
                    "commonest cause is pointing --runner at a proxy, a "
                    "static site, or the wrong port. GET /api/info on a real "
                    "runner returns {\"service\": \"steerlab-server\", …}."),
                code="notARunner", state="blocked") from exc

    # -- capabilities / identity -------------------------------------------

    def info(self) -> dict:
        """``GET /api/info`` — who this runner is and what it is serving.

        Carries the engine's build identity (``engineVersion``: the identical
        string every run config and frozen manifest stamps), the artifact root
        it serves, whether that root looks like a source checkout rather than a
        workspace, and an embedded capability snapshot.
        """
        return self._json("GET", "/api/info")

    def capabilities(self) -> dict:
        """``GET /api/capabilities`` — the capability snapshot on its own.

        The same document ``info()["capabilities"]`` embeds. Kept as its own
        method because it is the cheap poll: a client checking whether a
        runner's substrate still matches a study's pins does not need the
        controller-chain probe ``/api/info`` performs.
        """
        return self._json("GET", "/api/capabilities")

    def identity(self, info: dict | None = None) -> dict:
        """The runner-identity block a submit envelope reports."""
        document = info if info is not None else self.info()
        return {key: document.get(key) for key in IDENTITY_KEYS}

    # -- run bundles -------------------------------------------------------

    def upload_run_bundle(self, path: str) -> dict:
        """``POST /api/bundles/upload`` — stage a bundle on the runner, and
        prove it arrived intact.

        The client hashes the file as it sends it; the route hashes what it
        receives and returns ``sha256``. A disagreement is refused HERE, before
        the path can be handed to a submit — a bundle is a hash-pinned
        artifact, and half of one is not a smaller study, it is a different
        one.

        Returns the route's own document (``path``, ``filename``, ``sha256``,
        ``bytes``, ``bundle`` — the inspected metadata — ``executable``,
        ``stagingDirectory``) with the locally computed digest added as
        ``localSha256``, so a caller keeps both numbers rather than trusting
        that the comparison happened.
        """
        source = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(source):
            raise FileNotFoundError(2, "no such bundle", source)
        filename = os.path.basename(source)
        local = sha256_file(source)
        local_bytes = os.path.getsize(source)

        def chunks():
            with open(source, "rb") as handle:
                while True:
                    block = handle.read(UPLOAD_CHUNK_BYTES)
                    if not block:
                        return
                    yield block

        document = self._json(
            "POST", "/api/bundles/upload", content=chunks(),
            headers={FILENAME_HEADER: filename,
                     "Content-Type": "application/octet-stream"})
        reported = str(document.get("sha256") or "")
        if reported != local:
            raise RunnerRefusal(
                f"the runner received different bytes than were sent: it "
                f"reports sha256 {reported or '<none>'}, this client computed "
                f"{local} for {filename}",
                code="uploadDigestMismatch",
                repair_action=(
                    "do NOT submit the staged path — re-upload. A digest "
                    "disagreement means the archive was truncated or altered "
                    "in flight, and every hash-pinned thing downstream (the "
                    "run's manifest snapshot, its evidence, its citation) "
                    "would inherit the wrong identity. Re-running "
                    "`steerlab runner upload` is safe: each upload lands in "
                    "its own staging directory."),
                detail={"reportedSha256": reported, "localSha256": local,
                        "bundle": filename})
        document["localSha256"] = local
        document["localBytes"] = local_bytes
        return document

    def inspect_remote_bundle(self, remote_path: str) -> dict:
        """``POST /api/bundles/inspect`` — read a staged bundle's metadata and
        recompute its outer digest, on the runner, without executing anything.
        """
        return self._json("POST", "/api/bundles/inspect",
                          json_body={"bundlePath": remote_path})

    def submit_uploaded_bundle(self, *, remote_path: str, verb: str,
                               expected_sha256: str | None = None,
                               executor: str | None = None,
                               target_root: str | None = None,
                               dry_run: bool = False,
                               dtype: str | None = None,
                               device: str | None = None,
                               prompts_path: str | None = None,
                               source_path: str | None = None,
                               package_evidence: bool = True,
                               parallel_jobs: int = 1,
                               force: bool = False,
                               resources: dict | None = None,
                               env: dict | None = None) -> dict:
        """``POST /api/studies/submit-bundle`` — execute a staged bundle.

        **This call is not idempotent.** It creates a job and, on a Slurm
        runner, spends a scheduler allocation. Do not retry it automatically;
        a timeout here means "look at the job list", not "submit again".

        ``expected_sha256``, when given, is checked FIRST against the runner's
        own ``POST /api/bundles/inspect`` of that path. That extra round trip
        is deliberate and cheap: verifying is free while ``inspect`` is
        idempotent, and it stops the one mistake this surface makes easy —
        submitting the path of a *different* staged bundle (an earlier upload,
        another study's) because the two look alike in a shell history.

        Returns ``StudySubmission.to_dict()``: ``jobId``, ``experiment``,
        ``verb``, ``executor``, ``dryRun``, ``runBundle``, ``slurmBundle``,
        ``slurmJobID``, ``command``, ``recordsDirectory``,
        ``submissionDirectory``, ``preflight``, ``shardJobIDs``.
        """
        expected = (expected_sha256 or "").strip().lower()
        if expected:
            staged = self.inspect_remote_bundle(remote_path)
            actual = str(staged.get("bundleSha256") or "")
            if actual != expected:
                raise RunnerRefusal(
                    f"the bundle staged at {remote_path} hashes to "
                    f"{actual or '<none>'}, not the pinned {expected} — "
                    "nothing was submitted",
                    code="bundleDigestMismatch",
                    repair_action=(
                        "submit the path that `steerlab runner upload` "
                        "returned for THIS digest, or re-upload the bundle "
                        "and use the fresh path. Nothing ran, so nothing "
                        "needs undoing."),
                    detail={"expectedSha256": expected,
                            "runnerSha256": actual,
                            "bundlePath": remote_path})

        # Only keys the caller actually set: the route reads `body.get(...)`
        # with its own defaults, and a client that sent `None` for every
        # optional field would be overriding those defaults with nulls.
        body: dict = {"bundlePath": remote_path, "verb": verb,
                      "dryRun": bool(dry_run),
                      "packageEvidence": bool(package_evidence)}
        for key, value in (("executor", executor), ("targetRoot", target_root),
                           ("dtype", dtype), ("device", device),
                           ("promptsPath", prompts_path),
                           ("sourcePath", source_path)):
            if value is not None:
                body[key] = value
        if parallel_jobs and int(parallel_jobs) != 1:
            body["parallelJobs"] = int(parallel_jobs)
        if force:
            body["force"] = True
        if resources:
            body["resources"] = dict(resources)
        if env:
            body["env"] = dict(env)
        return self._json("POST", "/api/studies/submit-bundle", json_body=body)

    # -- jobs --------------------------------------------------------------

    def job(self, job_id: str) -> dict:
        """``GET /api/jobs/{id}`` — one job record.

        ``Job.to_dict()``: ``id``, ``kind``, ``status``, ``createdAt``,
        ``startedAt``, ``finishedAt``, ``result``, ``error``, ``logTail``,
        ``requestedResources``, ``outputArtifacts``, ``executor``,
        ``executorJobID``, ``cancellationRequested``, ``capabilitySnapshot``.
        """
        return self._json("GET", f"/api/jobs/{quote(str(job_id), safe='')}")

    def jobs(self) -> list:
        """``GET /api/jobs`` — every job this runner knows, each with a log
        tail."""
        document = self._json("GET", "/api/jobs")
        listed = document.get("jobs")
        return list(listed) if isinstance(listed, list) else []

    def cancel_job(self, job_id: str) -> dict:
        """``POST /api/jobs/{id}/cancel``.

        A 502 from this route is meaningful and is NOT swallowed: it means
        ``scancel`` did not confirm, so a billed allocation may still be
        running. The engine's own sentence names the Slurm job id to chase.
        """
        return self._json(
            "POST", f"/api/jobs/{quote(str(job_id), safe='')}/cancel")

    def job_logs(self, job_id: str, *, follow: bool = False,
                 max_lines: int = 10_000) -> dict:
        """The job's log lines.

        Two modes, because the engine offers two surfaces and they answer
        different questions:

        * ``follow=False`` (default) — ``GET /api/jobs/{id}``'s ``logTail``.
          A bounded read that returns immediately. The engine caps the tail
          (50 lines by default), so this is a TAIL and says so: the result's
          ``complete`` is False whenever the tail is at the cap.
        * ``follow=True`` — ``GET /api/jobs/{id}/stream``, the SSE surface,
          consumed until the runner closes it. The runner closes when the job
          reaches a terminal status (``prepared`` included — a dry run is
          terminal), emitting ``[<status>]`` as its last line. A job that is
          still running holds the connection open; the read timeout is the
          bound, and a timeout there returns what arrived with
          ``complete: False`` rather than raising, because "no new output for
          a while" is not a failure.

        Returns ``{"jobId", "status", "lines", "lineCount", "complete",
        "followed"}``.
        """
        if not follow:
            record = self.job(job_id)
            lines = [str(line) for line in (record.get("logTail") or [])]
            return {"jobId": record.get("id", job_id),
                    "status": record.get("status"),
                    "lines": lines, "lineCount": len(lines),
                    # The engine's tail limit is 50; at exactly the cap we
                    # cannot tell a full tail from a truncated one, and
                    # claiming completeness is the wrong guess.
                    "complete": len(lines) < 50,
                    "followed": False}

        import httpx
        url = self._url(f"/api/jobs/{quote(str(job_id), safe='')}/stream")
        lines: list = []
        complete = False
        try:
            with self._client.stream("GET", url,
                                     headers=self._headers()) as response:
                self._raise_for_status(response, url=url)
                for raw in response.iter_lines():
                    if not raw.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(raw[len("data:"):].strip())
                    except (ValueError, json.JSONDecodeError):
                        continue
                    lines.append(str(payload.get("line", "")))
                    if len(lines) >= max_lines:
                        break
                else:
                    complete = True
        except httpx.TimeoutException:
            complete = False
        except httpx.HTTPError as exc:
            raise RunnerUnreachable(
                self.scrub(f"{type(exc).__name__} streaming {url}: {exc}"),
                repair_action=(
                    "the log stream dropped — the lines already read are "
                    "above. `steerlab runner logs <id> --runner <url>` "
                    "without --follow reads the tail without holding a "
                    "connection open.")) from exc
        status = None
        if lines and lines[-1].startswith("[") and lines[-1].endswith("]"):
            status = lines[-1][1:-1]
        return {"jobId": str(job_id), "status": status, "lines": lines,
                "lineCount": len(lines), "complete": complete,
                "followed": True}

    # -- evidence ----------------------------------------------------------

    def evidence_reference(self, job_id: str, *,
                           record: dict | None = None) -> dict | None:
        """The ``{bundlePath, bundleSha256, …}`` a finished job points at, or
        ``None``.

        Reads all three shapes the engine really writes (see
        :data:`EVIDENCE_RESULT_PATHS`) rather than the one a reader of any
        single code path would expect.
        """
        job = record if record is not None else self.job(job_id)
        result = job.get("result")
        if not isinstance(result, dict):
            return None
        for path in EVIDENCE_RESULT_PATHS:
            cursor = result
            for key in path:
                cursor = cursor.get(key) if isinstance(cursor, dict) else None
                if cursor is None:
                    break
            if isinstance(cursor, dict) and cursor.get("bundlePath"):
                return cursor
        return None

    def download_bundle(self, *, remote_path: str, expected_sha256: str,
                        destination: str, temp_path: str | None = None,
                        max_bytes: int | None = None) -> dict:
        """``GET /api/bundles/download`` — fetch an archive and verify it
        before it reaches ``destination``.

        The order is the whole point and it is the client half of
        ``evidence-outer-hash-pre-extraction`` (contracts §3, Phase 1a G3):

        1. stream into a temp file — by default a UNIQUE one this method
           creates in the destination's own directory (same filesystem, so the
           final commit is a cheap metadata operation), or ``temp_path`` when
           the caller names one explicitly;
        2. hash and count bytes as the chunks arrive, refusing the moment the
           cap is exceeded rather than after;
        3. compare against ``expected_sha256`` — the digest the RUNNER
           reported out of band, in the job record;
        4. only then commit onto ``destination``, with a primitive that CANNOT
           overwrite (:func:`_commit_no_replace`).

        A mismatch deletes the temp file and leaves ``destination`` untouched
        and non-existent. Nothing is ever extracted here: importing is a
        separate, explicit act against a named workspace
        (``steerlab bundle import --sha256 <digest>``).

        Refuses rather than overwrites when ``destination`` already exists.
        Evidence is immutable; a silent overwrite is how one run's results
        quietly become another's — and that rule now survives the RACE as well
        as the check (external review, 2026-08-24). The existence check happens
        before a download that can take minutes; a second client, or the same
        one run twice, can create the destination inside that window, and
        ``os.replace`` would have silently clobbered it. The temp file is
        unique per call for the same reason: a predictable ``<dest>.partial``
        made two concurrent downloads write the same staging path and hand each
        other's bytes to the digest check.

        ``temp_path``, when named, is CREATED EXCLUSIVELY: an existing file
        there is a ``tempPathExists`` refusal, never a truncation. Naming a
        staging path says where the bytes may wait, not that anything already
        standing there may be destroyed — and "somewhere with room" is exactly
        the kind of path that holds somebody else's large file.

        The staging file is reached by DESCRIPTOR, never by name, from its
        reservation to its commit (external review, 2026-08-26). Reserving a
        name and then reopening it — which both branches did, with
        ``open(temp_path, "wb")`` — is a time-of-check/time-of-use window: in
        it the name can become a symlink, and the reopen then truncates
        whatever it points at. The reviewer overwrote an unrelated file that
        way and still got ``verified: True`` back. A descriptor cannot be
        redirected after the fact, so the bytes go to the reserved inode, the
        digest is taken from that same inode, and the commit publishes that
        inode or refuses.
        """
        expected = (expected_sha256 or "").strip().lower()
        if not expected:
            raise RunnerRefusal(
                "no expected sha256 for the download — this client will not "
                "fetch an evidence bundle it cannot verify",
                code="unverifiableDownload",
                repair_action=(
                    "the digest travels beside the archive, in the job "
                    "record's evidenceBundle.bundleSha256. If the job carries "
                    "none, the runner never packaged evidence for it — check "
                    "`steerlab runner jobs <id>` for the run directory and "
                    "package it there."))
        cap = int(self.max_download_bytes if max_bytes is None else max_bytes)
        destination = os.path.abspath(os.path.expanduser(destination))
        if os.path.exists(destination):
            raise RunnerRefusal(
                f"{destination} already exists",
                code="destinationExists",
                repair_action=(
                    "name a path that does not exist — evidence is immutable, "
                    "and overwriting one bundle with another is how a run's "
                    "results quietly become a different run's. Move the "
                    "existing file aside if you meant to replace it."))
        destination_dir = os.path.dirname(destination) or "."
        os.makedirs(destination_dir, exist_ok=True)
        # Before the reservation, so a missing dependency cannot leave a
        # staging file (and its descriptor) behind.
        import httpx
        if temp_path:
            # An operator who named a staging path owns it — the usual reason
            # is a filesystem with room the destination's does not have. They
            # do NOT thereby authorise destroying whatever is already there:
            # the path was opened "wb" and silently truncated it (third-round
            # review, 2026-08-24), which is the same overwrite the destination
            # rule refuses, one path to the left. The name is RESERVED with an
            # exclusive create, so an existing file is a typed refusal and a
            # racing second download loses the reservation rather than the
            # bytes.
            #
            # The exclusive create is also the LAST time this path is opened:
            # the handle it returns stays open and is what the download writes
            # through. Reserving the name and reopening it would put the whole
            # TOCTOU back — see the method docstring.
            temp_path = os.path.abspath(os.path.expanduser(temp_path))
            os.makedirs(os.path.dirname(temp_path) or ".", exist_ok=True)
            try:
                staged = open(temp_path, "x+b")
            except FileExistsError:
                raise RunnerRefusal(
                    f"{temp_path} already exists — this client will not "
                    "truncate a file it was merely asked to stage through",
                    code="tempPathExists",
                    repair_action=(
                        "name a --temp path that does not exist, or move that "
                        "file aside. Omit --temp entirely and the client mints "
                        "a unique staging file beside the destination, which "
                        "is the path with no such question in it."),
                    detail={"tempPath": temp_path}) from None
        else:
            # ``mkstemp`` hands back the descriptor precisely so the name never
            # has to be opened a second time; it is wrapped, not closed.
            handle_fd, temp_path = tempfile.mkstemp(
                prefix=".steerlab-download-", suffix=".partial",
                dir=destination_dir)
            try:
                staged = os.fdopen(handle_fd, "w+b")
            except BaseException:
                # Same identity rule as every other cleanup here: prove the
                # name still refers to the inode ``mkstemp`` handed back
                # before removing it. Stat FIRST — the descriptor is the only
                # proof, and closing it throws that away.
                try:
                    reserved = os.fstat(handle_fd)
                except OSError:
                    reserved = None
                os.close(handle_fd)
                if reserved is not None:
                    _unlink_if_reserved(temp_path, reserved)
                raise

        url = self._url("/api/bundles/download")
        hasher = hashlib.sha256()
        total = 0
        try:
            with self._client.stream(
                    "GET", url, params={"path": remote_path},
                    headers=self._headers()) as response:
                self._raise_for_status(response, url=url)
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > cap:
                    raise RunnerRefusal(
                        f"the runner declares {int(declared)} bytes, over the "
                        f"{cap}-byte limit — nothing was downloaded",
                        code="downloadTooLarge",
                        repair_action=(
                            "raise the client's limit if the bundle really is "
                            "this big, or fetch it out of band; the archive "
                            "was not touched."),
                        detail={"declaredBytes": int(declared),
                                "limitBytes": cap})
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > cap:
                        raise RunnerRefusal(
                            f"the download passed the {cap}-byte limit — "
                            "stopped mid-stream, nothing kept",
                            code="downloadTooLarge",
                            repair_action=(
                                "raise the client's limit if the bundle "
                                "really is this big, or fetch it out of "
                                "band. A truncated archive would fail its "
                                "digest anyway, which is why this stops "
                                "instead of keeping what arrived."),
                            detail={"limitBytes": cap})
                    hasher.update(chunk)
                    staged.write(chunk)
                staged.flush()
        except httpx.HTTPError as exc:
            _discard_staged(staged, temp_path)
            raise RunnerUnreachable(
                self.scrub(f"{type(exc).__name__} downloading {url}: {exc}"),
                repair_action=(
                    "nothing was written to the destination. Retrying is "
                    "safe: a download is a GET, and this client verifies the "
                    "digest before the file moves anywhere.")) from exc
        except RunnerRefusal as refusal:
            # The cap refusals raised inside the stream. Same rule as the
            # commit's: the document says what cleanup did with the staging
            # name, and cleanup never removes an entry it cannot prove it owns.
            discarded = _discard_staged(staged, temp_path)
            refusal.detail.update(
                _foreign_staging_detail(discarded, temp_path))
            raise
        except BaseException:
            _discard_staged(staged, temp_path)
            raise

        digest = hasher.hexdigest()
        if digest != expected:
            discarded = _discard_staged(staged, temp_path)
            raise RunnerRefusal(
                f"the downloaded archive hashes to {digest}, not the "
                f"{expected} the runner reported — refused before anything "
                f"reached {destination}",
                code="evidenceDigestMismatch",
                repair_action=(
                    "do NOT import this file — it is not the archive the "
                    "runner described. A per-member check could not catch "
                    "this: a wholesale SUBSTITUTED bundle is internally "
                    "consistent, which is exactly what the outer digest is "
                    "for. Re-download; if it disagrees again, the copy on the "
                    "runner is the problem."),
                detail={"expectedSha256": expected, "downloadedSha256": digest,
                        "bytes": total, "destination": destination,
                        **_foreign_staging_detail(discarded, temp_path)})

        # ``hasher`` covers the bytes this client HANDED to the descriptor.
        # This covers the bytes the staging inode actually holds now, read back
        # through that same descriptor — no name, so nothing can substitute a
        # different file for it. The two disagree only when something else got
        # into the inode while the download ran, which an opener who held the
        # staging name before it was swapped away can arrange.
        restaged = _digest_open_file(staged)
        if restaged != expected:
            discarded = _discard_staged(staged, temp_path)
            raise RunnerRefusal(
                f"the staged file hashes to {restaged}, not the {expected} "
                "the arriving bytes hashed to — something else wrote into the "
                f"staging file; refused before anything reached {destination}",
                code="stagedBytesChanged",
                repair_action=(
                    "nothing was kept and the destination was not created. "
                    "Stage somewhere only this client can write: omit --temp "
                    "and the client mints a unique staging file beside the "
                    "destination, which no other process has the name of."),
                detail={"expectedSha256": expected, "stagedSha256": restaged,
                        "tempPath": temp_path, "bytes": total,
                        "destination": destination,
                        **_foreign_staging_detail(discarded, temp_path)})
        try:
            _commit_no_replace(staged, temp_path, destination)
        except FileExistsError:
            discarded = _discard_staged(staged, temp_path)
            raise RunnerRefusal(
                f"{destination} appeared while this download was in flight — "
                "refused rather than overwritten",
                code="destinationExists",
                repair_action=(
                    "another download (or another process) created that path "
                    "between this client's check and its commit. Evidence is "
                    "immutable: name a path that does not exist, or move the "
                    "file that appeared aside. Nothing of this download was "
                    "kept."),
                detail={"destination": destination, "sha256": digest,
                        "bytes": total,
                        **_foreign_staging_detail(discarded, temp_path)}
                ) from None
        except RunnerRefusal as refusal:
            # The commit's own typed refusal — ``stagingPathHijacked``, whose
            # whole finding is that the staging NAME stopped meaning what it
            # did. Its document must say what the cleanup then did with that
            # name, so the person is not left wondering whether the file that
            # took it over is still there. It is: nothing removes an entry it
            # cannot prove it owns.
            discarded = _discard_staged(staged, temp_path)
            refusal.detail.update(
                _foreign_staging_detail(discarded, temp_path))
            raise
        except BaseException:
            _discard_staged(staged, temp_path)
            raise
        staged.close()
        return {"path": destination, "sha256": digest, "bytes": total,
                "remotePath": remote_path, "verified": True,
                # Stated in the document, not only in the docs: this adapter
                # downloads and verifies, and stops.
                "imported": False}


# --- small helpers --------------------------------------------------------


def sha256_file(path: str, *, chunk: int = 1024 * 1024) -> str:
    """SHA-256 of a file, streamed.

    A local twin of ``experiment.bundles.sha256_file`` — deliberately NOT an
    import of it: that module reaches the archive machinery, and this adapter
    must stay reachable from a client that has only httpx installed.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _digest_open_file(handle, *, chunk: int = 1024 * 1024) -> str:
    """SHA-256 of an ALREADY-OPEN file, read through the caller's descriptor.

    The descriptor is the whole point: :func:`sha256_file` takes a path, and a
    path can name a different file by the time it is opened. This one hashes
    the inode the caller is holding and cannot be redirected. Leaves the
    handle positioned at end-of-file.
    """
    handle.flush()
    handle.seek(0)
    hasher = hashlib.sha256()
    while True:
        block = handle.read(chunk)
        if not block:
            break
        hasher.update(block)
    return hasher.hexdigest()


#: What a refusal says when cleanup found a STRANGER on the staging path.
#: The user needs to know two things: their own file is still there, and this
#: client did not touch it.
FOREIGN_STAGING_NOTE = (
    "the staging path is now occupied by a file this download did not create "
    "— it was left exactly as it was found, bytes and directory entry both")


def _foreign_staging_detail(discarded: bool, temp_path: str) -> dict:
    """The extra ``detail`` keys a refusal carries when :func:`_discard_staged`
    removed NOTHING because a stranger had taken the staging name.

    Empty on the ordinary path, so every refusal's document keeps the exact
    shape it had before this check existed.
    """
    if discarded:
        return {}
    return {"stagingPathForeignEntry": temp_path,
            "stagingPathNote": FOREIGN_STAGING_NOTE}


def _unlink_if_reserved(path: str, reserved) -> bool:
    """Unlink ``path`` ONLY while it still names ``reserved``.

    THE one identity check every cleanup in this adapter goes through
    (external review round 5). ``reserved`` is an ``os.stat_result`` taken
    from a descriptor this client is holding, or from the entry this call
    itself created with ``O_EXCL`` — an inode this code has a right to remove.
    ``lstat`` of the NAME is compared against it, and a mismatch means the
    directory entry no longer points at that inode: somebody replaced it.

    ``lstat``, not ``stat``: a symlink dropped onto the name has its own
    inode, so it can never match, and following it would delete whatever it
    aims at.

    Returns True when the entry was removed (or was already gone — the same
    outcome the caller wanted), False when the name refers to something else
    and was therefore LEFT ALONE.

    RESIDUAL: the name can still be swapped between the ``lstat`` and the
    ``os.remove``. Closing that would need an unlink-this-inode primitive
    POSIX does not offer; what this removes is the whole class of cases where
    the swap is already visible — which is every case a refusal has just
    DETECTED, and the one the review reproduced.
    """
    try:
        present = os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if ((present.st_dev, present.st_ino)
            != (reserved.st_dev, reserved.st_ino)):
        return False
    try:
        os.remove(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _discard_staged(handle, temp_path: str) -> bool:
    """Close the staging descriptor and drop its name, on any path that keeps
    nothing.

    THE INVARIANT (external review round 5): the staging NAME is unlinked only
    while it still refers to the inode this client reserved and wrote through
    — proved, by ``lstat`` of the name against ``fstat`` of the descriptor,
    never assumed.

    It used to be assumed, on the strength of "the entry is the one this client
    reserved". That premise is exactly what fails on the paths this function
    serves: ``stagedBytesChanged`` and ``stagingPathHijacked`` fire BECAUSE the
    name no longer means what it did, and the reviewer's repro moved an
    unrelated regular file onto the staging path mid-download — the refusal
    fired correctly and the cleanup then deleted the replacement, which for a
    file with no other link is the download client destroying somebody's data
    while declining to overwrite anything.

    The descriptor is closed first either way, so on the ordinary path the
    inode still goes away with its last link.

    Returns True when the staging entry was removed (or was already gone),
    False when a foreign entry now occupies the staging path and was left
    untouched — which every caller on an error path says out loud, via
    :func:`_foreign_staging_detail`.
    """
    try:
        reserved = os.fstat(handle.fileno())
    except (OSError, ValueError):
        # No descriptor left to prove identity with: refuse to guess, and
        # leave the name alone. A leftover partial is recoverable; a deleted
        # stranger is not.
        reserved = None
    try:
        handle.close()
    except OSError:
        pass
    if reserved is None:
        return False
    return _unlink_if_reserved(temp_path, reserved)


def _commit_no_replace(staged, temp_path: str, destination: str) -> None:
    """Move a VERIFIED download onto ``destination``, incapable of overwriting.

    ``os.link`` + ``os.remove``, not ``os.replace``. Replace clobbers, and the
    destination-exists check upstream runs BEFORE a download that may take
    minutes — so between the check and the move there is a window in which a
    second client, or the same one run twice, can create the file. ``link``
    raises ``FileExistsError`` in exactly that case, which lets the caller turn
    the race into the same refusal the up-front check gives instead of a silent
    overwrite of somebody's evidence.

    ``open(destination, "x")`` would reserve the name just as atomically, but
    it reserves an EMPTY file that then has to be filled by copying every byte
    a second time; ``link`` commits the bytes that are already on the disk and
    is a metadata operation. It is therefore the primary path — and the O_EXCL
    reservation is the fallback for the few filesystems without hardlinks
    (FAT/exFAT, some network mounts), where it is slower and just as unable to
    overwrite anything.

    RESIDUAL, on the fallback path only (third-round review, 2026-08-24): the
    O_EXCL reservation makes ``destination`` VISIBLE — empty, then partially
    written — for as long as the copy takes. Nothing can overwrite it and a
    failed copy removes it again, but a concurrent reader on such a filesystem
    can observe a short file under the final name. There is no way to close
    that window without a rename, and a rename is precisely the primitive that
    would clobber; the hardlink path (every filesystem this client actually
    runs on) has no window at all. A reader who must be certain has the outer
    digest to check, which is the same instrument this whole method exists for.

    ``staged`` is the OPEN descriptor the download wrote and verified through,
    and it is what makes the publish honest (external review, 2026-08-26).
    ``os.link`` can only take a NAME, so linking is still the one step where a
    swapped staging path could put a different inode under the destination —
    the fd's bytes verified, somebody else's bytes published. The link is
    therefore checked, not trusted: ``fstat`` of the descriptor against
    ``stat`` of what landed. They agree and the destination is exactly the
    inode that was verified; they disagree and the fresh link comes straight
    back out (unlinking a name this call created, never the file that owns
    it) as a ``stagingPathHijacked`` refusal. The destination side needs no
    such check — ``link`` cannot overwrite. The fallback path copies from
    ``staged`` itself, so there the published bytes are the verified inode's
    by construction.

    BOTH-OR-NEITHER (review round 10, finding 8, applied to all three mirrors
    of this primitive): ``destination`` is this call's reservation, and a
    commit that cannot FINISH must not leave it standing. Dropping the staging
    name is the last step; here it goes through ``_unlink_if_reserved``, which
    already reports a stranger on the name by RETURNING False rather than
    raising — deliberately, per the note at the call — so this mirror was
    never able to strand a landed destination the way its twins could. The
    guard is written anyway, and takes the destination back out if that last
    step raises at all (an interrupt is the only way left): three mirrors of
    one primitive hold one invariant, visibly, or the next reader has to
    re-derive which of them is the safe one.

    Raises ``FileExistsError`` when ``destination`` exists; leaves
    ``temp_path`` in place for the caller to clean up when it does.
    """
    reserved = os.fstat(staged.fileno())
    # The inode this call PUBLISHED under the destination name — the staged
    # one on the hardlink path, the freshly reserved one on the copy path.
    # The both-or-neither guard at the end removes the destination only while
    # it still names this, never a stranger that arrived afterwards.
    published_id = reserved
    try:
        os.link(temp_path, destination)
    except FileExistsError:
        raise
    except OSError:
        handle_fd = os.open(destination,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        # The inode THIS call reserved under the destination name, so the
        # cleanup below can prove it is removing its own reservation and not
        # something that arrived on the name afterwards
        # (:func:`_unlink_if_reserved`).
        landing_id = os.fstat(handle_fd)
        published_id = landing_id
        # From here the destination NAME exists, so every exit from this block
        # is accounted for in a `finally`: a copy that died half way (or an
        # `os.fdopen` that never handed the descriptor over) must not leave a
        # short file — or an empty one — wearing the destination's name. The
        # reservation comes back out unless the copy completed.
        landed = False
        try:
            try:
                landing = os.fdopen(handle_fd, "wb")
            except BaseException:
                os.close(handle_fd)
                raise
            # Read the staged bytes back through the descriptor that wrote
            # them, not through the staging name.
            staged.seek(0)
            with landing:
                shutil.copyfileobj(staged, landing)
            landed = True
        finally:
            if not landed:
                _unlink_if_reserved(destination, landing_id)
    else:
        # ``lstat``, not ``stat``: a symlink dropped onto the destination has
        # its own inode, so it can never pass for the verified one.
        published = os.lstat(destination)
        if ((published.st_dev, published.st_ino)
                != (reserved.st_dev, reserved.st_ino)):
            # Remove the link this call just made — proved to still BE that
            # entry, never merely named like it.
            _unlink_if_reserved(destination, published)
            raise RunnerRefusal(
                f"the staging path {temp_path} no longer names the file this "
                f"download verified — {destination} was not published",
                code="stagingPathHijacked",
                repair_action=(
                    "something replaced the staging path between the write "
                    "and the commit, so the name pointed at a different file "
                    "than the one whose digest matched. Nothing of this "
                    "download was kept and the destination was removed again. "
                    "Stage somewhere only this client can write — omit --temp "
                    "and it mints a unique staging file beside the "
                    "destination."),
                detail={"tempPath": temp_path, "destination": destination})
    # Dropping the staging NAME. It removes a directory entry, never content
    # reachable elsewhere — the bytes survive under the destination's link,
    # which is the point — but it removes that entry only while the name still
    # refers to the inode this call verified. The hardlink branch above has
    # just proved that (a swap there is the hijack refusal); the O_EXCL branch
    # never looked at ``temp_path`` at all, since it copies from the
    # descriptor, so a stranger sitting on the staging name would have been
    # deleted here on an otherwise SUCCESSFUL download. One helper, both
    # branches. A stranger is left in place: a leftover partial is
    # recoverable, somebody else's deleted file is not.
    try:
        _unlink_if_reserved(temp_path, reserved)
    except BaseException:
        # The destination is this call's reservation and the commit did not
        # finish: it comes back out (only while it still names the inode this
        # call published) before the failure propagates.
        _unlink_if_reserved(destination, published_id)
        raise
