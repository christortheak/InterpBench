"""The rendered controller script's provenance, read from the node it runs on.

**The gap this closes** (open-issues §1 field report, 2026-08-20). The serverd
self-chain — the USR1 trap and ``chain_successor`` that queue a successor ten
minutes before walltime — lives in ``Server/scripts/controller-job.sbatch
.template``. It is TEMPLATE-side: it is shell code in the job script, not
anything the Python server does. A deploy rsyncs the template; it does not
touch ``$STEERLAB_METADATA_ROOT/controller-job.sbatch``, the RENDERED copy the
operator's standing ritual (``sbatch ~/.steerlab/controller-job.sbatch``)
actually submits. So a node can run brand-new code under a three-day-old
launching script, and did: serverd 47564632 served 52a7176 for its entire
24:00:21 and left no successor.

**What this module knows.** The render now writes one machine-readable line
into the artifact naming the template bytes it came from, and exports a chain
marker for the server it launches:

.. code-block:: text

    # steerlab-render-stamp: template=controller-job.sbatch.template sha256=<hex> renderedAt=<iso> source=<rev>
    export STEERLAB_CONTROLLER_CHAIN="template-<hex>"

``<hex>`` is the SHA-256 of the TEMPLATE FILE'S BYTES, placeholders included —
an invariant any reader can recompute from the deployed template with no render
step of its own. This module recomputes it, compares, and reports; the SWIFT
renderer (``ClusterProvisioner.controllerRemoteCommand``) is the only thing
that writes it, and the two are held together by the fixture-and-assert pair in
``Server/tests/test_controller_render_stamp.py`` and
``Tests/ExperimentKitTests/ClusterProvisionerTests.swift``.

Nothing here writes anything. The one repair is a single command, spelled once
in :data:`RERENDER_COMMAND` and quoted verbatim by every surface that reports a
problem — the ledger, ``site qualify``'s repair text, the serverd boot warning,
and the runbook.
"""

from __future__ import annotations

import hashlib
import os

#: The stamp line's fixed prefix, byte-identical to Swift's
#: ``ClusterProvisioner.renderStampMarker``.
STAMP_MARKER = "# steerlab-render-stamp:"

#: The env var the RENDERED script exports for the server it launches. Its
#: ABSENCE under the controller topology is the cheapest honest signal that the
#: launching script predates the chain.
CHAIN_ENV_VAR = "STEERLAB_CONTROLLER_CHAIN"

#: The chain marker's value prefix; the rest is the template digest.
CHAIN_VALUE_PREFIX = "template-"

#: The template placeholders a render substitutes. Finding one of these still
#: standing means "not rendered" (or hand-rendered), never a real identity.
PLACEHOLDERS = ("@TEMPLATE_SHA256@", "@RENDERED_AT@", "@SOURCE_REVISION@")

#: The rendered artifact's file name inside the metadata root.
RENDERED_NAME = "controller-job.sbatch"

#: The template's path relative to a code tree root, in resolution order:
#: a checkout / cluster payload keeps ``Server/scripts/`` beside the package;
#: a payload flattened to the package root keeps ``scripts/``.
TEMPLATE_RELATIVE = (
    os.path.join("Server", "scripts", "controller-job.sbatch.template"),
    os.path.join("scripts", "controller-job.sbatch.template"),
)

#: THE re-render command. One string, named identically by the ledger, the
#: qualify repair, the boot warning, and ``docs/UGA-CLUSTER-RUNBOOK.md``.
#:
#: It is Swift-side because the RENDERER is Swift-side: the ``#SBATCH`` block
#: is composed from the site profile, and the site profile lives on the Mac.
#: Inventing a second, cluster-local renderer that reconstructed the header
#: from the stale file would be exactly the drift this whole item is about.
RERENDER_COMMAND = (
    "steerlab-cli cluster controller start --site <site> --render-only")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.dirname(_HERE)
_CODE_ROOT = os.path.dirname(_SERVER_DIR)


def template_path(root: str | None = None) -> str | None:
    """The deployed controller-job template, or ``None`` when this deployment
    carries no template at all (a payload that shipped without ``scripts/``)."""
    base = root or _CODE_ROOT
    for relative in TEMPLATE_RELATIVE:
        candidate = os.path.join(base, relative)
        if os.path.isfile(candidate):
            return candidate
    return None


def sha256_file(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def rendered_path(metadata_root: str | None = None) -> str:
    """Where the rendered script lives. ``STEERLAB_METADATA_ROOT`` is the
    authority; the template's own default (``$HOME/.steerlab``) is the
    fallback, because a hand-launched controller may not export it."""
    root = metadata_root or os.environ.get("STEERLAB_METADATA_ROOT") \
        or os.path.join(os.path.expanduser("~"), ".steerlab")
    return os.path.join(root, RENDERED_NAME)


def parse_stamp(text: str) -> dict:
    """The stamp line's ``key=value`` pairs, from a rendered script's TEXT.

    Returns ``{}`` when the file carries no stamp line at all, and drops any
    field whose value is still a placeholder — an unresolved placeholder is the
    absence of a fact, and must never be reported as one.
    """
    for line in text.splitlines():
        if not line.startswith(STAMP_MARKER):
            continue
        fields = {}
        for token in line[len(STAMP_MARKER):].strip().split():
            key, separator, value = token.partition("=")
            if separator and value and value not in PLACEHOLDERS:
                fields[key] = value
        return fields
    return {}


def parse_chain_marker(text: str) -> str | None:
    """The exported chain marker's value, or ``None``.

    An unresolved placeholder counts as absent for the same reason as above.
    """
    needle = f'export {CHAIN_ENV_VAR}="'
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(needle) or not stripped.endswith('"'):
            continue
        value = stripped[len(needle):-1]
        if any(placeholder in value for placeholder in PLACEHOLDERS):
            return None
        return value or None
    return None


def inspect_rendered_script(*, metadata_root: str | None = None,
                            root: str | None = None) -> dict:
    """Compare the rendered controller script against the deployed template.

    The answer is a dict, not a verdict — the two callers (``site qualify``'s
    check row and any operator-facing report) each decide what a status means
    in their own vocabulary. ``status`` is one of:

    ``current``
        the artifact's stamp names the deployed template's exact bytes;
    ``stale``
        it names different bytes, or carries no stamp at all (the pre-chain
        era, whose rendered copies silently dropped the self-chain);
    ``absent``
        nothing is rendered here yet;
    ``unknown``
        the template or the artifact could not be read, so nothing is proven.
        Deliberately NOT ``stale``: crying stale over an unreadable file
        teaches an operator to ignore the finding.
    """
    script = rendered_path(metadata_root)
    template = template_path(root)
    template_sha = sha256_file(template) if template else None
    payload = {
        "status": "unknown",
        "renderedPath": script,
        "templatePath": template,
        "templateSha256": template_sha,
        "stampedSha256": None,
        "renderedAt": None,
        "sourceRevision": None,
        "chainMarker": None,
        "detail": "",
        "rerenderCommand": RERENDER_COMMAND,
    }
    if not os.path.isfile(script):
        payload["status"] = "absent"
        payload["detail"] = (
            f"no rendered controller script at {script} — this node has never "
            "had one rendered, or its metadata root is elsewhere")
        return payload
    try:
        with open(script, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:      # noqa: BLE001 - unreadable artifact
        payload["detail"] = f"{script} could not be read ({exc})"
        return payload

    fields = parse_stamp(text)
    payload["stampedSha256"] = fields.get("sha256")
    payload["renderedAt"] = fields.get("renderedAt")
    payload["sourceRevision"] = fields.get("source")
    payload["chainMarker"] = parse_chain_marker(text)

    if not payload["stampedSha256"]:
        payload["status"] = "stale"
        payload["detail"] = (
            f"{script} carries no render stamp — it predates render "
            "provenance, which is the era whose rendered copies silently "
            "dropped the serverd self-chain")
        return payload
    if not template_sha:
        payload["detail"] = (
            "this deployment carries no readable controller-job template, so "
            f"the artifact's stamp ({payload['stampedSha256'][:12]}) cannot be "
            "checked against anything")
        return payload
    if payload["stampedSha256"] != template_sha:
        payload["status"] = "stale"
        payload["detail"] = (
            f"rendered from template {payload['stampedSha256'][:12]} on "
            f"{payload['renderedAt'] or 'an unrecorded date'}; the template "
            f"deployed here is now {template_sha[:12]}")
        return payload
    if not payload["chainMarker"]:
        payload["status"] = "stale"
        payload["detail"] = (
            f"the rendered script matches the deployed template "
            f"({template_sha[:12]}) but exports no {CHAIN_ENV_VAR} marker")
        return payload
    payload["status"] = "current"
    payload["detail"] = (
        f"rendered {payload['renderedAt'] or 'at an unrecorded time'} from the "
        f"template deployed here ({template_sha[:12]}), source "
        f"{payload['sourceRevision'] or 'unrecorded'}")
    return payload


# =============================================================================
# The boot-time signal
# =============================================================================


def chain_marker_from_environment(environ=None) -> str | None:
    """The chain marker THIS process was launched with, or ``None``."""
    env = os.environ if environ is None else environ
    raw = (env.get(CHAIN_ENV_VAR) or "").strip()
    if not raw or any(placeholder in raw for placeholder in PLACEHOLDERS):
        return None
    return raw


def is_controller_topology(profile) -> bool:
    """True only for a daemon-in-a-job controller on a Slurm site — i.e. only
    for a server the controller-job script itself launched.

    All three halves matter, and the narrowest is ``launch_topology == "batch"``:
    ``server_role`` also DERIVES "controller" from profile=cluster +
    executor=slurm, which is true of a LOGIN-NODE daemon (topology 1) that has
    no controller-job script and therefore no chain to miss. Warning there
    would be a false positive, and a false positive is how an operator learns
    to skip the warning. A workstation, a local executor, and a GPU-session
    worker are excluded for the same reason — which is what keeps this change
    silent for every non-Slurm/local serve.
    """
    from .api.profile import server_role
    return (getattr(profile, "launch_topology", None) == "batch"
            and profile.executor == "slurm"
            and server_role(profile) == "controller")


def boot_chain_status(profile, environ=None) -> dict:
    """What serverd knows at boot about its own launching script.

    ``state`` is one of ``notApplicable`` (not a Slurm controller — nothing to
    say), ``chained`` (the launching script exported a marker), ``mismatched``
    (it exported one naming a DIFFERENT template than the one deployed here),
    or ``unchained`` (it exported none: launched from a pre-chain rendered
    script, e.g. by the standing ``sbatch ~/.steerlab/controller-job.sbatch``
    ritual over a copy that predates the fix).

    A WARNING, never a refusal. A chain-less controller works perfectly well;
    it simply has to be cycled by hand before each walltime, and the operator
    needs to know that at boot rather than from a missing successor 24 h later.
    """
    if not is_controller_topology(profile):
        return {"state": "notApplicable", "marker": None,
                "templateSha256": None, "warnings": []}
    marker = chain_marker_from_environment(environ)
    template = template_path()
    template_sha = sha256_file(template) if template else None
    status = {"state": "chained", "marker": marker,
              "templateSha256": template_sha, "warnings": []}
    if marker is None:
        status["state"] = "unchained"
        status["warnings"] = [
            "this controller was launched by a script that carries NO "
            f"{CHAIN_ENV_VAR} marker, which means a PRE-CHAIN rendered copy "
            "of the controller-job template.",
            "Consequence: no successor job will be queued before this "
            "allocation's walltime. serverd will simply stop, and anything "
            "in flight waits for a manual restart (this is exactly what "
            "happened to job 47564632 on 2026-08-20).",
            "It is safe to keep serving — cycle the controller by hand before "
            "the walltime, and re-render the launching script so the next "
            f"generation self-chains: {RERENDER_COMMAND}",
        ]
        return status
    stamped = marker[len(CHAIN_VALUE_PREFIX):] \
        if marker.startswith(CHAIN_VALUE_PREFIX) else marker
    if template_sha and stamped != template_sha:
        status["state"] = "mismatched"
        status["warnings"] = [
            f"this controller's launching script names template "
            f"{stamped[:12]}, but the template deployed here is "
            f"{template_sha[:12]} — the script is a rendered copy of an OLDER "
            "template.",
            "The chain it carries is that older template's; whether it matches "
            "the fix you deployed cannot be assumed.",
            f"Re-render before the next controller cycle: {RERENDER_COMMAND}",
        ]
    return status


def boot_warning_lines(profile, environ=None) -> list[str]:
    """The loud stderr block, or an empty list when there is nothing to say."""
    status = boot_chain_status(profile, environ)
    if not status["warnings"]:
        return []
    lines = ["=" * 60,
             "WARNING: controller self-chain — this daemon cannot resubmit "
             "itself"]
    lines += [f"  {line}" for line in status["warnings"]]
    lines.append("=" * 60)
    return lines


__all__ = ["CHAIN_ENV_VAR", "CHAIN_VALUE_PREFIX", "PLACEHOLDERS",
           "RENDERED_NAME", "RERENDER_COMMAND", "STAMP_MARKER",
           "TEMPLATE_RELATIVE", "boot_chain_status", "boot_warning_lines",
           "chain_marker_from_environment", "inspect_rendered_script",
           "is_controller_topology", "parse_chain_marker", "parse_stamp",
           "rendered_path", "sha256_file", "template_path"]
