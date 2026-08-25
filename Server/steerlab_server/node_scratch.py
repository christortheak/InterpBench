"""How this site cleans up node-local scratch — ONE definition, two renderers.

**The gap this closes** (ledger 2026-08-23, operator complaint 2026-08-22).
The node-scratch contract — request the space as a gres, remove the staged
directory in an EXIT trap — landed 2026-08-19 inside
``executors.render_slurm_script``. Everything that goes through ``study
submit`` therefore cleans up after itself, and everything that does NOT gets
neither half, silently. Hand-rolled sbatch is not an abuse: continuing a
parked run and chaining behind another job were, until now, genuinely easier
to write by hand. Those two reasons are closed elsewhere (``study submit
--resume`` / ``--dependency``); what remains is the residue of legitimate
ad-hoc work, and it needs a wrapper that starts from the site's own rules.

The rule is stated ONCE, here, and read by both renderers:

- :func:`cleanup_lines` is the trap. ``executors.render_slurm_script`` emits
  it into every study job; :func:`render_wrapper` emits the same lines into
  the canonical ad-hoc wrapper. A second copy is the thing this module exists
  to prevent — "how this site cleans up" that can drift between the studies
  that matter and the one-offs that leak.
- The path removed is the one the SITE PROFILE names
  (``constraints.storage.nodeStageDirTemplate``, reaching the job as
  ``STEERLAB_NODE_STAGE_DIR``), expanded on the node. Never a hardcoded
  ``/lscratch/$SLURM_JOB_ID``: that is one site's spelling, and a site that
  spells it ``$SLURM_TMPDIR`` used to get no cleanup at all while the render
  looked correct.
- The gres request is the site's (``constraints.storage.nodeScratchGres`` →
  ``STEERLAB_SLURM_SCRATCH_GRES``).
- A site whose scheduler purges node scratch itself declares
  ``constraints.storage.nodeScratchPurgedByScheduler`` (→
  :data:`SCHEDULER_PURGES_ENV`) and gets NO trap — an epilog that has already
  removed the directory does not need a job racing it.

Nothing here decides policy; it renders text. The safety argument lives in
:func:`cleanup_lines`.
"""

from __future__ import annotations

import os
import shlex

#: The job's node-local staging directory template, as the job sees it. Its
#: value is the site profile's ``nodeStageDirTemplate``, carried VERBATIM (the
#: env file single-quotes it) so ``$SLURM_JOB_ID`` and friends expand on the
#: compute node and never on the submitting host.
STAGE_DIR_ENV = "STEERLAB_NODE_STAGE_DIR"

#: The site's node-local scratch gres token (``nodeScratchGres``).
SCRATCH_GRES_ENV = "STEERLAB_SLURM_SCRATCH_GRES"

#: Set by a site that declares its SCHEDULER removes node scratch at job end
#: (a Slurm epilog). Truthy → render no trap. Absent → today's behaviour, so a
#: site that has never declared it renders byte-identically to before.
SCHEDULER_PURGES_ENV = "STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER"

#: The environment variables whose presence in a stage-dir template makes that
#: template JOB-SCOPED BY CONSTRUCTION — i.e. two jobs on the same node cannot
#: name the same directory through it.
#:
#: This set IS the safety argument for ``rm -rf``. A template outside it may
#: perfectly well be a shared node cache (``/lscratch/steerlab-models``), which
#: is emphatically not this job's to delete, so an unrecognised template is
#: cleaned up by nobody rather than by everybody.
#:
#: ``SLURM_TMPDIR`` earns its place for the same reason as ``SLURM_JOB_ID``,
#: not by analogy: Slurm allocates it per job and points it at a per-job path.
#: Adding a name here is a deliberate act — it authorises a recursive delete.
JOB_SCOPING_VARIABLES: tuple[str, ...] = (
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_TMPDIR",
)

#: Variables the trap expands that are NOT job-scoping on their own, but
#: routinely appear beside one (``/local_scratch/$USER/$SLURM_JOB_ID``). Listed
#: separately so nobody reads this list as an authorisation to delete.
COMPANION_VARIABLES: tuple[str, ...] = ("USER",)


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def scheduler_purges_node_scratch(environ=None) -> bool:
    """True when the SITE declares its scheduler reclaims node scratch."""
    env = os.environ if environ is None else environ
    return _truthy(env.get(SCHEDULER_PURGES_ENV))


def _variable_case(name: str, *, scoping: bool) -> list[str]:
    """Expand ONE authorised variable, refusing outright if it is unset.

    Both spellings (``$X`` and ``${X}``) are handled: they are identical to
    the shell, a site profile may use either, and a ``case`` pattern is
    literal text that would silently miss the other one.

    The unset check is the safety property this function exists for. Without
    it, a template of ``$SLURM_TMPDIR/models`` on a node where Slurm set no
    ``SLURM_TMPDIR`` expands to ``/models`` — an absolute path with nothing
    left unexpanded, which passes every downstream sanity check and is
    somebody else's directory. A variable the template names and the node has
    not set means the path is UNKNOWN, and an unknown path is never removed.
    """
    return [
        f'  case "${{stage_dir}}" in',
        f"    *'${name}'*|*'${{{name}}}'*)",
        f'      [ -n "${{{name}:-}}" ] || return 0',
        f'      stage_dir="${{stage_dir//\\${name}/${{{name}}}}}"',
        f'      stage_dir="${{stage_dir//\\${{{name}\\}}/${{{name}}}}}"',
        *(["      scoped=1"] if scoping else []),
        "      ;;",
        "  esac",
    ]


def _leading_variable_case(name: str) -> list[str]:
    """The containment ANCHOR for a template that BEGINS with ``name``.

    A template like ``$SLURM_TMPDIR`` or ``$SLURM_TMPDIR/models`` has no
    static literal prefix to anchor containment against, and refusing every
    such template would silently drop cleanup for the sites that spell their
    scratch that way — the exact failure this module was written to end. The
    honest anchor there is the leading variable's own expansion: the removal
    must land on that directory or inside it, so ``$SLURM_TMPDIR`` cleans up
    and ``$SLURM_TMPDIR/..`` (which resolves to its PARENT) does not.

    Both spellings, for the same reason :func:`_variable_case` handles both.
    """
    return [
        f'    case "${{template}}" in',
        f"      '${name}'|'${name}'/*|'${{{name}}}'|'${{{name}}}'/*)",
        f'        anchor="${{{name}}}" ;;',
        "    esac",
    ]


#: The rendered canonicalization helper's name. Prefixed because these lines
#: are injected into somebody else's job script and must collide with nothing.
CANONICAL_FUNCTION = "steerlab_canonical_path"


def _canonical_path_function() -> list[str]:
    """The shell helper that resolves a path to its PHYSICAL form.

    Every check in :func:`cleanup_lines` above this point is lexical, and a
    lexical prefix test is exactly what a symlink defeats: with
    ``<anchor>/alice`` a link into somebody else's tree,
    ``<anchor>/alice/<job>`` passes the string comparison and ``rm -rf``
    follows the link out of the anchor (third-round review, 2026-08-24).

    ``readlink -f`` is NOT used, even though Slurm compute nodes are Linux and
    would have it: BSD/macOS ``readlink -f`` fails outright on a path whose
    last component does not exist, and the stage directory routinely does not
    exist when the trap runs — a job that staged nothing, or that cleaned up
    already. ``cd -P``/``pwd -P`` are bash builtins, need no coreutils, and
    resolve the EXISTING part of the path, which is the only part a symlink can
    live in; the not-yet-created tail is re-appended literally. That is enough,
    because a symlink can only exist where a file exists.
    """
    return [
        f"{CANONICAL_FUNCTION}() {{",
        '  # Physical form of "$1": symlinks resolved through the part of the',
        "  # path that exists, the rest appended as written. Prints it; fails",
        "  # only if even the deepest existing ancestor will not resolve.",
        "  local target rest name base",
        '  target="$1"',
        '  rest=""',
        '  while [ ! -d "${target}" ]; do',
        '    case "${target}" in /|"") break ;; esac',
        '    name="${target##*/}"',
        '    rest="${name}${rest:+/}${rest}"',
        '    target="${target%/*}"',
        '    [ -n "${target}" ] || target="/"',
        "  done",
        '  base="$(cd -P -- "${target}" 2>/dev/null && pwd -P)" || base=""',
        '  [ -n "${base}" ] || return 1',
        '  if [ -n "${rest}" ]; then',
        "    printf '%s/%s\\n' \"${base%/}\" \"${rest}\"",
        "  else",
        "    printf '%s\\n' \"${base}\"",
        "  fi",
        "}",
    ]


def cleanup_lines(*, environ=None) -> list[str]:
    """THE node-scratch cleanup block, as script lines.

    Emitted verbatim by every renderer. The guard is the whole safety
    argument: a directory is removed only when the site's template is
    job-scoped by construction (:data:`JOB_SCOPING_VARIABLES`) AND the job
    actually has a job id, so a shared node cache never matches and a
    non-Slurm shell never fires.

    The guard is layered, and the layers are cumulative: the template must name
    a job-scoping variable, carry no ``.``/``..`` component, expand to an
    absolute path with nothing left unexpanded, and sit at or under its own
    anchor. All of that is a judgement about TEXT. The last gate, immediately
    before the removal, is the only one that asks the FILESYSTEM: both the
    anchor and the target are resolved to their physical form and the
    containment question is put again — because a symlink in the middle of an
    otherwise blameless path is exactly what a string comparison cannot see
    (third-round review, 2026-08-24).

    The function never calls ``exit`` and always ends successfully, which is
    what lets it be trapped on EXIT alongside the checkpoint trap without
    disturbing the job's recorded exit status.
    """
    if scheduler_purges_node_scratch(environ):
        return [
            "",
            "# This site declares that its SCHEDULER reclaims node-local",
            f"# scratch at job end ({SCHEDULER_PURGES_ENV}), so no cleanup",
            "# trap is rendered: a job racing the epilog for the same",
            "# directory adds risk and removes nothing the site was not",
            "# already going to remove.",
        ]
    return [
        "",
        "# The one filesystem-aware helper the trap below uses; see",
        "# node_scratch._canonical_path_function for why it is not readlink -f.",
        *_canonical_path_function(),
        "",
        "# cluster-operator requirement (2026-08-19): a job that stages to",
        "# node-local scratch must remove its own directory — some sites do",
        "# NOT wipe it at job end, and unrequested/unremoved staging leaks",
        "# TBs. The path is the SITE's, not this engine's: whatever",
        f"# {STAGE_DIR_ENV} names (the profile's",
        "# storage.nodeStageDirTemplate), expanded here on the node.",
        "# Guard: only a template that embeds a job-scoping variable —",
        f"# {', '.join('$' + name for name in JOB_SCOPING_VARIABLES)} —",
        "# is ever removed; a shared node cache never matches. The env value",
        "# arrives single-quoted, so the case pattern sees the literal",
        "# variable text.",
        "cleanup_node_scratch() {",
        "  local template stage_dir scoped anchor canon_stage canon_anchor",
        f'  [ -n "${{{STAGE_DIR_ENV}:-}}" ] || return 0',
        '  [ -n "${SLURM_JOB_ID:-}" ] || return 0',
        f'  template="${{{STAGE_DIR_ENV}}}"',
        '  stage_dir="${template}"',
        "  scoped=0",
        "  # TRAVERSAL, in the template itself (external review, 2026-08-24).",
        "  # '/lscratch/$SLURM_JOB_ID/../../shared' is job-scoped by every",
        "  # test below — it names $SLURM_JOB_ID, it expands to an absolute",
        "  # path, nothing is left unexpanded — and it resolves to a shared",
        "  # directory this job does not own. A '.' or '..' COMPONENT is",
        "  # never anything a staging template needs, so it is refused",
        "  # outright rather than normalized away.",
        '  case "/${template}/" in */../*|*/./*) return 0 ;; esac',
        *[line for name in JOB_SCOPING_VARIABLES
          for line in _variable_case(name, scoping=True)],
        *[line for name in COMPANION_VARIABLES
          for line in _variable_case(name, scoping=False)],
        "  # Job-scoped by construction, or nobody's to remove.",
        '  [ "${scoped}" = 1 ] || return 0',
        "  # …and the same traversal check on the RESULT: the template may be",
        "  # clean while a variable's VALUE carries the '..' instead.",
        '  case "/${stage_dir}/" in */../*|*/./*) return 0 ;; esac',
        "  # Belt and braces: an absolute path, with nothing left unexpanded.",
        "  # Neither can fail given the checks above; both are cheap, and the",
        "  # cost of being wrong here is somebody else's data.",
        '  case "${stage_dir}" in /*[!/]*) ;; *) return 0 ;; esac',
        '  [ "${stage_dir#*\\$}" = "${stage_dir}" ] || return 0',
        "  # CONTAINMENT. The anchor is the template's static literal prefix —",
        "  # the text before its first variable — and the path about to be",
        "  # removed must BE that directory or sit under it. For a template",
        "  # that begins with a variable there is no literal prefix, so the",
        "  # anchor is that leading variable's own expansion (which is what",
        "  # makes a bare $SLURM_TMPDIR removable and $SLURM_TMPDIR/.. not).",
        "  # With '.' and '..' components already refused above, the expanded",
        "  # path is normalized by construction, so this prefix test IS the",
        "  # normalized-containment test.",
        '  anchor="${template%%\\$*}"',
        '  while [ -n "${anchor}" ] && [ "${anchor}" != "${anchor%/}" ]; do',
        '    anchor="${anchor%/}"',
        "  done",
        '  if [ -z "${anchor}" ]; then',
        *[line for name in (*JOB_SCOPING_VARIABLES, *COMPANION_VARIABLES)
          for line in _leading_variable_case(name)],
        "  fi",
        "  # The anchor must itself be a real absolute directory, never '/':",
        "  # an empty or root anchor would contain everything, which is not a",
        "  # containment check at all.",
        '  case "${anchor}" in /*[!/]*) ;; *) return 0 ;; esac',
        '  case "/${anchor}/" in */../*|*/./*) return 0 ;; esac',
        '  [ "${anchor#*\\$}" = "${anchor}" ] || return 0',
        '  case "${stage_dir}" in',
        '    "${anchor}"|"${anchor}"/*) ;;',
        "    *) return 0 ;;",
        "  esac",
        "  # CANONICAL CONTAINMENT, immediately before the delete (third-round",
        "  # review, 2026-08-24). Everything above is a check on TEXT, and a",
        "  # symlink is precisely what text cannot see: with '<anchor>/alice' a",
        "  # link into another tree, '<anchor>/alice/<job>' passes the prefix",
        "  # test above and 'rm -rf' follows the link straight out of the",
        "  # anchor. So both sides are resolved to their physical form here and",
        "  # the containment question is asked again of THOSE.",
        "  #",
        "  # A symlinked scratch ROOT ('/scratch -> /lustre/scratch') is",
        "  # legitimate and common, and needs no exception: it resolves on BOTH",
        "  # sides, so canonical containment still holds. Only a link that",
        "  # LEAVES the anchor is refused — an intermediate symlink is not",
        "  # suspicious in itself.",
        "  #",
        "  # A stage dir that does not exist canonicalizes to its own lexical",
        "  # form (its nearest existing ancestor, resolved, plus the missing",
        "  # tail), stays contained, and is handed to an 'rm -rf' that removes",
        "  # nothing and says nothing — there was nothing to remove.",
        f'  canon_stage="$({CANONICAL_FUNCTION} "${{stage_dir}}")" || return 0',
        f'  canon_anchor="$({CANONICAL_FUNCTION} "${{anchor}}")" || return 0',
        '  case "${canon_stage}" in',
        '    "${canon_anchor}"|"${canon_anchor}"/*) ;;',
        "    *) return 0 ;;",
        "  esac",
        "  # The path the SITE named is what is removed, not its canonical",
        "  # form: resolving was the check, not a relocation of the delete.",
        '  rm -rf -- "${stage_dir}" 2>/dev/null || true',
        "}",
        "# EXIT only, so it composes with any checkpoint trap (USR1/TERM) and",
        "# never disturbs the job's recorded exit status: the function never",
        "# calls exit and always ends successfully.",
        "trap cleanup_node_scratch EXIT",
    ]


# =============================================================================
# The canonical ad-hoc wrapper
# =============================================================================

#: The rendered wrapper's file name inside the metadata root — beside
#: ``controller-job.sbatch``, for the same reason: it is a per-node artifact an
#: operator submits by hand, not code.
WRAPPER_NAME = "node-scratch-wrapper.sbatch"

#: The stamp line a rendered wrapper carries, so a reader can tell a rendered
#: artifact from something hand-edited. Same shape as the controller render's.
WRAPPER_STAMP_MARKER = "# steerlab-node-scratch-wrapper:"


def wrapper_path(metadata_root: str | None = None) -> str:
    """Where the wrapper lives. ``STEERLAB_METADATA_ROOT`` is the authority and
    ``$HOME/.steerlab`` the fallback — identical resolution to
    ``controller_render.rendered_path``, so the two artifacts sit together."""
    from .controller_render import rendered_path

    return os.path.join(os.path.dirname(rendered_path(metadata_root)),
                        WRAPPER_NAME)


def render_wrapper(*, environ=None, resources=None) -> str:
    """The canonical ad-hoc sbatch wrapper, as text.

    Its contract in one sentence: **the payload command is the only variable.**
    Everything else — the node-scratch gres request, the stage-dir export, the
    cleanup trap — comes from the site, through the same
    :func:`cleanup_lines` the study renderer uses.

    The header is deliberately SMALL. It carries the placement facts an ad-hoc
    job cannot get right by guessing (partition, account, qos, gres including
    the scratch token) and nothing else; it does not carry ``--export=NONE``
    or the checkpoint signal, because those belong to a job that reconstructs
    its runtime from the bundle env and this one does not. An operator edits
    the walltime and memory at the top; the trap is not theirs to edit.
    """
    from .api.executors import SlurmResources, combined_gres

    env = os.environ if environ is None else environ
    res = resources if resources is not None else SlurmResources.from_env(
        job_name="steerlab-adhoc")

    lines = [
        "#!/usr/bin/env bash",
        f"{WRAPPER_STAMP_MARKER} the site's node-scratch contract, rendered",
        "#",
        "# Submit an ad-hoc job THROUGH this wrapper rather than writing a bare",
        "# sbatch:",
        "#",
        f"#     sbatch {WRAPPER_NAME} <your command> [args...]",
        "#",
        "# Everything below the payload line is the SITE's node-scratch",
        "# contract and is rendered, not authored — re-render it rather than",
        "# editing it, and never copy the trap into a script of your own.",
        "# Resource lines above the payload are yours to adjust.",
        "",
        f"#SBATCH --job-name={res.job_name}",
    ]
    for directive, value in (
        ("partition", res.partition),
        ("account", res.account),
        ("qos", res.qos),
        ("time", res.walltime),
        ("mem", res.memory),
    ):
        if value:
            lines.append(f"#SBATCH --{directive}={value}")
    lines.append(f"#SBATCH --cpus-per-task={res.cpus_per_task}")
    gres = combined_gres(_wrapper_gpu_gres(res), res.normalized_scratch_gres())
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    else:
        lines.append(
            "# (this site declares no gres, GPU or node-local scratch, so no")
        lines.append("#  --gres directive is rendered)")

    lines += ["", "set -euo pipefail", ""]
    stage = (env.get(STAGE_DIR_ENV) or "").strip()
    if stage:
        lines += [
            "# The site's node-local staging template, carried VERBATIM: the",
            "# variables inside it expand HERE, on the compute node.",
            f"export {STAGE_DIR_ENV}={shlex.quote(stage)}",
        ]
    else:
        lines += [
            f"# This site declares no {STAGE_DIR_ENV}, so nothing stages to",
            "# node-local scratch by default. If you stage something by hand,",
            "# export the variable ABOVE this line and the trap below will",
            "# clean it up — provided the path is job-scoped.",
            f"# export {STAGE_DIR_ENV}='/node/scratch/$SLURM_JOB_ID'",
        ]
    lines += cleanup_lines(environ=env)
    lines += [
        "",
        "# ---- the payload: the only variable in this script --------------",
        'echo "SteerLab ad-hoc job on $(hostname) at $(date -Is)"',
        '"$@"',
    ]
    return "\n".join(lines) + "\n"


def _wrapper_gpu_gres(res) -> str | None:
    """The GPU half of the wrapper's gres, or None.

    A site with an UNDECLARED GPU vocabulary refuses to normalize a gres
    (declare-or-refuse, ``SlurmResources.normalized_gres``). That refusal is
    right for a study job and wrong for this artifact: a wrapper that cannot
    render because the GPU list is missing would leave the operator with no
    canonical way to request node scratch at all — which is the exact hole
    being closed. Degrade to "no GPU directive"; the operator adds one.
    """
    try:
        return res.normalized_gres()
    except ValueError:
        return None


def write_wrapper(*, metadata_root: str | None = None, environ=None,
                  resources=None) -> dict:
    """Render the wrapper into the metadata root and describe what happened.

    Returns ``{"path", "written", "schedulerPurgesNodeScratch",
    "nodeStageDirTemplate", "nodeScratchGres"}``. ``written`` is False when the
    bytes on disk already match — re-rendering is idempotent, so a standing
    ritual can call it every time without churning mtimes.
    """
    env = os.environ if environ is None else environ
    path = wrapper_path(metadata_root)
    text = render_wrapper(environ=env, resources=resources)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = None
    try:
        with open(path, encoding="utf-8") as handle:
            existing = handle.read()
    except OSError:
        pass
    written = existing != text
    if written:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(path, 0o755)
    return {
        "path": path,
        "written": written,
        "schedulerPurgesNodeScratch": scheduler_purges_node_scratch(env),
        "nodeStageDirTemplate": (env.get(STAGE_DIR_ENV) or "").strip() or None,
        "nodeScratchGres": (env.get(SCRATCH_GRES_ENV) or "").strip() or None,
    }


__all__ = [
    "CANONICAL_FUNCTION",
    "COMPANION_VARIABLES", "JOB_SCOPING_VARIABLES", "SCHEDULER_PURGES_ENV",
    "SCRATCH_GRES_ENV", "STAGE_DIR_ENV", "WRAPPER_NAME",
    "WRAPPER_STAMP_MARKER", "cleanup_lines", "render_wrapper",
    "scheduler_purges_node_scratch", "wrapper_path", "write_wrapper",
]
