"""Project-tree layout, with a **runtime-injectable root** (parallel to Swift
``VectorCatalog`` paths — but not baked at compile time).

On the Swift side ``projectRoot`` is derived from ``#filePath`` (the build
machine's source dir, baked into the binary). On a cluster that is wrong: the
cluster owns the canonical data tree, so the root must come from the
environment (``STEERLAB_ROOT``) or be passed in. Everything else
(``runs/``, ``prompts/concepts``, ``experiments/``) hangs off that root with the
same names the Swift app uses, so artifacts interoperate.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def project_root() -> str:
    """Canonical data tree root. ``STEERLAB_ROOT`` env, else the current dir."""
    return os.environ.get("STEERLAB_ROOT") or os.getcwd()


def looks_like_source_checkout(root: str | None = None) -> bool:
    """True when the artifact root appears to be the SteerLab SOURCE CHECKOUT
    rather than a data workspace.

    Heuristic: the tree carries the code (``Server/steerlab_server`` or the
    Swift ``Package.swift``) AND lacks the ``WORKSPACE.md`` marker that
    app-created workspaces carry. Serving from the checkout is legal (dev
    workflows use it), but it means every server-side authoring/build/run
    write lands in the code repo — the caller should warn loudly, and
    ``GET /api/info`` exposes the same verdict so clients can badge the
    pairing state.
    """
    base = root or project_root()
    if os.path.isfile(os.path.join(base, "WORKSPACE.md")):
        return False
    return (os.path.isdir(os.path.join(base, "Server", "steerlab_server"))
            or os.path.isfile(os.path.join(base, "Package.swift")))


def seed_workspace(root: str) -> None:
    """Create a minimal data workspace at ``root``: the three canonical
    subtrees plus the ``WORKSPACE.md`` marker app-created workspaces carry
    (the marker keeps ``looks_like_source_checkout`` honest for the new
    tree). Mirrors the Mac app's workspace seeding at the directory-skeleton
    level; ``git init`` is deliberately NOT done here — the freeze
    cleanliness gate will surface an untracked workspace, and initializing
    repositories at caller-named server paths is not this instrument's call.
    Idempotent: seeding an existing workspace touches nothing it has."""
    for sub in ("prompts", "experiments", "runs"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    marker = os.path.join(root, "WORKSPACE.md")
    if not os.path.exists(marker):
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("# SteerLab workspace\n\n"
                     "Created by the steerlab-server workspace switch "
                     f"({datetime.now(timezone.utc).isoformat()}).\n")


def runs_directory(root: str | None = None) -> str:
    if root is None and os.environ.get("STEERLAB_RUN_ROOT"):
        return os.environ["STEERLAB_RUN_ROOT"]
    return os.path.join(root or project_root(), "runs")


def concepts_directory(root: str | None = None) -> str:
    return os.path.join(root or project_root(), "prompts", "concepts")


def neutral_corpus_path(root: str | None = None) -> str:
    return os.path.join(root or project_root(), "prompts", "neutral", "corpus.jsonl")


def experiments_directory(root: str | None = None) -> str:
    return os.path.join(root or project_root(), "experiments")


def templates_directory(root: str | None = None) -> str:
    """RepE reader task-template registry (one JSON per template)."""
    return os.path.join(root or project_root(), "prompts", "templates")


def template_path(template_id: str, root: str | None = None) -> str:
    return os.path.join(templates_directory(root), f"{template_id}.json")


def repe_readers_directory(root: str | None = None) -> str:
    """Pinned RepE reader pair corpora — the same tree the Swift engine writes
    (``ConceptBuilder``: ``prompts/readers/<concept>/pairs.jsonl``). Stimuli are
    substrate-agnostic; only the *fitted* artifacts are per-substrate."""
    return os.path.join(root or project_root(), "prompts", "readers")


def reader_pairs_path(concept: str, root: str | None = None) -> str:
    """Paired reader stimuli for one concept (unrendered; see repe_reader)."""
    return os.path.join(repe_readers_directory(root), concept, "pairs.jsonl")


def concept_directory(name: str, root: str | None = None) -> str:
    return os.path.join(concepts_directory(root), name)


def jlens_lenses_directory(root: str | None = None) -> str:
    """Imported J-lens artifacts — a mutable LIBRARY subtree inside ``runs/``,
    like ``runs/model-variants`` and ``runs/neutral-pcs``.

    A lens is a derived measuring instrument, not source data, so it does not
    belong under ``prompts/``; and it is not a run, so it is not immutable —
    a lens record accumulates qualification entries over time. The upstream
    ``.pt`` stays in the HF cache; what lands here is the converted per-layer
    representation plus the record (see the J-lens plan §4.1–4.2).

    NOTE for cluster use: the workspace commonly lives on ``/scratch``, which
    is **30-day purge, no recovery**. That is acceptable only because
    everything here is re-derivable from the hash-pinned HF cache on
    ``/work`` — keep the import/convert step idempotent so a purged workspace
    rebuilds without re-downloading.
    """
    return os.path.join(runs_directory(root), "jlens-lenses")


def jlens_lens_directory(lens_id: str, root: str | None = None) -> str:
    """One imported lens's artifact directory."""
    return os.path.join(jlens_lenses_directory(root), lens_id)


def resolve(path: str, root: str | None = None) -> str:
    """Resolve a (possibly project-relative) artifact path to an absolute one —
    relative paths are joined to the project root (parallel to Swift
    ``ModelVariantStore.absoluteURL``), so variant artifacts, adapters, neutral
    bases, and scenarios open from the canonical tree, not the cwd."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(project_root() if root is None else root, path)


#: The workspace subtrees an artifact reference can name. A path is rebased
#: only from one of these — an arbitrary shared prefix is not evidence that
#: two paths name the same artifact. (Swift twin:
#: ``ArtifactIdentity.rebasableRoots``.)
_REBASABLE_ROOTS = ("runs", "experiments", "adapters", "prompts")


def _artifact_present(path: str) -> bool:
    """A reference may be an extension-less vector locator (``<dir>/<name>``
    with ``<name>.json`` + ``<name>.safetensors`` beside it) or a plain
    file/directory — the sidecar is what proves a locator is really there."""
    return os.path.exists(path) or os.path.exists(path + ".json")


def resolve_artifact(reference: str, root: str | None = None) -> str:
    """:func:`resolve`, plus the foreign-machine rebase fallback — the Python
    twin of Swift ``ArtifactIdentity.rebasedToWorkspace``.

    An artifact reference recorded on ANOTHER machine's filesystem
    (``/Users/…/Workspace/runs/<run>/<leaf>`` written by the Mac app, a
    ``/scratch/…`` path written on a cluster) names nothing here even though
    ``runs/<run>/<leaf>`` may be sitting right in this workspace, imported.
    Observed live 2026-08-04: six app-promoted agents carried absolute Mac
    paths and every cluster panel/run died on them after the model load,
    while the bundle had shipped the vectors at the correct relative paths.

    Deliberately a FALLBACK, applied only when the direct resolution names
    nothing: it can turn a certain failure into a hit and can never redirect
    a reference that already resolves. The rebased tail must itself exist,
    so an unrelated path with a ``runs/`` segment is left alone rather than
    silently repointed."""
    resolved = resolve(reference, root)
    if not reference or not os.path.isabs(reference):
        return resolved
    if _artifact_present(resolved):
        return resolved
    parts = [p for p in os.path.normpath(reference).split(os.sep) if p]
    for anchor in _REBASABLE_ROOTS:
        if anchor not in parts:
            continue
        # Last occurrence: a foreign workspace root may itself contain
        # a "runs" component.
        index = len(parts) - 1 - parts[::-1].index(anchor)
        if index + 1 >= len(parts):
            continue
        candidate = resolve(os.path.join(*parts[index:]), root)
        if _artifact_present(candidate):
            return candidate
    return resolved


#: Bounded retry budget for :func:`make_unique_run_directory`. Far above any
#: real same-millisecond fan-out (Slurm shard counts are tens, not hundreds);
#: exhausting it means something is wrong, so it raises rather than looping.
MAX_RUN_DIRECTORY_ATTEMPTS = 500


class UnsafeRunSlug(ValueError):
    """A run-directory slug that would not stay inside ``runs/``."""


def _checked_slug(slug: str) -> str:
    """Refuse a run slug that could leave ``runs/`` — the containment check
    at the helper, not at each of its ~50 call sites.

    Most callers build the slug from a verb plus a name they safe-named
    first, but several interpolate data straight from a request body, a
    bundle's metadata, or a scenario file (``f"gemmascope-{name}"``,
    ``f"submit-bundle-{experiment}-{verb}"``). One separator in any of those
    turns ``runs/<stamp>-<slug>`` into an arbitrary filesystem path:
    ``gemmascope-../../../escaped`` resolved outside the workspace entirely.

    REFUSES rather than sanitizes on purpose. Rewriting a bad slug would
    silently change where a run lands — and a legitimate slug is left exactly
    as it was, so every existing on-disk run directory keeps its name and
    stays resolvable.
    """
    text = str(slug)
    if not text:
        raise UnsafeRunSlug("run directory slug is empty")
    if "\0" in text:
        raise UnsafeRunSlug(f"run directory slug contains a NUL byte: {text!r}")
    separators = {os.sep, os.altsep, "/", "\\"} - {None}
    for separator in separators:
        if separator in text:
            raise UnsafeRunSlug(
                f"run directory slug may not contain a path separator: "
                f"{text!r} (a run always lands directly in runs/)")
    if os.path.isabs(text) or text in (os.curdir, os.pardir):
        raise UnsafeRunSlug(
            f"run directory slug may not be a path: {text!r}")
    return text


def make_unique_run_directory(slug: str, root: str | None = None) -> str:
    """Create a fresh immutable ``runs/<stamp>-<slug>`` directory.

    Fractional-second UTC stamp + numeric suffix on collision, matching the
    Swift convention. Run directories are **never overwritten or mutated**.

    Creation itself is the exclusivity test: ``os.makedirs`` *without*
    ``exist_ok`` is an atomic ``mkdir(2)``, so only one caller can win a given
    name and the loser retries under the next suffix. A check-then-create loop
    (``while os.path.exists(...)`` then ``makedirs``) leaves a TOCTOU window —
    two Slurm generation shards resolving the same millisecond stamp on a
    shared filesystem both exit the loop with the same path, and one dies with
    ``FileExistsError`` (or, worse on the Swift side, both proceed into one
    directory and mutate an "immutable" run).

    Raises :class:`UnsafeRunSlug` (a ``ValueError``) for a slug that would
    escape ``runs/`` — see :func:`_checked_slug`.
    """
    slug = _checked_slug(slug)
    runs_root = runs_directory(root)
    os.makedirs(runs_root, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}"
    url = os.path.join(runs_root, f"{stamp}-{slug}")
    counter = 1
    for _ in range(MAX_RUN_DIRECTORY_ATTEMPTS):
        try:
            os.makedirs(url)
            # Register it for retention (2026-07-24): a stage that fails
            # never returns its directory, and several verbs report it
            # nowhere else, so the failure paths would have nothing to
            # package. Recording it HERE covers every verb on every path
            # with no per-task plumbing.
            from . import run_status
            run_status.current_run_directory.set(url)
            return url
        except FileExistsError:
            counter += 1
            url = os.path.join(runs_root, f"{stamp}-{slug}-{counter}")
    raise RuntimeError(
        f"could not create a unique run directory under {runs_root!r} for slug "
        f"{slug!r}: {MAX_RUN_DIRECTORY_ATTEMPTS} consecutive names at stamp "
        f"{stamp} were already taken")
