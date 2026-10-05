"""Absolute paths taken out of text that leaves this machine.

A failure's ``error`` and ``traceback`` travel inside every partial evidence
bundle, and both name paths. A traceback names every frame's source file, which
is the server's install: a checkout or a virtual environment inside a home
folder, on a Mac or on a cluster account. An exception message often names the
run's own files. A bundle is then shared, with a collaborator, into a
repository, or into a public fixture, and every copy named the account that
packaged it.

``redact_paths`` rewrites those paths to spellings that name no machine and
still say what a reader needs, which is the module, the line, and the run file:

    the run's own directory          runs/<run ID>
    a sibling run directory          runs/<run ID>
    the workspace that holds them    <workspace>
    this server's package            <steerlab_server>
    installed packages               <site-packages>
    the Python standard library      <stdlib>
    a macOS per-user temp folder     <tmp>
    a home folder                    <home>
    the account's name, as a folder  <user>

Roots this process knows are replaced first, longest first, so the most
specific spelling wins: a server installed into site-packages reads
``<steerlab_server>/…``, not ``<site-packages>/steerlab_server/…``. Generic
patterns follow, for paths from another install (a GPU worker's traceback, a
dependency's message). Only text changes; nothing here is parsed back.
"""

from __future__ import annotations

import getpass
import os
import re
import site
import sys
import sysconfig

#: Characters that continue a path component. A root must start and end on a
#: component boundary, so ``/a/runs/R`` never rewrites inside ``/a/runs/R2``
#: and ``/var/folders`` never rewrites inside ``/private/var/folders``.
_COMPONENT = r"[\w.@+~-]"
_BEFORE = rf"(?<!{_COMPONENT})"
_AFTER = rf"(?!{_COMPONENT})"

_GENERIC = (
    # Any absolute path down to an installed-packages folder.
    (re.compile(rf"{_BEFORE}(?:/{_COMPONENT}+)+?"
                rf"/(?:site|dist)-packages{_AFTER}"), "<site-packages>"),
    # macOS per-user temp: /var/folders/<xx>/<id>, also spelled /private/….
    (re.compile(rf"{_BEFORE}(?:/private)?/var/folders"
                rf"/{_COMPONENT}+/{_COMPONENT}+{_AFTER}"), "<tmp>"),
    (re.compile(rf"{_BEFORE}/(?:Users|home)/{_COMPONENT}+{_AFTER}"), "<home>"),
)


def _spellings(path: str, *, depth: int = 2) -> set[str]:
    """``path`` as written and as resolved; a traceback names frames the way
    the interpreter imported them, which need not be the resolved form.

    A root shallower than ``depth`` components is skipped: a workspace at
    `/tmp` names no one, and rewriting it would relabel every unrelated path
    that happens to start there. A home folder is one component deep on some
    machines (`/root`), and it is still that account's."""
    out = set()
    for spelling in (os.path.abspath(path), os.path.realpath(path)):
        spelling = spelling.rstrip(os.sep)
        if spelling.count(os.sep) >= depth:
            out.add(spelling)
    return out


def _package_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _installed_package_roots() -> set[str]:
    roots = set()
    for key in ("purelib", "platlib"):
        roots.add(sysconfig.get_paths().get(key) or "")
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except (AttributeError, TypeError):  # a stripped-down `site`
        pass
    roots.update(entry for entry in sys.path
                 if os.path.basename(entry) in ("site-packages", "dist-packages"))
    return {root for root in roots if root}


def _known_roots(run_directory: str | None) -> dict[str, str]:
    """Each root's spellings and the placeholder that replaces them. A
    spelling two roots share (a workspace that IS the home folder) keeps the
    first, more specific, placeholder."""
    roots: dict[str, str] = {}

    def add(path: str | None, placeholder: str, *, depth: int = 2) -> None:
        if path:
            for spelling in _spellings(path, depth=depth):
                roots.setdefault(spelling, placeholder)

    if run_directory:
        run = os.path.abspath(run_directory).rstrip(os.sep)
        runs_root = os.path.dirname(run)
        if os.path.basename(os.path.realpath(runs_root)) == "runs":
            # The whole runs tree, so a chain's sibling stages read
            # `runs/<their ID>` too, and the workspace around it.
            add(runs_root, "runs")
            add(os.path.dirname(runs_root), "<workspace>")
        else:
            add(run, f"runs/{os.path.basename(run)}")
    add(_package_root(), "<steerlab_server>")
    for root in _installed_package_roots():
        add(root, "<site-packages>")
    for key in ("stdlib", "platstdlib"):
        add(sysconfig.get_paths().get(key), "<stdlib>")
    add(os.path.expanduser("~"), "<home>", depth=1)
    return roots


def _account_name() -> str | None:
    try:
        name = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name is nothing to redact
        return None
    return name if name and len(name) >= 2 else None


def _redactor(run_directory: str | None):
    roots = _known_roots(run_directory)
    # Longest first: re alternation takes the first alternative that matches,
    # not the longest, and the most specific root must win.
    known = re.compile(
        _BEFORE + "("
        + "|".join(re.escape(root) for root in sorted(roots, key=len,
                                                       reverse=True))
        + ")" + _AFTER) if roots else None
    name = _account_name()
    # A cluster's scratch and lscratch trees name the account as a folder
    # (`/scratch/<name>/…`), outside any home folder.
    account = (re.compile(rf"(?<=/){re.escape(name)}{_AFTER}")
               if name else None)

    def redact(text: str) -> str:
        if known is not None:
            text = known.sub(lambda match: roots[match.group(1)], text)
        for pattern, placeholder in _GENERIC:
            text = pattern.sub(placeholder, text)
        if account is not None:
            text = account.sub("<user>", text)
        return text
    return redact


def redact_paths(text: str, *, run_directory: str | None = None) -> str:
    """``text`` with every absolute path rewritten as the module docstring
    lists. ``run_directory`` is the run the text is about; without it, its
    paths fall through to the generic patterns (a home folder, say)."""
    return _redactor(run_directory)(text)


def redact_value(value, *, run_directory: str | None = None):
    """``value`` with :func:`redact_paths` applied to every string inside it,
    at any depth. Keys and non-string values are kept."""
    redact = _redactor(run_directory)

    def walk(item):
        if isinstance(item, str):
            return redact(item)
        if isinstance(item, dict):
            return {key: walk(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(inner) for inner in item]
        return item
    return walk(value)
