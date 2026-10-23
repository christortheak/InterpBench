#!/usr/bin/env python3
"""The site's torch, as bootstrap.sh needs to see it (2026-10-22).

bootstrap.sh installs torch from the site's index (``--torch-index``), then
the rest of the committed lock. Run by the environment's own interpreter, this
answers three questions about the torch installed there, without importing it:

  version       the installed torch version ("2.11.0+cu128"), or nothing
  cuda          the CUDA version that torch was built for ("12.8"), or nothing
  filter-lock   the lock lines on stdin that the installed torch allows; a pin
                torch forbids is left out and named on stderr

Why the last one exists: the lock is resolved against the lock's torch, and a
site whose index carries a different build can contradict it (a pinned
setuptools==84 against torch 2.11's setuptools<82). Left free, pip settled that
by replacing the site's torch with PyPI's. The pin gives way instead, by name,
and each run's stamp records what was installed.

Standard library only, with packaging taken from pip when it is not installed:
this runs before anything else is.
"""
import importlib.metadata as md
import importlib.util
import re
import sys
from pathlib import Path


def _packaging():
    """packaging, or pip's copy of it in a fresh environment. Imported only
    where requirements are read, so `version` and `cuda` never depend on it."""
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.utils import canonicalize_name
    return Requirement, canonicalize_name


def torch_version():
    try:
        return md.version("torch")
    except md.PackageNotFoundError:
        return None


def torch_cuda():
    """``cuda`` in torch/version.py, read rather than imported: None for a
    CPU or ROCm build, or when torch is not installed."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    version_file = Path(list(spec.submodule_search_locations)[0]) / "version.py"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError:
        return None
    found = re.search(r"^cuda\s*(?::[^=]*)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
    return found.group(1) if found else None


def torch_limits():
    """What the installed torch requires of other packages, by name."""
    if torch_version() is None:
        return {}
    Requirement, canonicalize_name = _packaging()
    limits = {}
    for text in md.requires("torch") or []:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        limits.setdefault(canonicalize_name(requirement.name), []).append(requirement.specifier)
    return limits


def filter_lock(lines, limits, torch):
    """The lines torch allows, and a warning on stderr for each it does not."""
    _, canonicalize_name = _packaging()
    for line in lines:
        name, pinned_sep, rest = line.strip().partition("==")
        pinned = re.split(r"[\s;\\]", rest, maxsplit=1)[0] if pinned_sep else ""
        refused = [str(specifier) for specifier in limits.get(canonicalize_name(name), [])
                   if pinned and not specifier.contains(pinned, prereleases=True)]
        if refused:
            print(f"bootstrap.sh: WARNING the lock pins {name}=={pinned}, which the site's "
                  f"torch {torch} does not allow ({', '.join(refused)}); leaving {name} to "
                  "torch's requirement", file=sys.stderr)
            continue
        yield line if line.endswith("\n") else line + "\n"


def main(argv):
    command = argv[1] if len(argv) > 1 else ""
    if command == "version":
        print(torch_version() or "")
    elif command == "cuda":
        print(torch_cuda() or "")
    elif command == "filter-lock":
        sys.stdout.writelines(filter_lock(sys.stdin, torch_limits(), torch_version()))
    else:
        print("usage: site_torch.py version|cuda|filter-lock", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
