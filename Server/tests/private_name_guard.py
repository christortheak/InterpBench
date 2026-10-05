"""How a test in this suite reaches the private-name list.

The neutrality guards used to carry one site's identifiers as string
literals — which put the names they guard against into a public repository
and, through the bundled payloads, into the app. The list now lives outside
the repository and is read by one loader, ``scripts/ci/private_names.py``;
this module is the suite's door to it.

A guard calls :func:`private_names` first. With the list present it gets the
terms. With the list absent the guard is SKIPPED with the reason, unless
``STEERLAB_REQUIRE_PRIVATE_NAMES=1``, in which case it FAILS — the main
repository's CI and the maintainer's release gate set that, so a guard can
never pass vacuously because a file went missing.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

LOADER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "ci" / "private_names.py")


def loader():
    """``scripts/ci/private_names.py`` as a module, or ``None`` in a tree
    that carries ``Server/`` without ``scripts/`` (a deployed payload)."""
    if not LOADER.is_file():
        return None
    spec = importlib.util.spec_from_file_location("steerlab_private_names", LOADER)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module defines a dataclass, and
    # dataclasses resolves annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def private_names():
    """The private-name list for a guard, or a skip (a failure when the list
    is required) that says what is missing."""
    module = loader()
    if module is None:
        reason = (f"no scripts/ci/{LOADER.name} beside this suite (the tree "
                  "is not a full checkout)")
        required = os.environ.get(
            "STEERLAB_REQUIRE_PRIVATE_NAMES", "").strip().lower() in ("1", "true", "yes")
    else:
        names = module.load()
        if names is not None:
            return names
        reason = module.absent_reason()
        required = module.is_required()
    if required:
        pytest.fail(
            f"STEERLAB_REQUIRE_PRIVATE_NAMES is set, but there is {reason}. "
            "Put the list there before running the release gate.")
    pytest.skip(f"{reason}, so there is nothing to check for")


def assert_names_nothing_private(text: str, names, where: str, ignoring=()) -> None:
    """Fails when ``text`` contains a private term.

    ``where`` says what the text is. ``ignoring`` lists strings the TEST
    itself put there — its own temporary directory, say. This machine's
    checkout and home folder are always ignored too: on a developer's machine
    they carry the account name (a shell error message quotes the script's
    own path), and that is the machine's doing, not the product's.

    The failure names each term by its position in the list, never the term:
    this output lands in a CI log anyone can read."""
    __tracebackhide__ = True
    machine = own_paths(LOADER.parents[2], pathlib.Path.home())
    for own in sorted(set(map(str, ignoring)) | set(machine), key=len, reverse=True):
        if own:
            text = text.replace(own, "<path>")
    hits = names.hits(text)
    if hits:
        pytest.fail(f"{where} contains a private name: {', '.join(hits)} of "
                    f"{names.path.name}", pytrace=False)


def own_paths(*paths) -> list:
    """Every spelling of a test's own directories, for ``ignoring``: as given,
    with symlinks resolved, and without the ``/private`` prefix macOS adds."""
    spellings = []
    for path in paths:
        for spelling in (str(path), str(pathlib.Path(path).resolve())):
            spellings.append(spelling)
            if spelling.startswith("/private/"):
                spellings.append(spelling[len("/private"):])
    return spellings
