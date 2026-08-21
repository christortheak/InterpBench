"""What the measurement stack ACTUALLY was: resolved package versions for the
run stamp, and a non-blocking drift advisory against the platform lock (WP6 R1).

Why (docs/GENERAL-DISTRIBUTION-WORK-PLAN.md, WP6): ``pyproject.toml`` declares
floors, so two sites can satisfy the same manifest with different
torch/transformers — on the substrate where the reproducibility claims actually
live. ``Server/requirements-<platform>.lock`` is the intended resolution; this
module records the achieved one into each run's ``config.json`` and says so out
loud at ``experiment run`` start when the two disagree.

Two deliberate limits:

* **Provenance, never a gate.** Nothing here refuses a run. A version mismatch
  is information a reader needs, not a reason to lose a queued cluster job
  (post-submit drift policy: continue loudly + stamp).
* **Declared set, not ``pip freeze``.** Only packages whose version can move a
  NUMBER are stamped — the compute substrate, the model implementations, the
  tokenizer, the artifact IO, the linear algebra, the statistics, and the two
  optional science backends. Transport (fastapi/uvicorn/pydantic) is excluded:
  it cannot change a measurement, and a stamp full of irrelevant churn is a
  stamp nobody diffs. Versions come from ``importlib.metadata`` — never a
  ``pip`` subprocess, which is slow, needs a network-capable pip, and reports
  a different environment than the one this process imported from.
"""

from __future__ import annotations

import os
import platform as _platform_mod
import re
from importlib import metadata as _metadata

#: The declared science-relevant package set (sorted; distribution names as
#: they appear on PyPI and in the locks). A package that is not installed is
#: stamped ``null`` rather than omitted — the same rule the run-config contract
#: uses at top level: absent knowledge is explicit, so a reader never has to
#: guess whether "no scipy" means "not installed" or "not asked about".
SCIENCE_PACKAGES = (
    "accelerate",       # device_map / dispatch — decides where layers live
    "huggingface-hub",  # revision resolution; what "pinned commit" resolves to
    "jlens",            # J-space reading instruments (git-pinned extra)
    "numpy",            # vector arithmetic, extraction math
    "peft",             # LoRA variants
    "safetensors",      # vector/artifact IO
    "sae-lens",         # Gemma Scope SAE analysis
    "scipy",            # bootstrap CIs, Wilcoxon, FDR
    "tokenizers",       # tokenization — upstream of every prompt hash
    "torch",            # the compute substrate itself
    "transformers",     # model implementations, generation loop
)

#: Packages the run-start advisory compares against the lock. Narrower than
#: the stamp on purpose: these are the two whose drift changes numbers most
#: directly, and a short warning is a warning that gets read.
_ADVISORY_PACKAGES = ("torch", "transformers")

#: ``run_platform()`` value -> committed lock filename. Platforms absent here
#: (linux-aarch64, windows, …) simply have no lock and no advisory.
_LOCK_BY_PLATFORM = {
    "macOS-arm64": "requirements-macos-arm64.lock",
    "linux-x86_64": "requirements-linux-x86_64.lock",
}

_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;#]+)")


def package_version(name: str) -> str | None:
    """Installed version of a distribution, or None when it is not installed.
    ``importlib.metadata`` only — the version of what THIS interpreter would
    import, which is the only version a run's numbers can depend on."""
    try:
        return _metadata.version(name)
    except Exception:  # PackageNotFoundError, and broken dist-info in the wild
        return None


def python_environment() -> dict:
    """The ``pythonEnvironment`` payload for a run's ``config.json``.

    ``{"python": "3.12.13", "implementation": "cpython",
       "packages": {"torch": "2.13.0+cu128", …}}`` — every declared package
    present as a key, ``null`` when not installed. The torch value keeps its
    local version segment (``+cu128``) when the wheel carries one: which CUDA
    build ran is exactly the site fact this stamp exists to preserve."""
    return {
        "python": _platform_mod.python_version(),
        "implementation": _platform_mod.python_implementation().lower(),
        "packages": {name: package_version(name) for name in SCIENCE_PACKAGES},
    }


def lock_path(run_platform_value: str) -> str | None:
    """The committed lock for this platform, or None when there is none (an
    unlocked platform, or a non-editable install whose lock did not ship).

    ``STEERLAB_LOCK_FILE`` overrides — the seam the tests drive, and the escape
    hatch for a site that pins its own resolution."""
    override = (os.environ.get("STEERLAB_LOCK_FILE") or "").strip()
    if override:
        return override if os.path.exists(override) else None
    name = _LOCK_BY_PLATFORM.get(run_platform_value)
    if not name:
        return None
    path = os.path.join(_SERVER_DIR, name)
    return path if os.path.exists(path) else None


def parse_lock(path: str) -> dict[str, str]:
    """``name -> pinned version`` from a ``uv pip compile`` / pip-tools lock.
    Names are lowercased with ``_`` normalized to ``-`` (PEP 503-ish), so
    ``huggingface_hub`` and ``huggingface-hub`` compare equal. Comment lines,
    ``# via`` continuations, hash lines, and non-``==`` requirements (the git
    pins) are skipped."""
    pins: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line[:1].strip():   # blank, or an indented continuation
                    continue
                if line.startswith("#"):
                    continue
                match = _PIN_RE.match(line.strip())
                if match:
                    name = match.group(1).lower().replace("_", "-")
                    pins[name] = match.group(2)
    except (OSError, UnicodeDecodeError):
        # An unreadable or non-text lock yields "nothing to compare", not an
        # exception: every caller here is advisory, and none may sink a run.
        return {}
    return pins


def _public_version(value: str) -> str:
    """Version without its local segment: ``2.13.0+cu128`` -> ``2.13.0``.

    The local segment is where a site's own torch build lives, and a site
    building its own CUDA variant of the LOCKED version is the intended path,
    not drift. Comparing public versions keeps the advisory about the number
    that changes behavior."""
    return value.split("+", 1)[0]


def lock_drift(run_platform_value: str) -> list[str]:
    """Human-readable drift lines (empty when in agreement, or when there is
    no lock to compare against). Advisory only — never a refusal."""
    path = lock_path(run_platform_value)
    if not path:
        return []
    pins = parse_lock(path)
    if not pins:
        return []
    lines: list[str] = []
    for name in _ADVISORY_PACKAGES:
        pinned = pins.get(name)
        installed = package_version(name)
        if pinned is None or installed is None:
            continue
        if _public_version(installed) != _public_version(pinned):
            lines.append(
                f"{name}: installed {installed}, lock pins {pinned} "
                f"({os.path.basename(path)})")
    return lines
