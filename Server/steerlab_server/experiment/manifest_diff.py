"""Which FIELDS differ between two manifests — the readable half of every
hash comparison in the lifecycle.

A hash comparison answers "same or not" and nothing else, which is the right
primitive and the wrong message. "produced under experiment hash 6eb30c…, but
the study now hashes 9a12f4…" is unactionable: it does not say what changed,
so the researcher cannot tell a meaningful edit from a stray one, and the only
remaining move is ``--force``. Gates that can only be forced stop being gates.

Deliberately generic over the serialized manifest rather than enumerating
fields — a per-field implementation goes stale the moment the manifest grows a
key, and going stale silently is how the epoch guard's diagnosis drifted from
its check in the first place.

Cross-engine twin of ``Sources/ExperimentKit/ManifestDiff.swift``.
"""

from __future__ import annotations

from typing import Any


def _flatten(value: Any, path: str, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        # An EMPTY container is itself a value: without this, ``{}`` and an
        # absent key flatten identically and a real edit reads as no change.
        if not value:
            out[path or "."] = "{}"
            return
        for key, child in value.items():
            _flatten(child, f"{path}.{key}", out)
    elif isinstance(value, (list, tuple)):
        if not value:
            out[path or "."] = "[]"
            return
        for index, child in enumerate(value):
            _flatten(child, f"{path}[{index}]", out)
    elif value is None:
        out[path] = "null"
    elif isinstance(value, bool):
        out[path] = "true" if value else "false"
    else:
        out[path] = str(value)


#: Dropped by ``Manifest.content_hash``; dropping them here too keeps the
#: diff consistent with the gate. Naming a field the hash ignores would
#: contradict the verdict the diff is explaining.
VOLATILE_KEYS = ("status", "frozenAt", "freezeHash", "gitCommit", "frozenBy",
                 "appVersion", "createdAt", "freezeForced",
                 "forcedGatesSkipped")


def flattened(manifest) -> dict[str, str]:
    """Dotted-path → rendered value for a Manifest (or a plain dict)."""
    raw = manifest if isinstance(manifest, dict) else manifest.raw
    payload = {k: v for k, v in raw.items() if k not in VOLATILE_KEYS}
    out: dict[str, str] = {}
    _flatten(payload, "", out)
    return out


def differences(left, right) -> list[tuple[str, str, str]]:
    """``(path, left, right)`` for every differing path, sorted. A key present
    on one side only renders the absent side as ``absent``."""
    a, b = flattened(left), flattened(right)
    return [(path, a.get(path, "absent"), b.get(path, "absent"))
            for path in sorted(set(a) | set(b))
            if a.get(path) != b.get(path)]


def summary(left, right, limit: int = 4) -> str:
    """One-line ``path: left → right; …`` summary, empty when identical."""
    all_diffs = differences(left, right)
    if not all_diffs:
        return ""
    shown = "; ".join(f"{p}: {l} → {r}" for p, l, r in all_diffs[:limit])
    rest = len(all_diffs) - min(limit, len(all_diffs))
    return f"{shown}; and {rest} more" if rest > 0 else shown


def changed_fields(live, snapshot) -> str:
    """The ``: fieldA: x → y`` clause a refusal appends, or "" when either
    manifest is unavailable. A refusal that prints only two hashes leaves
    ``--force`` as the sole next move."""
    if live is None or snapshot is None:
        return ""
    text = summary(snapshot, live)
    return f": {text}" if text else ""
