"""The cross-engine human-validation row contract, in ONE parser.

Rows mirror judge output rows:
``{"condition": …, "promptID": …, "outcome": "baseline"|"variant"|"tie"
[, "sampleIndex": …]}``. The unified semantics (2026-08-01/02 reviews):

- ``condition`` and ``promptID`` must be non-empty JSON STRINGS — a numeric
  promptID used to be ``str()``-coerced here while the Swift decoder refused
  it, so the same file parsed on one engine and not the other.
- ``sampleIndex`` present = that exact pair cell, and must be a non-negative
  integer; ABSENT = an explicit WILDCARD matching every judged cell of its
  (condition, promptID) that no exact row claims (exact beats wildcard).
- duplicate keys refuse; an empty file refuses.

Consumed by ``tasks._load_human_validation`` (evaluation) and
``Manifest.verify`` (pin integrity) — the verifier and the evaluator read
the same rules by construction. Swift twin:
``ExperimentTasks.parseHumanValidation``.
"""

from __future__ import annotations

import json

OUTCOMES = ("baseline", "variant", "tie")


def parse_rows(data: bytes, label: str) -> dict[tuple[str, str | None, str], str]:
    """``{(promptID, sampleIndex-as-str | None, condition): outcome}`` or a
    ``RuntimeError`` naming the offending line."""
    rows: dict[tuple[str, str | None, str], str] = {}
    for i, raw in enumerate(data.decode("utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"human validation '{label}' line {i + 1}: not valid JSON "
                f"({exc})") from exc
        if not isinstance(obj, dict):
            raise RuntimeError(
                f"human validation '{label}' line {i + 1}: not a JSON object")
        condition = obj.get("condition")
        prompt_id = obj.get("promptID")
        if not isinstance(condition, str) or not condition \
                or not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(
                f"human validation '{label}' line {i + 1}: condition and "
                f"promptID must be non-empty strings — a coerced numeric id "
                f"parses here and refuses on the Swift engine")
        outcome = obj.get("outcome")
        if outcome not in OUTCOMES:
            raise RuntimeError(
                f"human validation '{label}' line {i + 1}: outcome must be "
                f"{'|'.join(OUTCOMES)}, got {outcome!r}")
        sample = obj.get("sampleIndex")
        if sample is not None and (
                isinstance(sample, bool) or not isinstance(sample, int)
                or sample < 0):
            raise RuntimeError(
                f"human validation '{label}' line {i + 1}: sampleIndex "
                f"must be a non-negative integer when present, "
                f"got {sample!r}")
        key = (prompt_id, None if sample is None else str(sample), condition)
        if key in rows:
            raise RuntimeError(
                f"human validation '{label}' line {i + 1}: duplicate row "
                f"for {key} — one human judgment per pair cell")
        rows[key] = outcome
    if not rows:
        raise RuntimeError(f"human validation '{label}' has no labeled rows")
    return rows
