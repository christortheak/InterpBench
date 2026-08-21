"""Where a panel's judging will ACTUALLY happen — decided in advance, from
the manifest and this host's credentials alone.

Extracted from :mod:`tasks` (2026-08-20) with no behavior change: the
decision is needed by three callers that must not pay for ``import torch``
— the freeze advisory (``experiment_store.judging_custody_advisory``), the
submission preflight's walltime estimator (open issues §4: an evaluate whose
judging DEFERS renders packets and parks, so pricing it at generation speed
refused a ten-minute job as an eleven-hour one), and ``tasks`` itself, which
re-exports these names under their historical spellings.

Nothing here touches the filesystem beyond the judge key file's presence
check, loads a model, or has side effects.
"""

from __future__ import annotations

from .manifest import JudgeRef


#: The judge kinds that need an external credential (the manifest's
#: vocabulary; ``local`` is the third and never touches a key).
EXTERNAL_KINDS = ("claude", "openrouter")


def missing_external_credentials(roster) -> set[str]:
    """External judge kinds in ``roster`` this host holds no credential for.
    A malformed key file counts as CREDENTIALED (fail loud at judge time,
    never silently defer) — ``judge_credentials.available``'s rule."""
    from . import judge_credentials
    missing: set[str] = set()
    for ref in roster:
        if ref.kind in EXTERNAL_KINDS \
                and not judge_credentials.available(ref.kind):
            missing.add(ref.kind)
    return missing


def custody_plan(roster) -> dict:
    """Where this panel's judging will ACTUALLY happen, decided in advance.

    The inline/deferred fork turns on whether THIS host holds a credential
    for each external judge kind — and until 2026-07-24 nothing said which
    way it had gone until it had already gone. The confusing case is a
    mixed panel: the key file holds ONE kind, so a panel of one claude and
    one openrouter judge with an openrouter key credentials half the panel
    and defers the whole thing, despite a key having been deliberately
    pushed.

    Returns ``{"disposition", "reason", "missingKinds", "externalJudges",
    "localJudges"}`` where disposition is:

    - ``"local"``      — no external judges; nothing to defer.
    - ``"inline"``     — every external judge is credentialed here.
    - ``"deferred"``   — at least one is not; the panel becomes packets.
    - ``"refused"``    — a split local/external panel that cannot defer
                         coherently (two judging clocks, one report).
    """
    external = [r for r in roster if r.kind in EXTERNAL_KINDS]
    local = [r for r in roster if r.kind == "local"]
    missing = sorted(missing_external_credentials(roster))
    if not external:
        return {"disposition": "local", "reason": "no external judges",
                "missingKinds": [], "externalJudges": [],
                "localJudges": [r.name for r in local]}
    names = [f"'{r.name}' ({r.kind})" for r in external]
    if not missing:
        return {
            "disposition": "inline",
            "reason": f"this host holds a credential for every external "
                      f"judge ({', '.join(names)})",
            "missingKinds": [], "externalJudges": [r.name for r in external],
            "localJudges": [r.name for r in local]}
    uncredentialed = [f"'{r.name}' ({r.kind})" for r in external
                      if r.kind in missing]
    if local:
        return {
            "disposition": "refused",
            "reason": f"panel mixes local judges with uncredentialed "
                      f"{'/'.join(missing)} judges "
                      f"({', '.join(uncredentialed)}) — a split panel "
                      "cannot defer coherently",
            "missingKinds": missing,
            "externalJudges": [r.name for r in external],
            "localJudges": [r.name for r in local]}
    return {
        "disposition": "deferred",
        "reason": f"no {'/'.join(missing)} credential on this host for "
                  f"{', '.join(uncredentialed)} — the whole panel defers to "
                  "the Mac, including any judge that IS credentialed here",
        "missingKinds": missing,
        "externalJudges": [r.name for r in external],
        "localJudges": [r.name for r in local]}


def roster_from_judge_entries(entries) -> list[JudgeRef]:
    """The pinned panel as ``JudgeRef``s, from raw manifest ``judges`` rows.
    Same defaulting as ``Manifest.from_dict`` (an entry with no kind is an
    openrouter judge); rows without a name are not a panel member."""
    return [JudgeRef(name=str(j.get("name")),
                     kind=str(j.get("kind") or "openrouter"),
                     model=j.get("model"), provider=j.get("provider"))
            for j in (entries or [])
            if isinstance(j, dict) and j.get("name")]


def roster_for_manifest(manifest) -> list[JudgeRef]:
    """The panel an evaluate of ``manifest`` would judge with: the pinned
    ``judges`` when present, else the legacy single judge derived from
    ``evaluation.judgeModel`` (draft convenience — freeze requires ≥ 2).

    The engine twin is ``tasks._judge_roster``, which resolves the same two
    sources from an already-loaded evaluation spec; this one reads the
    manifest so a preflight can ask before any task machinery exists.
    """
    if getattr(manifest, "judges", None):
        return list(manifest.judges)
    from . import paired_judge
    spec = getattr(manifest, "evaluation", None)
    model = (getattr(spec, "judge_model", "") or "").strip() \
        or paired_judge.DEFAULT_JUDGE_MODEL
    kind = "claude" if paired_judge.is_claude_model(model) else "local"
    return [JudgeRef(name=model, kind=kind, model=model)]
