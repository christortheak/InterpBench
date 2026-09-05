"""Resolve pinned judging rubrics and report actionable missing-rubric refusals.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
from . import lifecycle_gates, paths
from . import manifest as manifest_module


def no_rubric_refusal(name: str) -> str:
    """The refusal both engines give an evaluation with no rubric at all.

    Byte-identical to Swift's ``JudgeRubricStore.noRubricRefusal``, and it
    names ``steerlab-cli`` on BOTH engines on purpose: authoring is
    Mac-authority (WP0 §10.x), and this CLI has no ``pin-rubric`` verb to
    name."""
    return (f"study '{name}' has no judge rubric — pin one: "
            f"'steerlab-cli experiment pin-rubric {name} "
            "prompts/rubrics/default-paired-v1.md' (any file under "
            "prompts/rubrics/; inline draft text is draft-only and cannot "
            "freeze)")


def no_rubric_repair(name: str) -> str:
    """The repair for :func:`no_rubric_refusal` on THIS engine: pin on the
    Mac (the only CLI with a ``pin-rubric`` verb), then re-run here."""
    return (f"steerlab-cli experiment pin-rubric {name} "
            "prompts/rubrics/default-paired-v1.md  (authoring is "
            f"Mac-authority) ; then steerlab-server experiment evaluate {name}")


def missing_rubric_refusal(path: str) -> str:
    """The refusal for a named rubric path that is not on disk.

    Byte-identical to Swift's ``JudgeRubricStore.missingRubricRefusal``. Both
    engines used to let the filesystem error out unhandled here — Swift printed
    an ``NSCocoaErrorDomain`` dump, this engine a ``FileNotFoundError``
    traceback and exit 1 (observed 2026-08-18)."""
    return f"judge rubric file not found: {path}"


def missing_rubric_repair(name: str, relative: str) -> str:
    """The repair on THIS engine: author the file under the convention
    directory, pin it on the Mac (authoring is Mac-authority — this CLI has no
    ``pin-rubric`` verb), then re-run here. The Swift twin
    (``JudgeRubricStore.missingRubricRepair``) is the same sentence without the
    Mac-authority note and the server re-run, exactly as
    :func:`no_rubric_repair` differs from Swift's."""
    default = "prompts/rubrics/default-paired-v1.md"
    # When the ABSENT path IS the shipped default, naming it as an example
    # would be the repair pointing at itself (Swift does the same).
    example = (f" (a seeded workspace ships {default})" if relative == default
               else f" (the shipped {default} is one)")
    return (f"author {relative} under prompts/rubrics/{example}, then "
            f"steerlab-cli experiment pin-rubric {name} {relative}  "
            f"(authoring is Mac-authority) ; then steerlab-server experiment "
            f"evaluate {name}")


def resolve_rubric(manifest: manifest_module.Manifest, root, _log) -> tuple[str, str | None, str | None]:
    """The rubric text evaluate judges with: ``(text, sha256|None, file|None)``.

    Frozen studies MUST judge from the pinned rubric file (never an unpinned
    inline string); a draft may fall back to the manifest's inline
    ``evaluation.judgePrompt`` with a loud warning. Drift between the file and
    its pin is refused at read time, like task prompts.

    An EMPTY inline fallback is refused, not warned about (WP0 dry run #2's
    skipped check, Christian-flagged): a draft study with an evaluation block
    and no rubric anywhere used to reach the judges with the empty string as
    its rubric — this engine emitted twelve blinded judging packets built on
    nothing and exited 0. Swift has always refused exactly this input
    (``JudgeRubricStore.resolveRubric``); the sentence is now the same one."""
    if manifest.judge_rubric_file:
        path = paths.resolve(manifest.judge_rubric_file, root)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except FileNotFoundError:
            raise lifecycle_gates.refusing(
                lifecycle_gates.MISSING_PREREQUISITE,
                missing_rubric_refusal(str(path)),
                repair=missing_rubric_repair(
                    manifest.name, manifest.judge_rubric_file)) from None
        live = hashlib.sha256(data).hexdigest()
        if manifest.judge_rubric_hash and live != manifest.judge_rubric_hash:
            raise RuntimeError(
                f"judge rubric '{manifest.judge_rubric_file}' drifted from the "
                f"pinned hash (have {live[:12]}…, pinned "
                f"{manifest.judge_rubric_hash[:12]}…)")
        if not manifest.judge_rubric_hash and manifest.status == "frozen":
            raise RuntimeError(
                "frozen study names a judge rubric file without judgeRubricHash "
                "— duplicate, pin both, and re-freeze")
        return data.decode("utf-8"), live, manifest.judge_rubric_file
    if manifest.status == "frozen":
        raise RuntimeError(
            f"frozen study '{manifest.name}' has no pinned judge rubric — a "
            "frozen evaluation must judge from a hashed rubric FILE "
            "(judgeRubricFile + judgeRubricHash; see prompts/rubrics/). "
            "Duplicate, pin one, and re-freeze")
    inline = (manifest.evaluation.judge_prompt if manifest.evaluation
              else "") or ""
    if not inline.strip():
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            no_rubric_refusal(manifest.name),
            repair=no_rubric_repair(manifest.name))
    _log("WARNING: judging with an UNPINNED inline rubric (draft study) — pin "
         "a rubric file under prompts/rubrics/ before freezing")
    return inline, None, None
