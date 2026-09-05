"""Declaration-only freeze rules; no workspace, model or evidence acquisition."""

from __future__ import annotations

import re

from .manifest_errors import ExperimentStoreError
from ..steering.vector_math import ExtractionMethod

_OPTVEC_METHOD = "optvec"

JUDGE_DTYPE_VOCABULARY = ("bfloat16", "float16", "float32")


_JUDGE_DTYPE_ALIASES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16",
    "float32": "float32", "fp32": "float32",
}


def model_output_surfaces_operative(d: dict) -> bool:
    """Whether the MODEL-OUTPUT freeze surfaces apply to this manifest.

    The one decision shared by readiness and freeze, so they cannot give
    opposite answers about the same manifest (external review round 13).

    A multi-agent study runs a SCENARIO. Under the app's never-delete rule it
    may carry concepts, injection conditions, agents, and a J-lens readout
    from before a kind switch — none of which it executes. `freeze_advisories`
    already tells the researcher that carried configuration is "preserved, but
    NOT verified, snapshotted, or bundled for this study kind"; the gates then
    verified it anyway and refused the freeze. Both statements came out of the
    same function.

    So carried model-output state may ADVISE, and may not block. What still
    applies to every kind — a panel loads a model and is judged like any other
    study — is deliberately outside this: pinned revision, loadable dtype,
    judge validity, and git cleanliness.
    """
    return (d.get("studyKind") or "modelOutput") == "modelOutput"


def optvec_pinned_concepts(d: dict) -> list[tuple[str, dict]]:
    """``[(concept name, vectorArtifact block)]`` for every concept pinned to
    an OptVec artifact. Pure manifest reading (no filesystem, no torch), so
    the freeze gates and advisories can both ask it."""
    out: list[tuple[str, dict]] = []
    for concept in d.get("concepts") or []:
        if not isinstance(concept, dict) or not concept.get("name"):
            continue
        if ((concept.get("options") or {}).get("method")
                != ExtractionMethod.PINNED_ARTIFACT.value):
            continue
        block = concept.get("vectorArtifact")
        if isinstance(block, dict) and block.get("sourceMethod") == _OPTVEC_METHOD:
            out.append((str(concept["name"]), block))
    return out


def optvec_exempt_from_validate_gate(d: dict) -> bool:
    """Whether the validate-evidence freeze gate has nothing to ask of this
    manifest because every concept it declares is an OptVec direction.

    THE RULE (OptVec plan §6, decided here): an optvec concept has nothing to
    validate. ``validate`` scores a held-out probe against the recipe's class
    means; an optvec vector has no stimuli, no classes and no
    validation.jsonl, so there is no probe to run and a validate run could
    never exist — gating on one would make an optvec confirm study
    freezable only under ``--force``, i.e. permanently non-citable, which
    would be a stamp about the FIREWALL rather than about the science. The
    evidence that certifies the direction is the OptVec eval run's
    ``eval.json`` (test split, untouched by gradients and by checkpoint
    selection), surfaced by :func:`freeze_advisories`.

    Deliberately narrow. It applies only when concepts exist and EVERY one is
    optvec-pinned: a mixed study still owes a validate run for its ordinary
    concepts, and a variant study still owes per-condition battery evidence
    (which is joined to the validate run), so both keep the gate.
    """
    concepts = [c for c in (d.get("concepts") or [])
                if isinstance(c, dict) and c.get("name")]
    if not concepts or d.get("variantConditions"):
        return False
    return len(optvec_pinned_concepts(d)) == len(concepts)


def _resolved_judge_identity(judge: dict, study_model: str) -> tuple[str, str, str]:
    """A judge's RESOLVED identity ``(kind, model, provider)`` — what will
    actually run, not what the manifest happens to spell. Cross-engine rules:
    a LOCAL judge with a blank model resolves to the STUDY model; a claude
    judge with a blank model resolves to the default Claude judge model;
    openrouter judges have no defaults (their own verify rules apply)."""
    kind = str(judge.get("kind") or "claude").strip() or "claude"
    model = str(judge.get("model") or "").strip()
    provider = str(judge.get("provider") or "").strip()
    if kind == "openrouter":
        from . import paired_judge
        provider = paired_judge.canonical_openrouter_provider(provider)
    if kind == "local" and not model:
        model = study_model
    elif kind == "claude" and not model:
        from . import paired_judge
        model = paired_judge.DEFAULT_JUDGE_MODEL
    return (kind, model, provider)


def judge_panel_indistinct_problem(d: dict) -> str | None:
    """External review 2026-07-22 (finding 4): two blank-model local judges
    both resolve to the study model at temperature 0 — identical
    deterministic judges whose perfect agreement is guaranteed by
    construction, satisfying a count-only panel gate while providing zero
    independence. Returns the plain-language problem (identical wording on
    both engines) when a panel of >= 2 named judges collapses to fewer than
    2 DISTINCT resolved identities, else None. Shared by the freeze gate
    (judgeValidity) and the pre-freeze advisories/data check."""
    judges = [j for j in (d.get("judges") or [])
              if isinstance(j, dict) and j.get("name")]
    if len(judges) < 2:
        return None
    study_model = str(d.get("modelID") or "")
    identities: dict[tuple[str, str, str], list[str]] = {}
    for judge in judges:
        identity = _resolved_judge_identity(judge, study_model)
        identities.setdefault(identity, []).append(str(judge["name"]))
    if len(identities) >= 2:
        return None
    (kind, model, provider), names = next(iter(identities.items()))
    quoted = [f"'{n}'" for n in names]
    joined = (" and ".join(quoted) if len(quoted) == 2
              else ", ".join(quoted[:-1]) + " and " + quoted[-1])
    quantifier = "both" if len(quoted) == 2 else "all"
    if kind == "local" and model == study_model:
        what = "the study model at temperature 0"
    elif provider:
        what = f"the {kind} judge '{model}' via '{provider}'"
    else:
        what = f"the {kind} judge '{model}'"
    return (f"judges {joined} {quantifier} resolve to the same deterministic "
            f"judge ({what}) — they would agree perfectly by construction; "
            "use judges with different models, kinds, or providers")


def _pipeline_stage_list(d: dict) -> list[str]:
    """The declared pipeline's stage list, or [] when no pipeline is
    declared or the block is malformed (malformed blocks have their own
    verify violations)."""
    block = d.get("pipeline")
    if not isinstance(block, dict):
        return []
    try:
        from .pipeline_spec import resolve_pipeline
        return list(resolve_pipeline(block).stages)
    except Exception:  # noqa: BLE001 - malformed pipeline refuses elsewhere
        return []


def normalize_judge_dtype(value: str | None) -> str | None:
    """Canonical spelling of a judge dtype alias, or None if unrecognized."""
    return _JUDGE_DTYPE_ALIASES.get((value or "").strip().lower())


def _is_commit_like(revision: str) -> bool:
    """Whether a revision names FIXED bytes rather than a moving ref.

    Hexadecimal — the shape of a git commit hash, full or abbreviated.
    Branch names (`main`, `master`), `HEAD`, `refs/...` paths, and
    conventional tags (`v1.0`, `latest`) all fail it, which is the point: a
    branch is re-pointed by definition, and a tag can be moved, so neither
    identifies the bytes a run used (external review round 5, finding 4).

    Honest residual: a tag whose name happens to be hexadecimal would pass.
    No format check can distinguish that from a short hash — only asking the
    hub could — and it is not a shape anyone tags in practice.
    """
    stripped = revision.strip()
    if not stripped:
        return False
    try:
        int(stripped, 16)
    except ValueError:
        return False
    return True


def symbolic_revision_problem(d: dict) -> str | None:
    """Revision pins that name a moving ref instead of a commit.

    Applies to the STUDY revision and to every local judge's. `"main"`
    passed the old gate — it only required non-emptiness — and the loader
    then recorded the symbolic name it was handed rather than the commit it
    resolved to, so two runs a week apart could record the same "pin" and
    have run different weights.

    Cross-engine wording (Swift twin:
    `ExperimentStore.symbolicRevisionProblem`).
    """
    offenders: list[str] = []
    study = str(d.get("modelRevision") or "").strip()
    if study and not _is_commit_like(study):
        offenders.append(f"the study model pins '{study}'")
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "openrouter").strip() or "openrouter") != "local":
            continue
        revision = str(judge.get("revision") or "").strip()
        if revision and not _is_commit_like(revision):
            offenders.append(f"judge '{judge['name']}' pins '{revision}'")
    if not offenders:
        return None
    return ("revision pin(s) name a moving reference rather than a commit: "
            + "; ".join(offenders)
            + ". A branch or tag is re-pointed by definition, so it cannot "
            "identify the weights a run used — two runs a week apart would "
            "record the same pin having loaded different bytes. Use the "
            "commit hash (the app's Resolve button reads it from whichever "
            "substrate will run the model)")


def study_model_judge_pin_conflict(d: dict) -> str | None:
    """A study-model local judge declaring pins that differ from the study's.

    Scoped to studies declaring a **judgeScore sweep** (external review
    round 5, finding 1). There, such a judge has no independent identity:
    the sweep judges with the already-HELD study model rather than loading
    anything, so a divergent `revision`/`dtype` is silently ignored while
    remaining in the criterion provenance.

    Deliberately NOT a blanket rule. `evaluate` genuinely LOADS a declared
    judge revision, so judging with a different checkpoint of the study repo
    is a legitimate design there — `_pin_local_judge_revisions` has always
    preserved a declared revision for exactly that reason. The defect is the
    SWEEP path silently ignoring what evaluate honors: one manifest, two
    identities, depending on the verb.

    Forbidding divergence is preferred over verify-and-stamp: it is
    checkable while authoring, rather than producing an artifact merely
    honest about having judged with something else. Pins that AGREE with the
    study stay legal — redundant, not wrong.

    Cross-engine wording (Swift twin:
    `ExperimentStore.studyModelJudgePinConflict`).
    """
    selection = (d.get("sweep") or {}).get("selection") \
        if isinstance(d.get("sweep"), dict) else None
    objective = (selection or {}).get("objective") \
        if isinstance(selection, dict) else None
    if ((objective or {}).get("metric")
            if isinstance(objective, dict) else None) != "judgeScore":
        return None
    study_model = str(d.get("modelID") or "")
    study_revision = str(d.get("modelRevision") or "").strip()
    study_dtype = str(d.get("dtype") or "").strip()
    offenders: list[str] = []
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "openrouter").strip() or "openrouter") != "local":
            continue
        declared = str(judge.get("model") or "").strip()
        # Blank model AND explicit study model both resolve to the study
        # model (`sweep_selection.resolve_local_judge_model`).
        if declared and declared != study_model:
            continue
        revision = str(judge.get("revision") or "").strip()
        if revision and revision != study_revision:
            offenders.append(
                f"'{judge['name']}' pins revision '{revision}' but the study "
                + (f"is pinned at '{study_revision}'" if study_revision
                   else "has no revision pinned"))
        dtype = str(judge.get("dtype") or "").strip()
        if dtype and normalize_judge_dtype(dtype) != \
                normalize_judge_dtype(study_dtype):
            offenders.append(
                f"'{judge['name']}' pins dtype '{dtype}' but the study "
                + (f"is pinned at '{study_dtype}'" if study_dtype
                   else "pins none (the device decides)"))
    if not offenders:
        return None
    return ("this study selects on judgeScore, and local judge(s) resolving "
            "to the STUDY model cannot pin a different identity: " + "; ".join(offenders)
            + ". Such a judge IS the study model — a sweep judges with the "
            "already-held weights and never loads anything else, so the "
            "divergent pin would be silently ignored. Drop the pin to "
            "inherit the study's, or name a different model to make it a "
            "genuinely separate judge")


def unloadable_study_dtype_problem(d: dict) -> str | None:
    """A study-level `dtype` outside the closed vocabulary.

    The Mac is the AUTHORING surface and the cluster is the measurement one,
    so this is validated here even though only the server consumes the key —
    a manifest must not reach the cluster carrying a dtype that refuses at load
    after a queue wait. Cross-engine wording (Swift twin:
    `ExperimentStore.unloadableStudyDtypeProblem`).
    """
    spelled = str(d.get("dtype") or "").strip()
    if not spelled or normalize_judge_dtype(spelled) is not None:
        return None
    return (f"study dtype '{spelled}' is not one this engine can load — the "
            "loader accepts only " + ", ".join(JUDGE_DTYPE_VOCABULARY)
            + " (aliases bf16/fp16/fp32). Leave it unset to let the device "
            "decide, which is what every study did before this pin existed")


def unpinned_foreign_local_judge_problem(d: dict) -> str | None:
    """Foreign local judges whose model bytes are not pinned.

    A local judge resolving to the STUDY model inherits the study's pinned
    revision, so "the same judge" across two sessions is a fact. A local
    judge naming a DIFFERENT model has no such pin to inherit — and freeze
    deliberately leaves its revision blank (`_pin_local_judge_revisions`).
    That was tolerable while a judgment artifact merely RECORDED what
    loaded, but targeted retry compares recorded identities to decide
    whether verdicts from an earlier session may be REUSED: two sessions
    can each load a different default revision while both records say
    ``null``, and null == null passes (external review round 2, finding 3).

    Requiring the pin is the cheaper of the two fixes and matches the
    pin-everything discipline everywhere else. `dtype` is required with it
    on this engine because the loader takes one and a bf16-vs-fp16 judge is
    a different judge.

    Returns the plain-language problem, or None when every foreign local
    judge is pinned. Cross-engine wording.
    """
    study_model = str(d.get("modelID") or "")
    offenders: list[str] = []
    unknown: list[str] = []
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "openrouter").strip() or "openrouter") != "local":
            continue
        # A dtype OUTSIDE the closed vocabulary is checked for every local
        # judge, pinned or not: the loader refuses it at run time, and
        # discovering that on a compute node after a queue wait is exactly
        # the failure this firewall exists to move forward in time
        # (external review round 4, finding 2).
        spelled = str(judge.get("dtype") or "").strip()
        if spelled and normalize_judge_dtype(spelled) is None:
            unknown.append(f"'{judge['name']}' declares dtype '{spelled}'")
        declared = str(judge.get("model") or "").strip()
        if not declared or declared == study_model:
            continue
        missing = [field for field in ("revision", "dtype")
                   if not str(judge.get(field) or "").strip()]
        if missing:
            offenders.append(
                f"'{judge['name']}' (model '{declared}') is missing "
                + " and ".join(missing))
    if unknown:
        return ("local judge(s) declare a dtype this engine cannot load: "
                + "; ".join(unknown) + ". The loader accepts only "
                + ", ".join(JUDGE_DTYPE_VOCABULARY)
                + " (aliases bf16/fp16/fp32). An unrecognized value used to "
                "load float32 silently, so the pin would be a false claim")
    if not offenders:
        return None
    return ("local judge(s) naming a model other than the study model must "
            "pin the exact bytes that will judge: " + "; ".join(offenders)
            + ". Without a revision pin two judging sessions can load "
            "different defaults while both records say 'none', so a "
            "resumed evaluation cannot prove its reused verdicts came from "
            "the same judge. Pin judges[].revision and judges[].dtype, or "
            "use the study model as judge")


def _foreign_local_judges(d: dict) -> list[str]:
    """Local judges whose declared model differs from the study model,
    rendered ``'name' (model 'id')``."""
    study_model = str(d.get("modelID") or "")
    offenders: list[str] = []
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "claude").strip() or "claude") != "local":
            continue
        declared = str(judge.get("model") or "").strip()
        if declared and declared != study_model:
            offenders.append(f"'{judge['name']}' (model '{declared}')")
    return offenders


def local_judge_pipeline_problem(d: dict) -> str | None:
    """Finding 1 gate, fan-out era (2026-07-23): a judged SWEEP inside the
    chain still cannot use a local judge whose model differs from the study
    model — sweep judging is interleaved with the selection, not a
    separable post-stage, so no fan-out exists for it. The EVALUATE stage
    no longer refuses (it routes to the post-generation judge fan-out —
    see :func:`local_judge_fanout_note`). Returns the sweep problem text,
    or None."""
    stages = _pipeline_stage_list(d)
    if "sweep" not in stages:
        return None
    selection = (d.get("sweep") or {}).get("selection") \
        if isinstance(d.get("sweep"), dict) else None
    objective = (selection or {}).get("objective") \
        if isinstance(selection, dict) else None
    metric = (objective or {}).get("metric") \
        if isinstance(objective, dict) else None
    if metric != "judgeScore":
        return None
    offenders = _foreign_local_judges(d)
    if not offenders:
        return None
    study_model = str(d.get("modelID") or "")
    return ("the declared pipeline's sweep stage holds ONE model — the "
            f"study model '{study_model}' — but local judge(s) "
            + ", ".join(offenders) + " resolve to a different model, which "
            "cannot load inside the chain (the judge fan-out covers the "
            "evaluate stage only). Leave a local judge's model empty to "
            "judge with the study model, pin claude/openrouter judges, or "
            "select on logprobShift")


def local_judge_fanout_note(d: dict) -> str | None:
    """Routing information (never a gate, 2026-07-23): a declared pipeline
    whose EVALUATE stage pins local judges resolving to models other than
    the study model will judge them as a post-generation fan-out — the
    chain emits blinded packets, one worker job per distinct judge model
    judges them, and the merge resumes the chain. Available on Slurm
    run-first pipeline submissions; elsewhere the packets await deferred
    (Mac) judging. Returns the note, or None."""
    stages = _pipeline_stage_list(d)
    if "evaluate" not in stages:
        return None
    offenders = _foreign_local_judges(d)
    if not offenders:
        return None
    return ("the pipeline's evaluate stage will judge local judge(s) "
            + ", ".join(offenders) + " as a post-generation judge fan-out "
            "(one worker job per distinct judge model; available on Slurm "
            "run-first pipeline submissions — elsewhere the emitted packets "
            "await deferred judging)")


def _pin_local_judge_revisions(d: dict) -> None:
    """Freeze-time pin for LOCAL judge revisions (cross-engine contract key
    ``judges[].revision``, 2026-07-23, omit-when-nil): a local judge that
    resolves to the STUDY model inherits the study's pinned revision when
    its own is blank — the judging path then loads exactly the pinned
    bytes. A local judge declaring a DIFFERENT model keeps its blank
    revision (there is no study pin to inherit; the judgment artifact
    stamps what actually loaded). Never overwrites a declared revision."""
    study_revision = d.get("modelRevision")
    if not study_revision:
        return
    study_model = str(d.get("modelID") or "")
    for judge in (d.get("judges") or []):
        if not isinstance(judge, dict):
            continue
        if (str(judge.get("kind") or "claude").strip() or "claude") != "local":
            continue
        if judge.get("revision"):
            continue
        declared = str(judge.get("model") or "").strip()
        if not declared or declared == study_model:
            judge["revision"] = study_revision


def _no_judge_declared_reason(name: str) -> str:
    """The ``judgeValidity`` refusal for a judged study with NO judge — the
    state the panel-size rule actually protects against. Swift twin:
    ``ExperimentStore.noJudgeDeclaredReason``; the sentence is the
    contract."""
    return ("judge-evaluated study pins no judge — a judged instrument with "
            "no judge codes nothing; pin a panel: 'steerlab-cli experiment "
            f"pin-rubric {name} <rubric> --judges <name>:<kind>[,…]'. Or "
            "freeze --force")


def single_judge_panel_advisory(d: dict) -> str | None:
    """The consequence of a one-judge panel, stated rather than forbidden.

    A one-judge panel used to be refused at freeze (``judgeValidity``
    required >= 2 so the report could carry agreement statistics). The
    maintainer's ruling is that a researcher may declare any number of
    judges including exactly one, so freeze accepts it and this says what it
    costs. None for a panel of two or more, and for a study that is not
    judged at all. Swift twin:
    ``ExperimentStore.singleJudgePanelAdvisory``."""
    judges = [j for j in (d.get("judges") or [])
              if isinstance(j, dict) and j.get("name")]
    evaluation = d.get("evaluation") or {}
    if evaluation.get("kind") != "pairedJudge" and not judges:
        return None
    return SINGLE_JUDGE_PANEL_ADVISORY if len(judges) == 1 else None


def _check_judged_evaluation(name: str, d: dict) -> None:
    """Judged studies need a versioned criterion and a real panel (evidence
    tier): a pairedJudge evaluation must pin its rubric as a hashed FILE
    (prompts/rubrics/) and declare at least ONE judge; a panel of two or
    more must have DISTINCT resolved identities, or inter-judge agreement —
    the check that the criterion measures anything — is trivially perfect.
    A one-judge panel freezes cleanly and carries
    ``single_judge_panel_advisory`` instead (maintainer ruling, 2026-08-28).
    ``freeze --force`` skips this loudly, like the other evidence gates."""
    evaluation = d.get("evaluation") or {}
    # Judge-evaluated = a pairedJudge evaluation OR an explicit judges panel
    # (Swift's rule exactly — declaring judges intends judged evaluation, and
    # the two engines' gates must agree or a manifest freezes on one engine
    # and not the other).
    if evaluation.get("kind") != "pairedJudge" and not d.get("judges"):
        return
    if not (d.get("judgeRubricFile") and d.get("judgeRubricHash")):
        raise ExperimentStoreError(
            f"cannot freeze '{name}': evaluation uses pairedJudge but no judge "
            "rubric is pinned — set judgeRubricFile + judgeRubricHash "
            "(rubrics live in prompts/rubrics/), or freeze --force")
    judges = [j for j in (d.get("judges") or [])
              if isinstance(j, dict) and j.get("name")]
    # ONE judge is a legal design (maintainer ruling, 2026-08-28): a
    # single-coder study is a real methodology, and the gate's job is to
    # refuse the INVALID state — a judged instrument with no judge — not to
    # legislate the panel size. What the >= 2 rule protected (inter-rater
    # agreement) survives as the non-blocking
    # ``single_judge_panel_advisory``. Swift twin:
    # ``checkJudgeEvaluationValidity`` / ``noJudgeDeclaredReason``.
    if not judges:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': " + _no_judge_declared_reason(name))
    indistinct = judge_panel_indistinct_problem(d)
    if indistinct:
        raise ExperimentStoreError(f"cannot freeze '{name}': {indistinct}")
    pipeline_problem = local_judge_pipeline_problem(d)
    if pipeline_problem:
        raise ExperimentStoreError(f"cannot freeze '{name}': {pipeline_problem}")
    unpinned = unpinned_foreign_local_judge_problem(d)
    if unpinned:
        raise ExperimentStoreError(f"cannot freeze '{name}': {unpinned}")
    study_model_conflict = study_model_judge_pin_conflict(d)
    if study_model_conflict:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': {study_model_conflict}")


SINGLE_JUDGE_PANEL_ADVISORY = (
    "single-coder design: this study pins 1 judge, so no inter-rater "
    "agreement statistics (percent agreement, Cohen's kappa) will exist for "
    "its codings — the coding report records fieldAgreement as absent with "
    "that reason rather than empty")
