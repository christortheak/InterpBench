"""Judgment identity, provenance, retry admission and human-validation evidence.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
from . import paths
from . import manifest as manifest_module


def normalized_judge_entries(raw, study_model: str | None = None) -> list[dict]:
    """Judge entries with kind and model RESOLVED (engineer review
    2026-07-18, second pass): kind defaults to claude, and an empty model
    pins the server's DEFAULT_JUDGE_MODEL at emission — the judging client
    must never resolve against its own ambient default. OpenRouter judges
    (2026-07-19) have NO defaults to fill: an explicit model slug and a
    pinned provider are required — normalization refuses rather than
    inventing either. LOCAL judges (judge fan-out, 2026-07-23): a blank
    model resolves to ``study_model`` when the caller provides it — the
    cross-engine local-judge rule — so worker judgments verify against the
    emission pin."""
    from . import paired_judge
    out: list[dict] = []
    for j in raw or ():
        entry = dict(j or {})
        entry["kind"] = entry.get("kind") or "claude"
        if entry["kind"] == "local" and study_model:
            entry["model"] = (str(entry.get("model") or "").strip()
                              or study_model)
            out.append(entry)
            continue
        if entry["kind"] == "openrouter":
            model = str(entry.get("model") or "").strip()
            provider = str(entry.get("provider") or "").strip()
            if not model:
                raise ValueError(
                    f"openrouter judge '{entry.get('name')}' has no model "
                    "slug — there is no default to pin at emission")
            if not provider:
                raise ValueError(
                    f"openrouter judge '{entry.get('name')}' has no pinned "
                    "provider — an unpinned provider is not a pinned judge")
            entry["model"], entry["provider"] = model, provider
            out.append(entry)
            continue
        entry["model"] = (str(entry.get("model") or "").strip()
                          or paired_judge.DEFAULT_JUDGE_MODEL)
        out.append(entry)
    return out


def judgment_key(row: dict) -> tuple[str, str, str]:
    """Pair-cell identity for aligning outcomes across judges (and against a
    human-labeled subset): the same id-keyed shape judge output rows carry.
    Keyed on ``sampleIndex`` (absent normalizes to 0) — the pairing join key
    — never the seed, which differs between the two sides of a pair under
    derived seeding."""
    return (str(row.get("promptID")), str(row.get("sampleIndex") or 0),
            str(row.get("condition")))


def verify_judgment_provider(row: dict, judge: str,
                              pinned_provider: str | None) -> str | None:
    """Per-judgment provider verification for BOTH completion verbs
    (engineer review 2026-07-18, provider-evidence pass): an openrouter
    judge's serving provider is emission-pinned exactly like its model, so
    every judgment must carry it and match it — a recorded string is
    provenance only if verified. A non-openrouter judgment claiming a
    provider is a mix-up, refused rather than shrugged past."""
    from . import paired_judge
    provider = str(row.get("provider") or "").strip()
    if pinned_provider is not None:
        if not provider:
            raise ValueError(
                f"judgment by {judge!r} carries no provider — the judging "
                "client must stamp the emission-pinned OpenRouter provider "
                "it verified against the response (update the app)")
        if (paired_judge.canonical_openrouter_provider(provider)
                != paired_judge.canonical_openrouter_provider(pinned_provider)):
            raise ValueError(
                f"judge {judge!r} judged via provider '{provider}' but the "
                f"emission pinned '{pinned_provider}' — refusing off-pin "
                "judgments")
        return paired_judge.canonical_openrouter_provider(provider)
    if provider:
        raise ValueError(
            f"judgment by {judge!r} carries provider '{provider}' but "
            "that judge is not openrouter-kind — only provider-pinned "
            "judges stamp one")
    return None


def verified_judgment_payload(row: dict, judge: str,
                               winner: str) -> tuple[float | None,
                                                     dict | None]:
    """``(confidence, full verdict payload)`` from one client judgment row —
    the winner-only closure (2026-07-20, both completion verbs): deferred
    judgments may now carry the judge's FULL verdict (scores, structured
    fields, confidence, brief reason — the same object the inline path
    records as ``judgment``), so deferred artifacts stop being
    winner-only. The payload is OPTIONAL (older judging clients send
    winner-only rows; completion still succeeds and records
    ``judgment: null``), but a present payload must be a JSON object whose
    own ``winner`` agrees with the row's verified winner — a verdict that
    contradicts the winner it supposedly explains is refused, never
    recorded (a recorded payload is provenance only if verified)."""
    def _as_confidence(value) -> float | None:
        return (float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool) else None)

    confidence = _as_confidence(row.get("confidence"))
    payload = row.get("judgment")
    if payload is None:
        return confidence, None
    if not isinstance(payload, dict):
        raise ValueError(
            f"judgment by {judge!r} carries a non-object verdict payload — "
            "refusing")
    if payload.get("winner") != winner:
        raise ValueError(
            f"judgment by {judge!r} carries a verdict payload whose winner "
            f"({payload.get('winner')!r}) contradicts the judged winner "
            f"({winner!r}) — refusing an inconsistent verdict")
    if confidence is None:
        confidence = _as_confidence(payload.get("confidence"))
    return confidence, payload


def agreement_entries(labeled: list[tuple[str, dict]]) -> list[dict]:
    """Percent agreement + Cohen's kappa for every judge pair, over the pair
    cells both judged. ``labeled`` is ``[(judge_name, {key: outcome})]``."""
    from . import study_stats
    entries: list[dict] = []
    for i in range(len(labeled)):
        for j in range(i + 1, len(labeled)):
            name_a, map_a = labeled[i]
            name_b, map_b = labeled[j]
            keys = sorted(set(map_a) & set(map_b))
            if not keys:
                continue
            a = [map_a[key] for key in keys]
            b = [map_b[key] for key in keys]
            entries.append({"judges": [name_a, name_b], "n": len(keys),
                            "percentAgreement": study_stats.percent_agreement(a, b),
                            "kappa": study_stats.cohens_kappa(a, b)})
    return entries


def load_human_validation(
        manifest: manifest_module.Manifest, root) -> dict[tuple[str, str | None, str], str]:
    """The pinned human-labeled subset, hash-checked then parsed through the
    ONE row parser (``human_validation.parse_rows`` — the same rules
    ``Manifest.verify`` applies, review 2026-08-02). See that module for the
    cross-engine row semantics; Swift twin:
    ``ExperimentTasks.parseHumanValidation`` + ``humanAgreement``."""
    from . import human_validation
    hv = manifest.human_validation
    path = paths.resolve(hv.path, root)
    with open(path, "rb") as handle:
        data = handle.read()
    live = hashlib.sha256(data).hexdigest()
    if live != hv.hash:
        raise RuntimeError(
            f"human validation '{hv.path}' drifted from the pinned hash "
            f"(have {live[:12]}…, pinned {hv.hash[:12]}…)")
    return human_validation.parse_rows(data, hv.path)


def materialize_human_validation(
        human: dict[tuple[str, str | None, str], str],
        outcome_maps: list[tuple[str, dict]]) -> dict[tuple[str, str, str], str]:
    """Resolve wildcard rows against the cells the judges actually judged,
    exact rows first — the agreement join is a key-set intersection, so the
    wildcard must be expanded to concrete keys before it can match."""
    resolved: dict[tuple[str, str, str], str] = {
        key: outcome for key, outcome in human.items() if key[1] is not None}
    judged: set[tuple[str, str, str]] = set()
    for _, outcome_map in outcome_maps:
        judged.update(outcome_map)
    for key in judged:
        prompt_id, _, condition = key
        if key not in resolved and (prompt_id, None, condition) in human:
            resolved[key] = human[(prompt_id, None, condition)]
    return resolved


#: What a partial evaluate run was judged UNDER. Written before the first
#: judge call so a later targeted retry can prove it is completing the same
#: evaluation rather than merging two different ones.
JUDGING_CONTEXT_FILENAME = "judging-context.json"


def judging_context(manifest, spec, run_dir: str, rubric_hash: str | None,
                     rubric_file: str | None, roster) -> dict:
    """The pins a resumed evaluation must match to reuse judgments.

    Every field here is something that, if it changed, would make an
    earlier verdict an answer to a DIFFERENT question. The bar is the
    deferred path's, which pins the source generations by content
    (``sourceGenerationsSha256``) — retry pinning less than deferred would
    be backwards, since retry is precisely the case where time has passed
    (external review 2026-07-24, finding 1).

    Judges are recorded by RESOLVED identity, not as spelled: a local judge
    with a blank model resolves to the study model, so comparing the raw
    spelling would refuse a resume that is in fact identical — and, worse,
    would ACCEPT one where the study model changed underneath a blank.
    ``revision`` and ``dtype`` ride along because a foreign local judge
    reloaded at a different revision is a different judge, whatever its
    name says.
    """
    from . import paired_judge, sweep_selection
    structured = getattr(spec, "structured_prompt", None)
    generations = os.path.join(run_dir, "generations.jsonl")
    try:
        with open(generations, "rb") as handle:
            source_sha = hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        # Refuse, never pin None: the run this context describes must exist
        # to be judged, and a None hash made the resume pin VACUOUS — the
        # resume equality check compared None == None and passed, so two
        # evaluations of two unreadable (possibly different) source runs
        # "matched". The deferred path already refuses by raising on open;
        # retry must not pin less than deferred.
        raise RuntimeError(
            f"cannot judge run '{os.path.basename(run_dir.rstrip(os.sep))}': "
            f"its generations.jsonl cannot be read ({type(exc).__name__}: "
            f"{exc}) — the source generations must exist to be judged, and "
            "without their hash this evaluation could never prove what it "
            "judged") from exc
    judges = []
    for r in roster:
        resolved = r.model
        revision = r.revision
        if r.kind == "local":
            resolved = sweep_selection.resolve_local_judge_model(
                r.model, manifest.model_id)
            # A study-model judge inherits the study's pinned revision —
            # the same fallback the loader applies.
            if not revision and resolved == manifest.model_id:
                revision = manifest.model_revision
        judges.append({
            "name": r.name, "kind": r.kind, "model": resolved,
            "revision": revision if r.kind == "local" else None,
            "dtype": r.dtype if r.kind == "local" else None,
            "provider": (paired_judge.canonical_openrouter_provider(r.provider)
                         if r.kind == "openrouter" else None),
        })
    return {
        "schemaVersion": 2,
        "experiment": manifest.name,
        "experimentHash": manifest.content_hash(),
        "sourceRun": os.path.basename(run_dir.rstrip(os.sep)),
        # The directory NAME only says which run; the hash says which
        # BYTES. Runs are immutable by convention, but retention now writes
        # into run directories, so "nobody touches a run" is no longer a
        # thing to rest an evidence claim on.
        "sourceGenerationsSha256": source_sha,
        "rubricFile": rubric_file,
        "rubricHash": rubric_hash,
        "structuredPromptSha256": (
            hashlib.sha256(structured.encode("utf-8")).hexdigest()
            if structured else None),
        "judges": judges,
    }


def load_resumable_judgments(name: str, resume_from: str, root: str | None,
                              context: dict, log) -> dict:
    """Judgments from a partial evaluate run, keyed ``(judge, cell)``.

    Every pin is checked before a single row is reused. Reuse is the whole
    point of a targeted retry — judging is the expensive, non-deterministic
    step — but reusing a verdict produced under a DIFFERENT rubric, a
    different manifest epoch, a different source run, or by a differently
    configured judge would silently merge two experiments into one table.
    Each of those is a refusal, naming what differs.
    """
    runs = paths.runs_directory(root)
    if not resume_from or "/" in resume_from or os.sep in resume_from \
            or resume_from in (".", ".."):
        raise RuntimeError(f"invalid resume run id: {resume_from!r}")
    partial = os.path.join(runs, resume_from)
    if not os.path.isdir(partial):
        raise RuntimeError(f"no run directory '{resume_from}' to resume from")
    if os.path.exists(os.path.join(partial, "judge-report.json")):
        raise RuntimeError(
            f"'{resume_from}' is a COMPLETED evaluation (it has a "
            "judge-report.json) — there is nothing to retry. Analyze it, or "
            "run a fresh evaluate")
    context_path = os.path.join(partial, JUDGING_CONTEXT_FILENAME)
    try:
        with open(context_path, encoding="utf-8") as handle:
            before = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"'{resume_from}' carries no readable {JUDGING_CONTEXT_FILENAME} "
            f"({type(exc).__name__}) — it predates targeted retry, so what it "
            "was judged under cannot be proven. Re-run evaluate from the "
            "source run instead of resuming") from exc

    if int(before.get("schemaVersion") or 1) < 2:
        # A schema-1 context pinned the source run by NAME only and carried
        # no judge revision/dtype, so it cannot prove the things a resume
        # now has to prove. Refuse rather than silently applying weaker
        # rules to older evidence.
        raise RuntimeError(
            f"cannot resume '{resume_from}': its {JUDGING_CONTEXT_FILENAME} "
            "predates the strengthened pins (no source-generations hash, no "
            "judge revision) — it cannot prove those judgments answer the "
            "same question. Run a fresh evaluate")
    for key, label in (("experimentHash", "experiment epoch"),
                       ("sourceRun", "source run"),
                       ("sourceGenerationsSha256", "source generations"),
                       ("rubricHash", "rubric"),
                       ("structuredPromptSha256", "structured prompt")):
        if before.get(key) != context.get(key):
            raise RuntimeError(
                f"cannot resume '{resume_from}': its {label} differs from "
                f"the current evaluation ({before.get(key)!r} vs "
                f"{context.get(key)!r}) — those judgments answered a "
                "different question. Run a fresh evaluate")
    before_judges = {j["name"]: j for j in before.get("judges") or []}
    for judge in context["judges"]:
        prior = before_judges.get(judge["name"])
        if prior is None:
            continue  # a judge added since: it simply has nothing to reuse
        if prior != judge:
            raise RuntimeError(
                f"cannot resume '{resume_from}': judge '{judge['name']}' was "
                f"configured differently ({prior} vs {judge}) — its earlier "
                "verdicts came from a different judge. Run a fresh evaluate")

    rows: dict[tuple[str, tuple[str, str, str]], dict] = {}
    path = os.path.join(partial, "judgments.jsonl")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                judge = str(row.get("judge") or "")
                if not judge:
                    continue
                rows[(judge, judgment_key(row))] = row
    except FileNotFoundError:
        rows = {}
    except json.JSONDecodeError as exc:
        # A torn tail is normal for a killed writer; everything before it is
        # still good. Refusing the whole file would throw away exactly the
        # data this feature exists to save.
        log(f"WARNING: '{resume_from}' judgments.jsonl has an unreadable "
            f"tail ({exc}) — reusing the {len(rows)} complete row(s) before it")
    log(f"resuming from '{resume_from}': {len(rows)} judgment(s) available "
        "for reuse")
    return rows


def judgment_stamp_judge(judgment: dict, ref) -> None:
    """Stamp a judgment with the judge that produced it, in place.

    Split out of the evaluate loop when judgments began being written as
    they are produced (2026-07-24): the stamp has to happen BEFORE the row
    reaches disk, not in a post-pass over the returned list."""
    from . import paired_judge
    judgment["judge"] = ref.name
    if ref.kind == "openrouter":
        # The VERIFIED serving provider from the verdict (the client refused
        # an unattributed or off-pin response) — the same per-judgment stamp
        # the deferred completion writes, so inline and deferred reports
        # carry one provenance shape.
        judgment["judgeProvider"] = paired_judge.canonical_openrouter_provider(
            (judgment.get("judgment") or {}).get("provider") or ref.provider)
