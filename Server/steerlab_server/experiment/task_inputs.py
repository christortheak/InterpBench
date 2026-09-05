"""Study prompt loading and admission; independent of model execution."""
from __future__ import annotations
import hashlib
import json
import os
from . import lifecycle_gates, paths, prompt_render, response_format
from .manifest import Manifest


def missing_task_prompts_refusal(path: str) -> str:
    """Twin of Swift's ``ExperimentTasks.loadTaskPrompts`` /
    ``ExperimentStore.pinTaskPrompts`` sentence for a prompt set that is not on
    disk. Swift has typed this since step 7; this engine raised a bare
    ``FileNotFoundError``."""
    return f"task prompt file not found: {path}"



def missing_task_prompts_repair(name: str, relative: str) -> str:
    return (f"author {relative} as {{\"id\": …, \"prompt\": …}} JSONL rows, "
            f"then steerlab-cli experiment pin-prompts {name} {relative}  "
            f"(authoring is Mac-authority) ; then steerlab-server experiment "
            f"run {name}")



def load_prompts(manifest: Manifest, prompts_file: str | None, root: str | None) -> list[dict]:
    """Load task prompts with the drift checks the firewall promises.

    The pinned file must match its pinned hash at RUN time, not only at
    freeze; an override (--prompts) on a frozen study must be byte-identical
    to the pin (frozen means frozen — dev iteration belongs on a duplicate);
    and a frozen study cannot run on unpinned prompts at all.
    """
    path = prompts_file or manifest.task_prompts_file
    if not path:
        raise RuntimeError("no task prompts file specified")
    if not os.path.isabs(path):
        path = os.path.join(paths.project_root() if root is None else root, path)
    try:
        with open(path, "rb") as handle:
            live_hash = hashlib.sha256(handle.read()).hexdigest()
    except FileNotFoundError:
        # Swift's twin (`ExperimentTasks.loadTaskPrompts`) has refused this as
        # a typed `missingPrerequisite` since step 7; this engine raised a bare
        # FileNotFoundError, so the same input answered `refused`/65 with a
        # runnable repair on the Mac and a traceback + 1 here.
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            missing_task_prompts_refusal(str(path)),
            repair=missing_task_prompts_repair(
                manifest.name,
                prompts_file or manifest.task_prompts_file or "")) from None
    frozen = manifest.status == "frozen"
    if prompts_file is None:
        if manifest.task_prompts_hash and live_hash != manifest.task_prompts_hash:
            raise lifecycle_gates.refusing(
                lifecycle_gates.PIN_DRIFT,
                f"task prompts '{manifest.task_prompts_file}' drifted from the "
                f"pinned hash (have {live_hash[:12]}…, pinned "
                f"{manifest.task_prompts_hash[:12]}…)",
                repair=(f"restore {manifest.task_prompts_file} to its pinned "
                        "bytes ; then re-run this verb (a frozen pin is never "
                        "re-pinned: duplicate the study on the Mac to change "
                        "it)"))
        if frozen and not manifest.task_prompts_hash:
            raise lifecycle_gates.refusing(
                lifecycle_gates.MISSING_PREREQUISITE,
                "frozen study has no pinned task prompts — duplicate, pin a "
                "prompt set, and re-freeze",
                repair=(f"steerlab-cli experiment duplicate {manifest.name} "
                        f"{manifest.name}-v2 && steerlab-cli experiment "
                        f"pin-prompts {manifest.name}-v2 prompts/…/file.jsonl "
                        f"&& steerlab-cli experiment freeze {manifest.name}-v2 "
                        "(authoring is Mac-authority)"))
    elif frozen and live_hash != manifest.task_prompts_hash:
        raise lifecycle_gates.refusing(
            lifecycle_gates.PIN_DRIFT,
            "prompt override on a FROZEN study must match the pinned prompt "
            "set byte-for-byte — duplicate the experiment to iterate",
            repair=(f"steerlab-cli experiment duplicate {manifest.name} "
                    f"{manifest.name}-v2 && steerlab-cli experiment "
                    f"pin-prompts {manifest.name}-v2 <the override file>"))
    prompts = []
    seen_ids: dict[str, int] = {}  # id → 1-based item ordinal
    with open(path, encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Auto-id parity with Swift `parseTaskPrompts` (2026-07-26).
            # This used to be the 0-based FILE LINE index, blank lines
            # included, while Swift used the 1-based ordinal of PARSED
            # prompts. For a file whose rows carry no explicit `id`, the two
            # engines therefore produced `prompt-0…N-1` and `prompt-1…N`.
            # Paired statistics key on promptID, so the intersection mapped
            # one engine's item k+1 onto the other's item k: it did not fail
            # to join, it joined the WRONG items and dropped one at each end.
            # A blank line anywhere shifted it further.
            # Explicit ids must be non-empty STRINGS; null and absent both
            # take the shared prompt-<ordinal> fallback (review 2026-08-03,
            # P2: `id: null` used to survive as None here while Swift fell
            # back — a divergence in the exact vocabulary the scope pin and
            # paired statistics key on). Message string is the cross-engine
            # contract (Swift twin: parseTaskPrompts).
            raw_id = obj.get("id")
            if raw_id is None:
                raw_id = f"prompt-{len(prompts) + 1}"
            elif not isinstance(raw_id, str) or not raw_id.strip():
                raise RuntimeError(
                    f"task prompts: item {len(prompts) + 1} declares an "
                    "empty or non-string 'id' — declare a non-empty string, "
                    "or omit the key for the prompt-<ordinal> fallback")
            entry = {"id": raw_id,
                     "prompt": obj.get("prompt") or obj.get("text", "")}
            # Duplicate item ids silently corrupt pairing on BOTH engines
            # (choice readouts and paired statistics key on promptID), so the
            # file refuses at LOAD — run/validate/sweep/logprob all inherit
            # the gate. Checked BEFORE the per-item transcript validation
            # (cross-engine ordering contract); the message string is the
            # cross-engine contract (Swift twin:
            # ExperimentTasks.parseTaskPrompts; fixture:
            # prompts/fixtures/task-prompts-validation/cases.json).
            item = len(prompts) + 1
            first = seen_ids.get(entry["id"])
            if first is not None:
                raise RuntimeError(
                    f"task prompts: duplicate item id '{entry['id']}' "
                    f"(items {first} and {item}) — ids must be unique for "
                    "pairing and reporting")
            seen_ids[entry["id"]] = item
            # Scripted transcript (the metacognition-study instrument): a
            # pinned multi-turn conversation — researcher-authored assistant
            # turns included — whose final user turn the model answers.
            # Schema-validated at LOAD on both engines (identical messages);
            # `text`/`prompt` becomes optional (display text derives from the
            # final user turn). Normalized to {role, content} so records are
            # cross-engine identical.
            if "transcript" in obj:
                violation = prompt_render.transcript_schema_violation(
                    obj["transcript"], entry["id"])
                if violation:
                    raise RuntimeError(violation)
                entry["transcript"] = prompt_render.normalize_transcript(
                    obj["transcript"])
                if not entry["prompt"]:
                    entry["prompt"] = prompt_render.transcript_display_text(
                        entry["transcript"])
            # Per-item attention check (the exclusion instrument's first
            # user): {"expected": …, "grading": <battery grading mode>?},
            # graded at ANALYSIS time against the record's output with the
            # capability battery's grading vocabulary. Validated at LOAD with
            # plain-language, cross-engine-identical messages; items without
            # a check are untouched (legacy files load unchanged).
            if "attentionCheck" in obj:
                from . import exclusions
                check_violation = exclusions.attention_check_violation(
                    obj["attentionCheck"], entry["id"])
                if check_violation:
                    raise RuntimeError(check_violation)
                entry["attentionCheck"] = exclusions.normalized_check(
                    obj["attentionCheck"])
            # Science-layer item metadata (all optional, carried into records):
            # answer options + which one the endpoint tracks, the presented
            # anchor and offense severity (Case 3), and the doctrine arm (Case 1).
            for key in ("options", "target", "anchorMonths", "severity", "arm", "caseID"):
                if key in obj:
                    entry[key] = obj[key]
            # What the prompt asks the model to EMIT — decides whether the
            # answer-token instruments can read this item at all. Closed
            # vocabulary, validated at LOAD: an unrecognised value refuses
            # rather than degrading to "unspecified", which would re-open the
            # hole `response_format` closes. Twin of Swift parseTaskPrompts.
            if obj.get("responseFormat") is not None:
                try:
                    entry["responseFormat"] = response_format.parse(
                        obj["responseFormat"])
                except ValueError as exc:
                    raise RuntimeError(
                        f"task prompt '{entry['id']}': {exc}") from exc
            # Factorial-design cell metadata (factor name → level name, the
            # generator's `factors` object): validated as a flat
            # string-to-string map at LOAD (identical message on both
            # engines) and carried into every record the item produces so
            # analysis can stratify by declared factors without rejoining
            # the input file. An empty object is treated as absent.
            if "factors" in obj:
                factors = obj["factors"]
                if (not isinstance(factors, dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in factors.items())):
                    raise RuntimeError(
                        f"task prompts: item '{entry['id']}' has a "
                        "'factors' value that is not a flat "
                        "string-to-string object — factor names and level "
                        "names must both be strings")
                if factors:
                    entry["factors"] = dict(factors)
            prompts.append(entry)
    return prompts



def check_response_formats(manifest, prompts: list[dict]) -> None:
    """Run-start gate: the declared option-consuming instruments must be able
    to read the items they will be pointed at, and any declared applicability
    scope must still select the items it was pinned to. Swift twin:
    ``ExperimentTasks.checkResponseFormats``.

    The vocabulary gate comes FIRST and is item-independent: a declaration
    this engine cannot dispatch is a worse failure than one it dispatches at
    the wrong items, because it produces no error at all — the study runs and
    measures nothing. Same gate as the rest of this function:
    ``responseFormat`` is the one gate whose subject is ``outcomeInstruments``
    and whose repair is ``set-instruments``, and its existing family is
    exactly "a declared instrument that would silently produce zero records"
    (the zero-item rules, 2026-08-06)."""
    from . import experiment_store  # local: admission needs the declared instrument scope
    unknown = experiment_store.unknown_outcome_instrument_problem(manifest.raw)
    if unknown:
        raise lifecycle_gates.refusing(
            lifecycle_gates.RESPONSE_FORMAT, unknown,
            repair=experiment_store.unknown_outcome_instrument_repair(
                manifest.name))
    items = response_format.items_of(prompts)
    scope = manifest.raw.get("outcomeInstrumentScope")
    drift = response_format.scope_drift_refusal(scope, items)
    if drift:
        # WP0 step 8: typed `responseFormat` (same prose, same exit code).
        raise lifecycle_gates.refusing(
            lifecycle_gates.RESPONSE_FORMAT, drift,
            repair=("steerlab-cli experiment set-instruments <name> "
                    "sampledText, or re-author the items so the pinned scope "
                    "selects them again and re-pin with steerlab-cli "
                    "experiment pin-prompts <name> <file>"))
    refusal = response_format.refusal(
        items, manifest.raw.get("outcomeInstruments"), scope)
    if refusal:
        raise lifecycle_gates.refusing(
            lifecycle_gates.RESPONSE_FORMAT, refusal,
            repair=("steerlab-cli experiment set-instruments <name> "
                    "sampledText, or re-author the items with "
                    '"responseFormat": "label" and re-pin with steerlab-cli '
                    "experiment pin-prompts <name> <file>"))



def check_transcript_prompts(manifest: Manifest, prompts: list[dict]) -> None:
    """Run-START refusal for scripted-transcript items (never a mid-run
    template error): rawCompletion cannot render a transcript, and every
    transcript must satisfy the study model family's chat-template
    constraints (Gemma's user-first strict alternation). Message strings are
    the cross-engine contract (Swift twin:
    ``ExperimentTasks.checkTranscriptPrompts``)."""
    transcripted = [p for p in prompts if p.get("transcript")]
    if not transcripted:
        return
    if manifest.prompt_mode == prompt_render.RAW_COMPLETION:
        raise RuntimeError(prompt_render.TRANSCRIPT_RAW_COMPLETION_MESSAGE)
    violations = []
    for prompt in transcripted:
        violation = prompt_render.transcript_family_violation(
            prompt["transcript"], prompt["id"], manifest.model_id)
        if violation:
            violations.append(violation)
    if violations:
        raise RuntimeError(
            f"scripted transcripts are incompatible with {manifest.model_id}'s "
            "chat template: " + "; ".join(violations))



def resolve_ordinal_aggregation(manifest: Manifest) -> str | None:
    """The manifest's declared ``ordinalAggregation`` when the ordinalScale
    instrument is declared, else None. Refuses (RuntimeError) a declared
    ordinalScale with a missing or unknown aggregation — the
    instrument-design choice is declared, never silently defaulted."""
    if "ordinalScale" not in manifest.outcome_instruments:
        return None
    from .logprob import ORDINAL_AGGREGATIONS
    aggregation = manifest.raw.get("ordinalAggregation")
    if aggregation not in ORDINAL_AGGREGATIONS:
        raise RuntimeError(
            "outcomeInstruments includes ordinalScale but ordinalAggregation "
            f"is {aggregation!r} — declare one of "
            + ", ".join(ORDINAL_AGGREGATIONS))
    return aggregation
