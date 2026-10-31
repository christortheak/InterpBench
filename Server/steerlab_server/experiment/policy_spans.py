"""Span policies in a study run: resolution, preflight, and the run-level
evidence sidecar.

A span policy (``positions: "span"``, policy schema version 2) acts while the
model READS a declared part of the study prompt — its tokens, in prefill, and
nowhere else. The declaration is :mod:`prompt_span`'s grammar; this module
turns it into positions for each prompt the run renders, before anything is
generated.

ONE DECLARATION PER POLICY, FOR THE WHOLE STUDY. Every prompt of the study is
resolved against the same declaration. Per-item declarations are later work.

RESOLVED ONCE PER DISTINCT RENDERED PROMPT. :class:`SpanResolutions` caches by
the declaration, the SHA-256 of the rendered text, and the rendering inputs
(the user text and the :func:`render_inputs` keywords; two different user
texts can render to one string when a template folds the system prompt into
the user turn, and a ``userMessage`` span selects different characters of
each). A prompt sampled eight times, or steered by two arms that render it
alike, is resolved once.

PREFLIGHT, NEVER HALFWAY. :func:`declared_preflight` checks every task prompt
with :func:`prompt_span.declared_user_range` before the model loads (no
tokenizer needed), and :func:`run_preflight` resolves every (span policy,
prompt) pair with the loaded tokenizer before the first generation of the run.
Any failure refuses the whole run up front with the resolver's typed refusal,
its code and its repair, as a ``studyDeclaration`` refusal. A location
control (``policy_controls``) also resolves the span it is matched to, and
refuses (:data:`LENGTH_MISMATCH`) on any prompt where the two token counts
differ. Scripted
transcripts and panels are refused there too: phase 1 resolves single-prompt
renders only.

THE SIDECAR (:data:`SIDECAR_FILENAME`). The resolver's evidence is stamped once
per distinct resolution, not once per response, in ``span-resolutions.json``
beside ``intervention-scope.json``: ``resolutions`` maps each evidence object's
content address (:func:`evidence_id`, the SHA-256 of its canonical JSON) to
the object, and ``uses`` names which condition, policy, and item read it. Each
response's ``interventionDecisions.spans`` entry references its resolution by
that address. The file is written once at run start for the WHOLE design
(every span arm and every prompt), so it is byte-identical across shards — a
deterministic shared artifact the shard merge verifies and carries — and a
resumed run keeps its file, refusing if a fresh resolution would differ.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os

from . import lifecycle_gates, paths, policy_artifacts, prompt_render, prompt_span

SIDECAR_FILENAME = "span-resolutions.json"
SCHEMA_VERSION = 1

#: Said wherever a path that cannot resolve a span meets a span policy.
UNSUPPORTED_PATH = (
    "A span policy acts while the model reads a declared part of one study "
    "prompt, so it runs only in single-prompt sampled studies on the Python "
    "engine. Scripted transcripts, panels, agent chat, and the cost instrument "
    "do not resolve spans yet. Use a study arm without scripted transcripts, "
    "or attach a policy that does not act on a span.")


def evidence_id(evidence: dict) -> str:
    """The content address of one resolver evidence object."""
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def token_ids_sha256(ids) -> str:
    """``promptTokensSHA256`` of a token-ID list, the resolver's own encoding."""
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def render_inputs(eff) -> dict:
    """The :func:`prompt_render.render` keywords one condition's generations use,
    spelled exactly as ``execute_condition`` hands them to ``generate``: the
    effort is passed only when the arm declares a reasoning budget, and
    otherwise the thinking boolean decides."""
    return {"prompt_mode": eff.prompt_mode, "system_prompt": eff.system_prompt,
            "qwen_thinking_enabled": eff.qwen_thinking_enabled,
            "reasoning_effort": eff.reasoning_effort if eff.reasoning_max_tokens else None}


class SpanResolutions:
    """A run's span resolutions, cached per declaration and distinct rendered
    prompt. ``resolver_calls`` counts calls into :func:`prompt_span.resolve`."""

    def __init__(self, tokenizer, model_id: str):
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.resolver_calls = 0
        self._cache: dict = {}
        self._uses: set = set()

    def resolve(self, declaration: dict, *, prompt: str, render: dict, rendered=None):
        """``(ResolvedSpan, evidence id)`` for ``declaration`` over ``prompt``.

        ``rendered`` is the prompt as generation renders it; without one it is
        rendered here, with the same inputs. Raises :class:`prompt_span.SpanRefusal`.
        """
        if rendered is None:
            rendered = prompt_render.render(self.tokenizer, prompt, model_id=self.model_id, **render)
        key = (json.dumps(declaration, sort_keys=True),
               hashlib.sha256(rendered.text.encode("utf-8")).hexdigest(),
               prompt, json.dumps(render, sort_keys=True))
        if key not in self._cache:
            self.resolver_calls += 1
            resolved = prompt_span.resolve(self.tokenizer, declaration, prompt=prompt,
                                           model_id=self.model_id, rendered=rendered, **render)
            self._cache[key] = (resolved, evidence_id(resolved.evidence))
        resolved, identifier = self._cache[key]
        if resolved.evidence["promptTokensSHA256"] != token_ids_sha256(rendered.input_ids):
            raise policy_artifacts.PolicyError(
                "The prompt this response renders differs from the one its span was "
                "resolved on. Report this run; its rendering changed while it ran.")
        return resolved, identifier

    def record_use(self, *, condition: str, policy: str, prompt_id, identifier: str) -> None:
        self._uses.add((condition, policy, str(prompt_id), identifier))

    def document(self, experiment: str) -> dict:
        """The sidecar payload: every resolution any recorded use reads."""
        used = {identifier for *_, identifier in self._uses}
        resolutions = {identifier: resolved.evidence
                       for resolved, identifier in self._cache.values() if identifier in used}
        return {"schemaVersion": SCHEMA_VERSION, "experiment": experiment,
                "resolverVersion": prompt_span.RESOLVER_VERSION,
                "resolutions": dict(sorted(resolutions.items())),
                "uses": [{"condition": c, "policySHA256": p, "promptID": i, "spanEvidenceSHA256": e}
                         for c, p, i, e in sorted(self._uses)]}


@dataclass(frozen=True)
class ResponseSpans:
    """What one response's span policies resolve against: the run's cache, the
    user text, and the rendering inputs exactly as generation renders them."""
    resolutions: SpanResolutions
    prompt: str
    render: dict

    def resolve(self, declaration: dict, rendered):
        return self.resolutions.resolve(declaration, prompt=self.prompt,
                                        render=self.render, rendered=rendered)


class SpanPreflightRefusal(lifecycle_gates.LifecycleRefusalMixin, ValueError):
    """A span policy the study cannot run as declared. Its gate is
    ``studyDeclaration``; ``problems`` carries each resolver refusal (code,
    reason, and repair) with the condition, policy, and item it belongs to."""

    def __init__(self, message: str, *, repair: str, problems: list[dict]):
        super().__init__(message, gate=lifecycle_gates.STUDY_DECLARATION, repair=repair)
        self.problems = problems
        self.code = problems[0]["code"] if problems else lifecycle_gates.STUDY_DECLARATION


#: At most this many problems are spelled out; the count says how many more.
_SHOWN = 20


def refuse(problems: list[dict], *, when: str) -> None:
    """Raise one :class:`SpanPreflightRefusal` naming every problem (up to
    :data:`_SHOWN`), or return when there are none."""
    if not problems:
        return
    lines = []
    for problem in problems[:_SHOWN]:
        where = " ".join(part for part in (
            f"condition '{problem['condition']}'" if problem.get("condition") else "",
            f"policy '{problem['policy']}'" if problem.get("policy") else "",
            f"item '{problem['promptID']}'" if problem.get("promptID") is not None else "") if part)
        lines.append(f"{where}: {problem['code']}: {problem['reason']}")
    if len(problems) > _SHOWN:
        lines.append(f"… and {len(problems) - _SHOWN} more")
    repairs = list(dict.fromkeys(problem["repairAction"] for problem in problems))
    raise SpanPreflightRefusal(
        f"span policy preflight: {len(problems)} problem(s) {when} — refusing "
        "before anything is generated:\n  - " + "\n  - ".join(lines),
        repair="; ".join(repairs[:4]), problems=problems)


def _problem(condition, policy, prompt_id, exc, label=None) -> dict:
    return {"condition": condition, "policy": policy, "promptID": prompt_id, "code": exc.code,
            "reason": f"{label}: {exc.reason}" if label else exc.reason, "repairAction": exc.repair_action}


def _declarations(doc):
    """The spans one policy needs on every prompt: its own, and a location control's
    controlled span, whose token count its own must equal."""
    yield doc["span"], None
    control = doc.get("control") or {}
    if control.get("kind") == "location":
        yield control["sourceSpan"], "the span this location control is matched to"


#: A location control whose span has another token count than the span it controls.
LENGTH_MISMATCH = "spanLengthMismatch"


def _transcript_problems(condition, policy, prompts) -> list[dict]:
    with_transcripts = [p.get("id") for p in prompts if p.get("transcript")]
    if not with_transcripts:
        return []
    return [{"condition": condition, "policy": policy, "promptID": None,
             "code": "spanTranscriptUnsupported",
             "reason": (f"{len(with_transcripts)} item(s) are scripted transcripts "
                        f"(first: '{with_transcripts[0]}'); a span resolves on a single "
                        "user prompt, so phase 1 does not steer transcripts"),
             "repairAction": ("run this agent on single-prompt items, or attach a policy "
                              "that does not act on a span")}]


def declared_problems(condition: str, variant, prompts) -> list[dict]:
    """Tokenizer-free problems for one agent arm: scripted transcripts, and every
    prompt :func:`prompt_span.declared_user_range` refuses."""
    problems = []
    for digest, doc in policy_artifacts.span_policies(getattr(variant, "intervention_policies", None)):
        problems += _transcript_problems(condition, doc["name"], prompts)
        for prompt in prompts:
            if prompt.get("transcript"):
                continue
            for declaration, label in _declarations(doc):
                try:
                    prompt_span.declared_user_range(declaration, prompt["prompt"])
                except prompt_span.SpanRefusal as exc:
                    problems.append(_problem(condition, doc["name"], prompt.get("id"), exc, label))
    return problems


def declared_preflight(arms, prompts) -> None:
    """Refuse before the model loads. ``arms`` is ``(condition name, variant)``
    pairs; an arm whose agent cannot be read is left to the run loop."""
    problems = []
    for condition, variant in arms:
        problems += declared_problems(condition, variant, prompts)
    refuse(problems, when="in the declared spans")


def panel_problems(seats) -> list[dict]:
    """Panels are out of scope for phase 1: a seat whose agent acts on a span."""
    return [{"condition": None, "policy": doc["name"], "promptID": None,
             "code": "spanPanelUnsupported",
             "reason": f"panel seat '{seat}' uses a span policy, which panels do not run yet",
             "repairAction": ("seat an agent without span policies, or run the span "
                              "policy in a single-prompt study")}
            for seat, variant in seats
            for _, doc in policy_artifacts.span_policies(getattr(variant, "intervention_policies", None))]


def resolve_condition(resolutions: SpanResolutions, condition: str, variant, prompts,
                      render: dict) -> list[dict]:
    """Resolve every (span policy, prompt) pair of one arm; the problems found.
    Each success is recorded as a use, for the sidecar."""
    problems = []
    for digest, doc in policy_artifacts.span_policies(getattr(variant, "intervention_policies", None)):
        transcripts = _transcript_problems(condition, doc["name"], prompts)
        if transcripts:
            problems += transcripts
            continue
        control = doc.get("control") or {}
        for prompt in prompts:
            try:
                resolved, identifier = resolutions.resolve(doc["span"], prompt=prompt["prompt"], render=render)
            except prompt_span.SpanRefusal as exc:
                problems.append(_problem(condition, doc["name"], prompt.get("id"), exc))
                continue
            if control.get("kind") == "location":
                # Matched on length, or not matched: a span of another length gets another dose.
                try:
                    controlled, _ = resolutions.resolve(control["sourceSpan"], prompt=prompt["prompt"], render=render)
                except prompt_span.SpanRefusal as exc:
                    problems.append(_problem(condition, doc["name"], prompt.get("id"), exc,
                                             "the span this location control is matched to"))
                    continue
                if controlled.token_count != resolved.token_count:
                    problems.append({
                        "condition": condition, "policy": doc["name"], "promptID": prompt.get("id"), "code": LENGTH_MISMATCH,
                        "reason": (f"this location control's span is {resolved.token_count} token(s) here, but the span it "
                                   f"controls is {controlled.token_count} token(s), so the two would receive different doses"),
                        "repairAction": ("edit the location control's span until it has as many tokens as the controlled span "
                                         "on every item (the counts are named per item), then review and publish the control again")})
                    continue
            resolutions.record_use(condition=condition, policy=digest,
                                   prompt_id=prompt.get("id"), identifier=identifier)
    return problems


def require_condition(resolutions: SpanResolutions, eff, prompts) -> None:
    """The condition-start check: every prompt of this arm resolves, or nothing
    of it is generated. After :func:`run_preflight` every lookup is cached."""
    refuse(resolve_condition(resolutions, eff.name, eff.variant, prompts, render_inputs(eff)),
           when=f"in condition '{eff.name}'")


def _declares_span(vc, root) -> bool:
    """Whether a variant condition's agent carries a span policy, read from its
    artifact alone; an agent that cannot be read is left to the run loop."""
    from . import model_variant
    try:
        variant = (model_variant.ModelVariant.from_dict(vc.artifact) if vc.artifact
                   else model_variant.ModelVariant.from_file(paths.resolve_artifact(vc.artifact_path, root)))
        return bool(policy_artifacts.span_policies(variant.intervention_policies))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def run_preflight(model, manifest, prompts, root, *, wants_choice: bool, run_directory: str,
                  log=None) -> SpanResolutions | None:
    """Resolve every span arm's spans on every prompt with the loaded tokenizer,
    before the first generation; write (or, on resume, check) the sidecar.
    Returns the run's :class:`SpanResolutions`, or None when no arm acts on a span.
    An arm whose agent cannot be resolved is left to the run loop, which records it."""
    from . import condition_execution
    arms = [vc for vc in manifest.variant_conditions if _declares_span(vc, root)]
    if not arms:
        return None
    resolutions = SpanResolutions(model.tokenizer, manifest.model_id)
    problems = []
    for vc in arms:
        try:
            eff = condition_execution.effective_variant_condition(
                vc, manifest, model, root, wants_choice=wants_choice)
        except Exception:  # noqa: BLE001 - the run loop owns this failure and records it
            continue
        problems += resolve_condition(resolutions, eff.name, eff.variant, prompts, render_inputs(eff))
    refuse(problems, when="resolving the declared spans on this model's tokenizer")
    document = resolutions.document(manifest.name)
    path = os.path.join(run_directory, SIDECAR_FILENAME)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != document:
            raise lifecycle_gates.refusing(
                lifecycle_gates.STUDY_DECLARATION,
                f"{SIDECAR_FILENAME} in this run differs from the span resolutions this "
                "engine computes now, so new responses would reference evidence the "
                "run does not hold",
                repair=("start a new run of the study; keep this run as it is, since "
                        "its existing responses match its own span-resolutions.json"))
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    if log is not None:
        log(f"span policies: {len(document['resolutions'])} distinct span resolution(s) "
            f"for {len(document['uses'])} use(s) → {SIDECAR_FILENAME}")
    return resolutions
