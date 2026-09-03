"""Paired-judge evaluation (parallel to Swift ``ClaudePairedJudge`` +
``ExperimentTasks`` evaluate).

Pairs each condition's generation with its same-prompt/seed **baseline**, shows
the judge two responses **A/B blinded** (so position can't leak which is
steered), asks for a winner + confidence + per-dimension 1–7 scores, and
aggregates per condition into baseline/variant/tie tallies. Requires
``ANTHROPIC_API_KEY``; degrades gracefully without it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re

from . import judge_credentials

DEFAULT_JUDGE_MODEL = os.environ.get("STEERLAB_JUDGE_MODEL", "claude-opus-4-8")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


#: Fixture path relative to the repo root — the ONE provider-identity table
#: both engines read (Swift twin: ``OpenRouterProviderIdentity``).
PROVIDER_FIXTURE = os.path.join(
    "prompts", "fixtures", "openrouter", "providers.json")


def _provider_fixture_candidates() -> list[str]:
    """Where the shipped provider fixture can live, in resolution order.

    1. The code tree this module runs from (three levels up from
       ``Server/steerlab_server/experiment/``) — a source checkout, an
       editable install, and the cluster payload all put
       ``prompts/fixtures/`` beside ``Server/`` (the push ships it
       explicitly, see ``scripts/make-server-payload.sh``).
    2. The artifact root (``STEERLAB_ROOT``/cwd, the same authority every
       other data path resolves through) — covers a server whose code is
       installed elsewhere but launched from the repo or a tree that
       carries the fixture.
    """
    from . import paths
    here = os.path.dirname(os.path.abspath(__file__))
    code_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidates = [os.path.join(code_root, PROVIDER_FIXTURE)]
    workspace = os.path.join(paths.project_root(), PROVIDER_FIXTURE)
    if workspace not in candidates:
        candidates.append(workspace)
    return candidates


def _load_provider_aliases(candidates: list[str]) -> dict[str, str]:
    """``{lowercased display name or slug: slug}`` from the first readable
    fixture among ``candidates``.

    A genuinely absent/unreadable fixture degrades to an EMPTY table rather
    than raising — canonicalization then falls back to plain lowercasing, so
    pins spelled as slugs keep working and everything else still fails
    closed. But it degrades LOUDLY: silent degradation reinstated the exact
    bug the fixture fixed (OpenRouter reports Vertex as ``Google``, which no
    longer matched a pinned ``google-vertex``), and nothing on the console
    said why correct judgments were being refused.
    """
    document = None
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as h:
                document = json.load(h)
            break
        except (OSError, json.JSONDecodeError):
            continue
    if document is None:
        print("WARNING: OpenRouter provider fixture "
              f"'{PROVIDER_FIXTURE}' not found (looked in: "
              + ", ".join(candidates)
              + ") — provider identity verification will refuse unknown "
              "aliases: display-name spellings (e.g. 'Google') will no "
              "longer match their slug (google-vertex), so correctly-served "
              "pinned judgments can be refused as off-pin")
        return {}
    aliases: dict[str, str] = {}
    for entry in document.get("providers") or []:
        slug = str(entry.get("slug") or "").strip()
        name = str(entry.get("name") or "").strip()
        if not slug:
            continue
        aliases[slug.lower()] = slug
        if name:
            aliases.setdefault(name.lower(), slug)
    return aliases


@functools.lru_cache(maxsize=1)
def _provider_aliases() -> dict[str, str]:
    """The alias table, loaded once per process.

    Cached: this is read on every judgment verification, and the file is
    committed data that cannot change under a running process. The cache
    also means the missing-fixture warning prints once, not per judgment.
    """
    return _load_provider_aliases(_provider_fixture_candidates())


def canonical_openrouter_provider(value: str | None) -> str:
    """Canonical provider identity shared by routing and evidence checks.

    OpenRouter ROUTES by lowercase slug but REPORTS the serving provider by
    display name, so the two spellings of one endpoint have to resolve to a
    single identity or a correct judgment gets refused as off-pin. The
    mapping comes from the committed fixture
    (``prompts/fixtures/openrouter/providers.json``), fetched from
    OpenRouter's public provider list — not from a hand-written alias list,
    which was wrong about Vertex (display name ``Google``, slug
    ``google-vertex``) and missing 86 of ~96 providers.

    Unknown spellings lowercase and are otherwise left alone, so they remain
    distinct and continue to FAIL CLOSED against any pin they do not match.
    """
    normalized = str(value or "").strip().lower()
    if not normalized:
        return normalized
    return _provider_aliases().get(normalized, normalized)


#: The ONE generation cap for judge verdicts, on every judging path of this
#: engine (inline Claude, OpenRouter, and the local-model ``_gen`` closures
#: in ``tasks``).
#:
#: The cap exists to bound RUNAWAY generation — a judge stuck in a loop must
#: not bill or block forever. It must never RATION reasoning: a judge that
#: thinks harder than we expected is doing its job well, and a run must never
#: stop because a judge spent tokens thinking (researcher directive
#: 2026-08-06).
#:
#: 1024 was the 2026-07-22 value (a winner + confidence + two-sentence reason
#: needs no more, and 512 had truncated a legible verdict on the cluster).
#: Reasoning-first models broke that arithmetic twice, because hidden
#: reasoning is billed against the SAME cap as the visible answer:
#:   - 2026-08-05: ``deepseek/deepseek-v4-flash-0731`` returned HTTP 200 with
#:     EMPTY content ("carried no content") — the whole 1024 went to
#:     reasoning;
#:   - 2026-08-06: ``google/gemini-3.6-flash`` @ google-ai-studio returned
#:     ~170-200-character mid-sentence prose fragments with no JSON, and the
#:     run refused after its retry (a 2026-08-05 workspace replication-study
#:     evaluate run, raw fragments in judge-failures.jsonl).
#: Both judges were doing nothing wrong; our budget truncated good analysis.
#: 8192 accommodates them, and the OpenRouter transport escalates further
#: (``JUDGE_MAX_ESCALATIONS``) when a provider still reports a length cut.
#:
#: Swift twin: ``PairedJudgeBudget.maxTokens`` (Sources/ExperimentKit/
#: ClaudePairedJudge.swift) — keep the two values identical.
JUDGE_MAX_TOKENS = 8192

#: How many times ONE OpenRouter judge call may DOUBLE its generation cap
#: when the provider reports a length cut: 8192 → 16384 → 32768. Escalation
#: is per-call and stateless — no learned cap is persisted, because the next
#: judgment may be a short one and a remembered cap would be an unaudited
#: change to the measurement. Swift twin:
#: ``OpenRouterPairedJudge.escalationLimit``.
JUDGE_MAX_ESCALATIONS = 2


def available(judge_kind: str = "claude") -> bool:
    """Whether an inline external judge of this kind can be armed — the key
    file (``judge_credentials``) or the matching env var."""
    return judge_credentials.available(judge_kind)


def is_claude_model(model_id: str) -> bool:
    """Whether the judge model is a Claude/Anthropic model (parallel to Swift
    ``ClaudePairedJudge.isClaudeModel``). Otherwise it's a local-model judge."""
    lid = (model_id or "").lower()
    return "claude" in lid or lid.startswith("anthropic")


def make_local_judge(generate_fn):
    """Build a judge function backed by a LOCAL served model (parallel to Swift
    ``LocalPairedJudge``). ``generate_fn(prompt)`` returns the model's text; we
    wrap it with the same blind-A/B rubric and JSON parsing as the Claude path,
    so it slots into ``evaluate``'s injectable ``judge`` hook. A malformed
    (unparseable) response RAISES ``ValueError`` rather than inventing a tie
    (invalid-verdict closure 2026-07-20 — a tie is substantive data); the
    callers' ``valid_verdict`` wrapper retries once and then refuses the
    judging phase."""
    def judge(model, rubric, a, b, structured, task_prompt=None):
        text = generate_fn(build_prompt(rubric, a, b, structured,
                                        task_prompt=task_prompt))
        try:
            return parse_response(text)
        except ValueError as exc:
            raise ValueError(
                "local judge returned no parseable JSON verdict "
                f"(got: {text[:300]!r})") from exc
    return judge


def _baseline_first(prompt_id: str, condition: str) -> bool:
    """Deterministic, seed-free A/B assignment: returns True when the baseline
    response is shown as A. Stable across re-runs (no RNG) and balanced across
    prompts so neither slot is systematically the steered one."""
    digest = hashlib.sha256(f"{prompt_id}|{condition}".encode()).digest()
    return digest[0] % 2 == 0


def build_prompt(judge_rubric: str, response_a: str, response_b: str,
                 structured_prompt: str | None,
                 task_prompt: str | None = None) -> str:
    """The canonical judge-prompt contract (engineer review 2026-07-18;
    wrapper unification 2026-07-22): every judging path — inline Claude,
    OpenRouter, local model, and the Mac's deferred judging — must give the
    judge the SAME information set: the rubric, the structured comparison
    fields (when declared), the TASK PROMPT the responses answered (when the
    caller has one), and the blinded A/B pair. A judge with the task prompt
    and a judge without it can reasonably reach different verdicts, which
    would make deferred selections incomparable to inline ones.

    This wrapper is BYTE-IDENTICAL to Swift's ``PairedJudgePrompt.build``
    (Sources/ExperimentKit/ClaudePairedJudge.swift) — both are pinned by the
    committed goldens in ``prompts/fixtures/paired-judge/`` and each
    engine's golden test fails loudly on drift. It explicitly demands a
    BRIEF reason: the 2026-07-22 incident judge stated a valid winner and
    then wrote an essay-length reasoning that outran the generation cap, so
    the JSON never closed. Change the wrapper only deliberately, on both
    engines at once, regenerating the goldens."""
    rubric = judge_rubric.strip() or (
        "Judge which response is higher quality and more appropriate.")
    structured = (structured_prompt or "").strip()
    structured_block = (
        "Structured comparison fields:\n" + structured + "\n\n"
        'Fill "structured_fields" with the requested conclusions. Use stable '
        "machine-readable values: booleans for yes/no findings, numbers for "
        "scales or deltas, and short strings for categories. If a field "
        "cannot be determined, use null rather than guessing. For "
        "directional per-response scores, name fields with A/B prefixes, "
        'such as "a_severity" and "b_severity".\n\n') if structured else ""
    task = (task_prompt or "").strip()
    task_block = ("=== Task prompt (both responses answered this) ===\n"
                  + task + "\n\n") if task else ""
    return (
        "You are a blinded evaluator comparing two model responses, A and "
        "B, to the same task prompt. Do not infer which response is a "
        "baseline or which was manipulated.\n\n"
        "Evaluation rubric:\n" + rubric + "\n\n"
        + structured_block + task_block +
        "=== Response A ===\n" + response_a.strip() + "\n\n"
        "=== Response B ===\n" + response_b.strip() + "\n\n"
        "Ignore style, length, and eloquence unless the rubric explicitly "
        'makes them relevant. Prefer "tie" when the evidence is genuinely '
        "indistinguishable.\n\n"
        "Return JSON only, stating the verdict fields first: "
        '{"winner": "A"|"B"|"tie", "confidence": 0..1, "brief_reason": '
        '"at most two sentences", "a_scores": {dim: 1-7}, "b_scores": '
        '{dim: 1-7}}. Keep "brief_reason" to at most two sentences — never '
        "an essay. If the rubric names scalar dimensions, use them for the "
        "1-7 scores; use null if a dimension does not apply. If structured "
        'comparison fields were requested, also include "structured_fields".')


class JudgeResponseError(ValueError):
    """An unparseable judge response, carrying the RAW text that produced it.

    Retention (2026-07-24): the parser's refusal used to be the only record
    of a malformed response — the text itself was discarded, so a failed
    evaluation left the researcher with "winner None" and nothing to look
    at. Subclasses ``ValueError`` so every existing caller (and every
    ``pytest.raises(ValueError)``) is unaffected; ``raw`` is what the
    failure log persists."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw
        #: Transport provenance (``transport_provenance``) for the call that
        #: produced this response, when a transport had any. Ridden into the
        #: failure log: the 2026-08-06 evaluate diagnosis was
        #: guesswork precisely because judge-failures.jsonl recorded no
        #: finish reason and no token usage next to the raw text.
        self.transport = None


#: A LEADING ``<think>`` tag (allowing attributes), whitespace permitted
#: before it. Only a leading block is stripped: a ``<think>`` appearing after
#: the verdict is content, not a reasoning preamble.
_THINK_OPEN = re.compile(r"\A\s*<think(?:\s[^>]*)?>", re.IGNORECASE)

#: Either think tag, anywhere — used to find the FIRST tag of any kind when
#: the text does not open with one (the orphan-closing-tag case below).
_THINK_ANY = re.compile(r"<(/?)think(?:\s[^>]*)?>", re.IGNORECASE)


def strip_leading_think_block(text: str) -> str:
    """Drop a leading ``<think>…</think>`` reasoning preamble.

    Some providers INLINE the model's reasoning in the visible content
    instead of delivering it out of band (2026-08-06 round). That preamble
    can contain braces, so it has to come off before JSON extraction or the
    decoder walks a reasoning aside looking for a verdict.

    An UNCLOSED leading ``<think>`` (the response was cut before the closing
    tag, or the provider simply omits it) is treated the same way:
    everything after the opening tag is the candidate text.

    An ORPHAN leading ``</think>`` — a closing tag with no opening tag
    anywhere before it — is treated as the block terminator (2026-09-03).
    Qwen3's chat template with ``enable_thinking`` emits the OPENING
    ``<think>`` as the last token of the PROMPT, so a generation decoded with
    ``skip_prompt`` begins inside the reasoning block and the first
    ``</think>`` is the only boundary the text carries. Everything up to and
    including that first ``</think>`` is dropped. The rule is deliberately
    narrow: the closing tag must precede EVERY ``<think>`` in the text, so a
    ``<think>…</think>`` aside after the verdict is still content, not a
    preamble. Reasoning that was truncated before its ``</think>`` in this
    mode carries no tag at all and is indistinguishable from plain text; it
    is returned unchanged. Cross-engine twin: Swift
    ``PairedJudgeVerdictParser.strippingLeadingThinkBlock``.
    """
    match = _THINK_OPEN.match(text or "")
    if match is None:
        first = _THINK_ANY.search(text or "")
        if first is not None and first.group(1) == "/":
            return text[first.end():]
        return text
    rest = text[match.end():]
    close = rest.lower().find("</think>")
    if close < 0:
        return rest
    return rest[close + len("</think>"):]


def parse_response(text: str) -> dict:
    """Extract the judge's JSON object from a (possibly fenced) response.

    REAL DECODER FIRST (external review 2026-07-23, finding 5): the old
    brace counter counted braces inside quoted strings, so valid JSON whose
    reason mentioned a brace was misread as truncated (an opening brace in a
    string made the object "never balance" and sent a perfectly valid
    verdict into salvage; a closing one truncated the slice mid-string).
    Every ``{`` position is now offered to the real JSON decoder — which is
    string-aware by construction and tolerates trailing prose/fences — and
    the first complete JSON object wins.

    Truncation salvage (2026-07-22 incident, cross-engine with Swift's
    ``PairedJudgeVerdictParser.parse``) runs ONLY after real decoding
    failed, ONLY within the candidate object (first ``{`` onward), and ONLY
    when exactly one complete ``"winner"`` field exists — a truncated
    response carrying two winner fields (duplicates, conflicting or not)
    has no single legible verdict and refuses into the existing
    retry-then-refuse path instead of parsing whichever came first. A
    single complete A/B/tie winner is still a legible, explicitly-written
    verdict, accepted with ``reasoningTruncated: true`` (plus the complete
    ``confidence`` number when present and whatever reasoning prefix was
    written).

    A leading ``<think>`` block is stripped first (see
    ``strip_leading_think_block``); the RAW text carried on the error is
    always the untouched response, because the failure log's job is to show
    what the judge actually sent."""
    candidate = strip_leading_think_block(text)
    start = candidate.find("{")
    if start < 0:
        raise JudgeResponseError("judge response had no JSON object", text)
    decoder = json.JSONDecoder()
    index = start
    while index != -1:
        try:
            obj, _end = decoder.raw_decode(candidate, index)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):
                return obj
        index = candidate.find("{", index + 1)
    salvaged = _salvage_truncated_verdict(candidate[start:])
    if salvaged is not None:
        return salvaged
    raise JudgeResponseError("unbalanced JSON in judge response", text)


#: A COMPLETE winner field: closing quote present, value exactly in the
#: A/B/tie vocabulary. Anything else refuses (never salvaged).
_WINNER_FIELD = re.compile(r'"winner"\s*:\s*"(A|B|tie)"')
#: A COMPLETE confidence number: digits followed by a delimiter — a number
#: cut mid-digits (end of text) is NOT complete and is dropped.
_CONFIDENCE_FIELD = re.compile(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)(?=[\s,}])')
#: The reasoning prefix: everything legibly written before truncation (the
#: field the unified wrapper asks for, plus the pre-unification "reasoning"
#: spelling so old-format truncations salvage too).
_REASON_PREFIX = re.compile(
    r'"(?:brief_reason|reasoning)"\s*:\s*"((?:[^"\\]|\\.)*)')


def _unescape_string_prefix(raw: str) -> str:
    """Minimal JSON-string unescape for a possibly-truncated prefix.
    Identical logic in Swift's ``PairedJudgeVerdictParser``: known escapes
    map, unknown escapes keep the escaped character, a trailing lone
    backslash (truncated mid-escape) is dropped."""
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(
                raw[i + 1], raw[i + 1]))
            i += 2
        elif ch == "\\":
            break
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _salvage_truncated_verdict(text: str) -> dict | None:
    """The truncated-verdict salvage rule (identical on both engines): a
    complete A/B/tie winner is accepted; the verdict is stamped
    ``reasoningTruncated: true`` (cross-engine key); confidence rides along
    only when its number is complete; the reasoning text is whatever
    complete prefix was written (possibly empty). Returns None when there
    is nothing legible to salvage — including when MORE than one complete
    winner field exists (finding 5, 2026-07-23): conflicting or duplicated
    winners in a truncated response have no single legible verdict, and
    picking the first would parse arbitrary data."""
    winners = _WINNER_FIELD.findall(text)
    if len(winners) != 1:
        return None
    verdict: dict = {"winner": winners[0], "reasoningTruncated": True}
    confidence = _CONFIDENCE_FIELD.search(text)
    if confidence is not None:
        verdict["confidence"] = float(confidence.group(1))
    reason = _REASON_PREFIX.search(text)
    verdict["brief_reason"] = (
        _unescape_string_prefix(reason.group(1)) if reason else "")
    return verdict


#: The only winners a verdict may record — anything else is invented data.
VALID_WINNERS = ("A", "B", "tie")


class JudgeNoncompliant(RuntimeError):
    """The judge ANSWERED, twice, and neither answer parsed to a verdict.

    Distinct from transport failure on purpose (2026-08-09): a judge that
    answers garbage is a per-item, classifiable outcome the evaluate loops
    record as a row and continue past; a judge whose CALL fails (HTTP error,
    network, credential) still fails the session, because the targeted-retry
    machinery exists to resume exactly those without re-paying completed
    judgments — and because recording a rate-limited call as "noncompliance"
    would misclassify a healthy judge."""


#: Per-judge noncompliance a run tolerates before it is systemic failure
#: (Christian, 2026-08-09): a few invalid verdicts become recorded rows and
#: the run completes; a judge failing more than this fraction of its column
#: fails the evaluation, because a report built mostly on holes is not a
#: result. Shared by the paired judge and the per-response coding loop.
NONCOMPLIANCE_CAP = 0.25


def _record_invalid(on_invalid, attempt, detail, raw, verdict,
                    judge_label, item_label, transport=None) -> None:
    """Hand one invalid attempt to the caller's recorder, if any. Best
    effort BY DESIGN: losing the diagnostic record is bad, but letting the
    recorder's own failure replace the judge's real error would be worse —
    the researcher would debug the logger instead of the judge.

    ``transport`` (2026-08-06 evaluate post-mortem) is the
    ``transport_provenance`` of the failed call — finishReason, truncation,
    the cap actually in force, token usage. Merged into the record so
    judge-failures.jsonl can DIAGNOSE (was this a length cut or an
    incoherent judge?) instead of forcing guesswork from raw text alone."""
    if on_invalid is None:
        return
    record = {"attempt": attempt + 1, "error": detail,
              "rawResponse": raw, "verdict": verdict,
              "judge": judge_label, "item": item_label}
    if transport:
        record.update(transport)
    try:
        on_invalid(record)
    except Exception:  # noqa: BLE001 - a recorder must never mask the judge
        pass


def valid_verdict(judge_fn, model, rubric, a, b, structured, *,
                  task_prompt=None, judge_label, item_label, on_invalid=None):
    """Call ``judge_fn`` and require a verdict whose winner is A/B/tie.

    Invalid-verdict closure (2026-07-20, cross-engine with Swift's
    ``judgmentWithValidWinner`` / ``SweepJudgmentRunner.judge``): an
    out-of-vocabulary winner (or an unparseable response, surfaced as
    ``ValueError``) used to be silently recorded as a substantive tie — or,
    worse, counted as a win for whichever side "A" wasn't — corrupting the
    preference mean with invented data. Judges are LLMs, so ONE malformed
    response is common: retry once; a second invalid verdict REFUSES the
    whole judging phase, naming the judge, the item, and what came back.
    Transport/credential errors (RuntimeError etc.) propagate immediately —
    only malformed VERDICTS are retried.

    ``on_invalid(attempt, detail, raw, verdict)`` (retention 2026-07-24) is
    called for EVERY invalid attempt, including the ones a successful retry
    papers over: the raw text and the parser's complaint are the diagnostic
    record a failed evaluation must leave behind. It is a recorder, never a
    judge — it cannot make an invalid verdict valid, and a recorder that
    raises must not become the failure."""
    failures = []
    for _attempt in range(2):
        raw = ""
        verdict = None
        try:
            verdict = judge_fn(model, rubric, a, b, structured,
                               task_prompt=task_prompt)
        except ValueError as exc:
            detail = str(exc)
            raw = getattr(exc, "raw", "")
            failures.append(detail)
            _record_invalid(on_invalid, _attempt, detail, raw, None,
                            judge_label, item_label,
                            transport=getattr(exc, "transport", None))
            continue
        winner = (verdict or {}).get("winner")
        if winner in VALID_WINNERS:
            return verdict
        detail = f"winner {winner!r} (expected 'A', 'B', or 'tie')"
        failures.append(detail)
        _record_invalid(on_invalid, _attempt, detail, raw, verdict,
                        judge_label, item_label)
    raise JudgeNoncompliant(
        f"judge {judge_label} returned an invalid verdict twice for "
        f"{item_label}: " + "; then ".join(failures) + " — refusing to "
        "record invented data; nothing was recorded for this phase — fix "
        "the judge or rubric, then re-run it")


def claude_complete(model: str, prompt: str) -> str:
    """One Claude completion under the unified judge cap — the raw transport
    shared by the paired judge and the per-response coding instrument
    (2026-08-04): both send ONE user message and read back the text blocks;
    only the prompt contract differs, and that lives with each caller."""
    credential = judge_credentials.credential_for("claude")
    if credential is None:
        raise RuntimeError(
            "no Anthropic credential on this server — the DEFAULT custody "
            "posture (2026-07-18): run Claude judging from the Mac app "
            "against the downloaded artifacts (its Keychain holds the key), "
            "pin a LOCAL judge, or — for the seamless pipeline — push a "
            "capped judge key from the app (Compute → External judge key; "
            "it lands at ~/.steerlab/judge-key, mode 600, and enables "
            "inline judging).")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("paired judge needs the 'anthropic' package: pip install anthropic") from exc
    client = anthropic.Anthropic(api_key=credential.key)
    message = client.messages.create(
        model=model, max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in message.content
                   if getattr(b, "type", "") == "text")


def judge_pair(model: str, judge_rubric: str, response_a: str, response_b: str,
               structured_prompt: str | None = None,
               task_prompt: str | None = None) -> dict:
    text = claude_complete(model, build_prompt(judge_rubric, response_a,
                                               response_b, structured_prompt,
                                               task_prompt=task_prompt))
    return parse_response(text)


#: Public, UNAUTHENTICATED catalogue of the endpoints serving one model.
#: Keyless matters: it lets provider pins be discovered and preflighted on a
#: machine with no judge credential — including at freeze, where keyless is
#: the default custody posture.
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


class OpenRouterModelNotFound(RuntimeError):
    """The catalogue ANSWERED and does not list the model — positive
    evidence the judge cannot run, not an unreachable-network advisory.

    The distinction is what the preflight's asymmetric refusal rule turns
    on: flattened into a generic RuntimeError, a nonexistent judge model
    (`deepseek/deepseek-v4-flash-0731`, 2026-08-04) was warned about and
    waved through, and the pipeline burned its full run stage before the
    first judge call 404'd."""


def openrouter_model_endpoints(model: str, *, transport=None,
                               timeout: float = 20.0) -> list[dict]:
    """The endpoints currently serving ``model``, newest catalogue state.

    Returns one dict per endpoint: ``provider`` (canonical routing slug),
    ``providerName`` (OpenRouter's display name), ``quantization``,
    ``contextLength``, ``status``, ``tag``. Raises RuntimeError on transport
    or protocol failure — callers decide whether that is fatal (discovery:
    yes, the researcher asked) or advisory (preflight: no, a compute node
    may simply have no outbound network).

    ``model`` is an ``author/slug`` id. ``transport`` is an httpx-transport
    seam for tests, matching ``openrouter_judge_pair``.
    """
    slug = (model or "").strip().strip("/")
    if slug.count("/") != 1 or not all(slug.split("/")):
        raise RuntimeError(
            f"'{model}' is not an OpenRouter model id — expected "
            "'author/slug' (e.g. 'google/gemma-3-27b-it')")
    import httpx
    url = OPENROUTER_ENDPOINTS_URL.format(model=slug)
    try:
        with httpx.Client(transport=transport, timeout=timeout) as client:
            response = client.get(url)
    except Exception as exc:  # noqa: BLE001 - surfaced as one clear failure
        raise RuntimeError(
            f"could not reach OpenRouter's model catalogue: "
            f"{type(exc).__name__}: {exc}") from exc
    if response.status_code == 404:
        raise OpenRouterModelNotFound(
            f"OpenRouter does not list a model '{slug}' — check the model "
            "slug (it is OpenRouter's id, which is not always the same as "
            "the Hugging Face repo id)")
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter model catalogue failed: HTTP "
            f"{response.status_code}: {response.text[:200]}")
    data = (response.json() or {}).get("data") or {}
    out: list[dict] = []
    for endpoint in data.get("endpoints") or []:
        name = str(endpoint.get("provider_name") or "").strip()
        if not name:
            continue
        out.append({
            "provider": canonical_openrouter_provider(name),
            "providerName": name,
            "quantization": (str(endpoint.get("quantization") or "").strip()
                             or None),
            "contextLength": endpoint.get("context_length"),
            "status": endpoint.get("status"),
            "tag": endpoint.get("tag"),
        })
    return out


def declared_external_service_egress() -> str:
    """This site's declared egress to non-hub external services (WP5 audit
    c52): ``yes`` | ``no`` | ``unknown``.

    Rendered from ``policy.externalServiceEgress`` into the cluster env file;
    anything unset or unrecognised reads as ``unknown``, which is what every
    site said before the profile could say otherwise."""
    value = (os.environ.get("STEERLAB_EXTERNAL_SERVICE_EGRESS") or "").strip().lower()
    return value if value in {"yes", "no", "unknown"} else "unknown"


def _egress_clause() -> str:
    """The site-declared reason an external catalogue may be unreachable."""
    egress = declared_external_service_egress()
    if egress == "no":
        return (". This site declares NO external-service egress from compute "
                "nodes (policy.externalServiceEgress), so an unreachable "
                "catalogue is expected here — verify the pin from a host that "
                "has egress")
    if egress == "yes":
        return (". This site declares external-service egress "
                "(policy.externalServiceEgress), so an unreachable catalogue "
                "is a fault to chase, not this site's normal posture")
    return ""


def preflight_openrouter_provider(model: str, provider: str, *,
                                  transport=None) -> dict:
    """Check a provider pin against OpenRouter's live catalogue.

    Returns ``{"problem": str|None, "warnings": [str], "checked": bool}``.

    The refusal rule is deliberately asymmetric: a pin is refused ONLY on
    positive evidence that it is wrong (the catalogue answered, and the
    pinned provider is not among the endpoints serving this model). A
    catalogue we could not reach yields ``checked: False`` and a warning,
    never a refusal — compute nodes routinely have no outbound network, and
    a study must not become unrunnable because a metadata endpoint was
    unreachable.

    WHY it was unreachable is site data since WP5 step 10 (audit c52):
    ``STEERLAB_EXTERNAL_SERVICE_EGRESS``, rendered from the profile's
    ``policy.externalServiceEgress``, says whether this site's compute nodes
    may reach non-hub services at all. It changes the WARNING, not the rule:
    a site that declares no such egress expects this and should verify pins
    from a host that has it; a site that declares egress has a real fault to
    chase. Undeclared keeps the historical wording, which assumed the former.

    The multi-quantization warning is the scientifically interesting one:
    the pin exists because quantization changes verdicts, but a provider
    that serves one model at several quantizations (``deepinfra/fp8`` and
    ``deepinfra/bf16``) is not fully pinned by provider alone.
    """
    warnings: list[str] = []
    pinned = canonical_openrouter_provider(provider)
    if not pinned:
        return {"problem": f"openrouter judge for '{model}' has no pinned "
                           "provider — an unpinned provider is not a pinned "
                           "judge",
                "warnings": warnings, "checked": False}
    try:
        endpoints = openrouter_model_endpoints(model, transport=transport)
    except OpenRouterModelNotFound as exc:
        # The catalogue ANSWERED: this is positive evidence, exactly what
        # the asymmetric rule refuses on. Treating it as "unverifiable"
        # (2026-08-04) let a pipeline burn its full GPU run stage before
        # the first judge call hit the same 404.
        return {"problem": str(exc), "warnings": warnings, "checked": True}
    except RuntimeError as exc:
        warnings.append(
            f"could not verify the provider pin for '{model}' ({exc}) — "
            "the pin is UNVERIFIED, not wrong; judging will still refuse "
            "an off-pin response at call time" + _egress_clause())
        return {"problem": None, "warnings": warnings, "checked": False}
    if not endpoints:
        # Also an ANSWERED catalogue: the model exists but nothing serves
        # it right now, so the judge call would fail with "No endpoints
        # found". Refusing is the cheap version of that failure — before
        # the run stage, not after it.
        return {"problem": (
                    f"OpenRouter currently lists NO serving endpoints for "
                    f"'{model}' — a judge call would fail with 'No endpoints "
                    "found'. Pick a served judge model (the app's Discover "
                    "button lists them), or retry when the model is served "
                    "again"),
                "warnings": warnings, "checked": True}

    matching = [e for e in endpoints if e["provider"] == pinned]
    if not matching:
        available = sorted({e["provider"] for e in endpoints})
        return {
            "problem": (
                f"provider '{provider}' does not serve '{model}' on "
                f"OpenRouter. Available: {', '.join(available)}. "
                "Pin one of those (they are routing slugs — the app's "
                "Discover button fills this in from this same catalogue)."),
            "warnings": warnings, "checked": True}

    quantizations = sorted({e["quantization"] for e in matching
                            if e["quantization"]})
    if len(quantizations) > 1:
        warnings.append(
            f"provider '{pinned}' serves '{model}' at more than one "
            f"quantization ({', '.join(quantizations)}) — the provider pin "
            "alone does not fix which one judges, and quantization is the "
            "reason this pin exists. Verdicts may vary between runs")
    degraded = [e for e in matching if e.get("status") not in (0, None)]
    if degraded and len(degraded) == len(matching):
        warnings.append(
            f"every '{pinned}' endpoint for '{model}' is currently flagged "
            "degraded by OpenRouter — judging may be slow or fail")
    return {"problem": None, "warnings": warnings, "checked": True}


#: finish_reason spellings that mean "the generation cap cut this off".
#: Providers disagree (OpenAI-style ``length``, Google ``MAX_TOKENS``,
#: others ``max_tokens``/``model_length``), and OpenRouter surfaces its own
#: normalization in ``finish_reason`` and the provider's raw value in
#: ``native_finish_reason`` — both are checked, because either one alone has
#: been the only signal in practice.
_LENGTH_FINISH_REASONS = frozenset({
    "length", "max_tokens", "maxtokens", "max_output_tokens",
    "model_length", "token_limit", "max_completion_tokens",
})


def is_length_finish(reason) -> bool:
    """Whether a finish_reason says the cap truncated the answer."""
    normalized = str(reason or "").strip().lower().replace("-", "_")
    normalized = normalized.replace(" ", "_")
    return normalized in _LENGTH_FINISH_REASONS


def _choice_was_truncated(choice: dict) -> bool:
    return any(is_length_finish(choice.get(key))
               for key in ("finish_reason", "native_finish_reason"))


def read_openrouter_usage(payload: dict) -> dict:
    """``{"completionTokens": n, "reasoningTokens": n}`` from a response's
    ``usage`` block, defensively — every key is optional and providers put
    reasoning tokens in either ``completion_tokens_details.reasoning_tokens``
    or a top-level ``reasoning_tokens``.

    REPORTED, NEVER GATED (researcher directive 2026-08-06): nothing in
    either engine reads these numbers to refuse, throttle, or shrink a
    judgment. They exist so the researcher can SEE that a judge is spending
    more than expected on hidden reasoning. Cross-engine keys with Swift's
    ``PairedJudgeUsage``."""
    def _count(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    out: dict = {}
    completion = _count(usage.get("completion_tokens"))
    if completion is not None:
        out["completionTokens"] = completion
    details = usage.get("completion_tokens_details")
    reasoning = (_count(details.get("reasoning_tokens"))
                 if isinstance(details, dict) else None)
    if reasoning is None:
        reasoning = _count(usage.get("reasoning_tokens"))
    if reasoning is not None:
        out["reasoningTokens"] = reasoning
    return out


def openrouter_complete(model: str, prompt: str, *, provider: str,
                        transport=None, usage_out: dict | None = None
                        ) -> tuple[str, str]:
    """One OpenRouter completion with the provider PINNED — the raw
    transport shared by the paired judge and the per-response coding
    instrument. OpenRouter is a router: the same model slug can be served by
    different backends running different quantizations with different
    outputs, so an unpinned provider is not a pinned judge — the request
    forbids fallbacks and the response's served-by provider is verified
    against the pin. Returns ``(text, canonical serving provider)``;
    ``transport`` is an httpx-transport seam for tests.

    ``usage_out``, when given, is FILLED IN PLACE with what this call cost
    and how it ended: ``completionTokens``/``reasoningTokens`` (when the
    response reported them), ``finishReason``, ``truncated`` (the cap cut the
    visible answer), ``cap`` (the last cap actually sent), and
    ``responseFormat`` (whether the JSON constraint was accepted). Callers
    use it for provenance and for the truncation DIAGNOSIS — never to gate.

    Truncation escalation (2026-08-06): a response whose finish_reason says
    "length" is retried with the cap DOUBLED, up to
    ``JUDGE_MAX_ESCALATIONS`` times. A reasoning-first model spends the cap
    on hidden reasoning, so the visible answer arrives truncated or empty
    through no fault of the judge; the fix is to give it room, never to
    refuse it for its token appetite."""
    credential = judge_credentials.credential_for("openrouter")
    if credential is None:
        raise RuntimeError(
            "no OpenRouter credential on this server — push a capped judge "
            "key from the app (Compute → External judge key; it lands at "
            "~/.steerlab/judge-key, mode 600), or set OPENROUTER_API_KEY")
    requested_provider = (provider or "").strip()
    if not requested_provider:
        raise RuntimeError(
            f"openrouter judge for '{model}' has no pinned provider — "
            "refusing an unpinned judgment")
    pinned = canonical_openrouter_provider(requested_provider)
    import httpx
    cap = JUDGE_MAX_TOKENS
    escalations = 0
    # Ask the ENDPOINT to constrain output to a JSON object (2026-08-06,
    # evaluate post-mortem): every caller of this transport demands
    # JSON ("Return JSON only" is in both prompt contracts), and 22 of 36
    # judgments on that run arrived as partial JSON rescued by
    # regex salvage. `json_object` is the portable spelling; `json_schema`
    # is deliberately NOT sent — the verdict's shape is rubric-dependent
    # (structured_fields is an open object the wrapper leaves to the judge),
    # and pinning a schema here would be a second, drifting copy of the
    # prompt contract. Falls back unconstrained when the pinned endpoint
    # rejects the parameter — the parser and salvage path behind us are
    # unchanged, so an unconstrained judge is exactly as safe as before.
    constrained = True
    while True:
        body = {
            "model": model,
            "max_tokens": cap,
            # Exclude only skips DELIVERING the reasoning text; it does NOT
            # limit thinking. There are deliberately no `effort`/`max_tokens`
            # reasoning knobs here: we welcome smarter models and do not
            # ration how hard a judge thinks (researcher directive
            # 2026-08-06). Not delivering the reasoning keeps the visible
            # answer — the JSON verdict — from sharing the response body
            # with an essay we never read.
            "reasoning": {"exclude": True},
            "provider": {"order": [pinned], "allow_fallbacks": False},
            "messages": [{"role": "user", "content": prompt}],
        }
        if constrained:
            body["response_format"] = {"type": "json_object"}
        with httpx.Client(transport=transport, timeout=300.0) as client:
            response = client.post(
                OPENROUTER_ENDPOINT, json=body,
                headers={"Authorization": f"Bearer {credential.key}"})
        if response.status_code != 200:
            # Graceful fallback, ONE axis at a time: a client error against
            # the constrained request is how OpenRouter surfaces an endpoint
            # that does not support `response_format` under a no-fallback
            # provider pin (400 invalid parameter / 404 no matching
            # endpoint / 422 unprocessable). Auth, billing, rate and server
            # errors (401/402/403/429/5xx) are NOT about the constraint —
            # dropping it there would just re-send a doomed request.
            if constrained and response.status_code in (400, 404, 422):
                constrained = False
                print(f"judge '{model}' @ {pinned} rejected response_format "
                      f"json_object (HTTP {response.status_code}) — retrying "
                      "unconstrained; the JSON parser and truncation salvage "
                      "still apply")
                continue
            raise RuntimeError(
                f"OpenRouter judge call failed: HTTP {response.status_code}: "
                f"{response.text[:300]}")
        payload = response.json()
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        text = (choice.get("message") or {}).get("content") or ""
        usage = read_openrouter_usage(payload)
        if "reasoningTokens" in usage:
            # Transparency, not a gate: the researcher sees what the judge
            # spent. Providers differ on whether completion_tokens already
            # includes the reasoning tokens, so both numbers are printed as
            # reported rather than differenced.
            print(f"judge '{model}' used {usage['reasoningTokens']} reasoning "
                  f"+ {usage.get('completionTokens', 0)} answer tokens")
        truncated = _choice_was_truncated(choice)
        if truncated and escalations < JUDGE_MAX_ESCALATIONS:
            escalations += 1
            cap *= 2
            print(f"judge '{model}' hit its {cap // 2}-token cap "
                  f"(finish_reason "
                  f"{choice.get('finish_reason') or choice.get('native_finish_reason')!r})"
                  f" — retrying with {cap}; a judge is never refused for "
                  "spending tokens on reasoning")
            continue
        break
    if usage_out is not None:
        usage_out.update(usage)
        usage_out["finishReason"] = (choice.get("finish_reason")
                                     or choice.get("native_finish_reason"))
        usage_out["truncated"] = truncated
        usage_out["cap"] = cap
        # Whether the endpoint accepted the JSON constraint — a verdict that
        # still arrives as prose under "json_object" is a different diagnosis
        # from one the endpoint never promised to constrain.
        usage_out["responseFormat"] = ("json_object" if constrained
                                       else "unsupported")
    # Empty content WITH a length finish is the 2026-08-05 incident
    # (deepseek-v4-flash: HTTP 200, no content, the whole cap eaten by
    # hidden reasoning). Escalation has already been exhausted by the time
    # we get here, so the empty text goes on to the normal parse path, which
    # names the mechanism — refusing here as "carried no content" would hide
    # it. Empty content WITHOUT a length finish keeps the old refusal: that
    # is a genuinely empty answer, not a truncated one.
    if not text and not truncated:
        raise RuntimeError(
            f"OpenRouter judge response for '{model}' carried no content")
    served_by = str(payload.get("provider") or "").strip()
    # The response must NAME its serving provider (engineer review
    # 2026-07-18, provider-evidence pass): a missing field would otherwise
    # record the REQUESTED provider as if it were verified — provenance is
    # refused rather than believed, never assumed.
    if not served_by:
        raise RuntimeError(
            f"OpenRouter response for '{model}' named no serving provider — "
            "cannot verify the provider pin; refusing an unattributed "
            "judgment")
    # allow_fallbacks=False should make a mismatch impossible; verify anyway
    # and refuse — a judgment served off-pin is not the declared judge.
    if canonical_openrouter_provider(served_by) != pinned:
        raise RuntimeError(
            f"OpenRouter served the judgment via '{served_by}' but the "
            f"pinned provider is '{requested_provider}' — refusing an off-pin "
            "judgment")
    return text, canonical_openrouter_provider(served_by)


def transport_provenance(info: dict) -> dict:
    """The transport facts a judgment record and a failure record BOTH carry
    (2026-08-06): how the call finished (``finishReason``), whether the cap
    cut it (``truncated``) and at what ceiling (``cap``), what it cost
    (``usage``), and whether the endpoint accepted the JSON constraint
    (``responseFormat``). Recorded provenance on every row — the 2026-08-06
    evaluate run's judge-failures.jsonl carried none of it, and the
    diagnosis was guesswork. Nothing gates on any of these."""
    out: dict = {}
    if info.get("finishReason") is not None:
        out["finishReason"] = info["finishReason"]
    if "truncated" in info:
        out["truncated"] = bool(info["truncated"])
    if info.get("cap"):
        out["cap"] = info["cap"]
    if info.get("responseFormat"):
        out["responseFormat"] = info["responseFormat"]
    usage = {key: info[key] for key in ("completionTokens", "reasoningTokens")
             if key in info}
    if usage:
        out["usage"] = usage
    return out


def truncation_diagnosis(exc: JudgeResponseError, model: str,
                         info: dict) -> JudgeResponseError:
    """Re-word an unparseable-response refusal when the evidence says the
    judge was TRUNCATED rather than incoherent (2026-08-06).

    "judge response had no JSON object" sent the researcher looking for a
    broken rubric when the real mechanism was a reasoning-first model
    spending the whole cap on hidden reasoning. Still a
    ``JudgeResponseError`` (hence a ``ValueError``), so ``valid_verdict``'s
    retry-once-then-refuse semantics and the raw-response failure log are
    untouched — only the wording changes."""
    if not (info.get("truncated") or info.get("reasoningTokens")):
        return exc
    cap = info.get("cap") or JUDGE_MAX_TOKENS
    return JudgeResponseError(
        f"judge '{model}' returned no parseable verdict: the judge's visible "
        f"answer was truncated after {cap} tokens despite escalation "
        "(reasoning models spend the cap on hidden reasoning) — the raw text "
        "is retained in judge-failures.jsonl",
        getattr(exc, "raw", "") or "")


def openrouter_judge_pair(model: str, judge_rubric: str, response_a: str,
                          response_b: str, structured_prompt: str | None = None,
                          task_prompt: str | None = None, *, provider: str,
                          transport=None) -> dict:
    """One provider-pinned OpenRouter judgment (see ``openrouter_complete``
    for the pin semantics). The returned verdict is stamped with the
    serving provider (recorded provenance, not ambient fact), the call's
    ``finishReason`` and ``responseFormat``, and — when the response
    reported them — the judge's token ``usage``: provenance the report
    aggregates and NOTHING gates on. An unparseable response raises with
    the same transport facts attached, so the failure log carries them."""
    info: dict = {}
    text, served = openrouter_complete(
        model, build_prompt(judge_rubric, response_a, response_b,
                            structured_prompt, task_prompt=task_prompt),
        provider=provider, transport=transport, usage_out=info)
    try:
        verdict = parse_response(text)
    except JudgeResponseError as exc:
        diagnosed = truncation_diagnosis(exc, model, info)
        # The failure record must carry the transport facts next to the raw
        # text (2026-08-06): `valid_verdict` reads this off the error and
        # writes it into judge-failures.jsonl.
        diagnosed.transport = transport_provenance(info)
        raise diagnosed from exc
    verdict["provider"] = served
    # How the call FINISHED, on every judgment record — not only failures
    # (2026-08-06): a clean verdict with finishReason "length" is a salvage
    # candidate a report reader must be able to spot without re-running.
    if info.get("finishReason") is not None:
        verdict["finishReason"] = info["finishReason"]
    if info.get("responseFormat"):
        verdict["responseFormat"] = info["responseFormat"]
    usage = {key: info[key] for key in ("completionTokens", "reasoningTokens")
             if key in info}
    if usage:
        verdict["usage"] = usage
    return verdict


def make_openrouter_judge(model: str, provider: str, transport=None):
    """A panel-shaped judge callable for one PINNED (model, provider). The
    entry's pins are authoritative: the positional model the panel passes is
    ignored in favor of the closure's — the judging client never substitutes
    an ambient default."""
    def judge(_model, rubric, a, b, structured, task_prompt=None):
        return openrouter_judge_pair(model, rubric, a, b, structured,
                                     task_prompt=task_prompt,
                                     provider=provider, transport=transport)
    return judge


# Cross-engine refusal (external review 2026-07-22, P0): a judged evaluate
# that pairs nothing must REFUSE, never write a successful-looking report
# with "pairs: 0". Swift twin: `ExperimentTasks.noPairsMessage` — keep the
# strings identical.
NO_PAIRS_MESSAGE = (
    "paired judging produced zero pairs: records join on (promptID, "
    "sampleIndex) and every non-baseline record needs a baseline record in "
    "the same cell — this source run has none. Likely causes: the run has "
    "no baseline condition, or no sampled text records (instrument readouts "
    "and error records are never judged). Refusing to write an empty judged "
    "report.")


def _pair_generations(generations: list[dict]) -> list[dict]:
    """Group generations by (promptID, sampleIndex) and pair each
    non-baseline condition with the baseline for that cell.

    The join key is the SAMPLE CELL, never the seed: under seedPolicy
    ``derivedSHA256`` the seed derivation deliberately includes CONDITION
    identity (``tasks.derive_seed`` — paired at the prompt level, not a
    common-random-numbers design), so a baseline and a variant draw of the
    same (prompt, sampleIndex) NEVER share a seed. The old (promptID, seed)
    join therefore paired nothing on every positive-temperature study
    (external review 2026-07-22, P0). An absent ``sampleIndex`` normalizes
    to 0, so greedy/single-sample records (local Swift runs, legacy server
    runs, multi-agent turns, manifestSeeds single-sample records) pair
    exactly as before. Each pair records both sides' seeds as provenance
    under the cross-engine keys ``baselineSeed``/``variantSeed`` — there is
    deliberately no field named ``seed`` on a pair.

    Only sampled-text records can form pairs: instrument readouts
    (``"instrument"`` present) and error records carry no ``output``, and
    pairing them would burn a judge call on an empty-vs-empty tie."""
    cells: dict[tuple, dict[str, dict]] = {}
    for g in generations:
        if "instrument" in g or "error" in g or "output" not in g:
            continue
        key = (g.get("promptID"), g.get("sampleIndex") or 0)
        cells.setdefault(key, {})[g.get("condition")] = g
    pairs = []
    for (prompt_id, sample_index), by_cond in cells.items():
        baseline = by_cond.get("baseline")
        if baseline is None:
            continue
        for cond, g in by_cond.items():
            if cond == "baseline":
                continue
            pairs.append({"promptID": prompt_id, "sampleIndex": sample_index,
                          "condition": cond,
                          "baselineSeed": baseline.get("seed"),
                          "variantSeed": g.get("seed"),
                          # The task prompt rides along for the canonical
                          # judge contract (same information set on every
                          # judging path — inline AND deferred packets).
                          "prompt": baseline.get("prompt", ""),
                          "baseline": baseline.get("output", ""),
                          "variant": g.get("output", "")})
    return pairs


def sum_judgment_usage(judgments: list[dict]) -> dict:
    """Per-judge token totals over a column of judgment rows.

    REPORT, NEVER GATE (researcher directive 2026-08-06): these sums land in
    the judge report so a researcher can see that a judge is spending far
    more on hidden reasoning than the answer needs. No code reads them to
    refuse, cap, or select. Rows whose verdict carried no ``usage`` (local
    and Claude judges, older artifacts) simply contribute nothing."""
    totals: dict = {}
    for row in judgments:
        usage = ((row or {}).get("judgment") or {}).get("usage")
        if not isinstance(usage, dict):
            continue
        for key in ("completionTokens", "reasoningTokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0) + int(value)
    return totals


class ReusedJudgmentMismatch(RuntimeError):
    """A judgment offered for reuse does not describe the same pair cell.

    Raised rather than silently re-judging or silently accepting: if the
    blinding orientation recorded for a cell disagrees with what this run
    computes, the earlier verdict was about a different presentation of the
    responses, and reusing it would mix two experiments.
    """


def evaluate(generations: list[dict], *, judge_model: str, judge_rubric: str,
             structured_prompt: str | None = None,
             judge=judge_pair, on_judgment=None,
             on_invalid=None, existing=None) -> tuple[list[dict], dict]:
    """Judge every paired generation; return (judgments, report). ``judge`` is
    injectable so the pairing/aggregation is testable without the network.

    Refuses (RuntimeError, ``NO_PAIRS_MESSAGE``) when pairing survives
    nothing: a judge is always configured on this path, and a quiet
    ``pairs: 0`` report was exactly how the seed-join P0 hid itself.

    ``on_judgment(judgment)`` (retention 2026-07-24) fires as each judgment
    is produced, so the caller can persist it BEFORE the next judge call
    can fail — the returned list stays the aggregate for the report.
    ``on_invalid`` is handed to ``valid_verdict`` for the raw-response
    diagnostic log.

    ``existing`` (targeted retry, 2026-07-24) maps a pair cell —
    ``(promptID, sampleIndex, condition)`` as strings — to a judgment row
    this judge already produced in an earlier, incomplete session. Those
    cells are REUSED verbatim instead of re-judged: judging is the
    expensive, non-deterministic part, and re-running a verdict that was
    already valid both costs money and silently replaces recorded data. The
    reused row's blinding orientation is verified against what this run
    computes, because a verdict about the other presentation is a verdict
    about a different question."""
    pairs = _pair_generations(generations)
    if not pairs:
        raise RuntimeError(NO_PAIRS_MESSAGE)
    judgments: list[dict] = []
    tally: dict[str, dict] = {}
    reused_count = 0
    noncompliant_count = 0
    for pair in pairs:
        baseline_is_a = _baseline_first(str(pair["promptID"]), pair["condition"])
        cell = (str(pair["promptID"]), str(pair.get("sampleIndex") or 0),
                str(pair["condition"]))
        reused = (existing or {}).get(cell)
        if reused is not None and reused.get("noncompliant"):
            # A noncompliant row from an earlier session is a RECORDED
            # failure, not a verdict — re-judge the cell rather than reuse
            # (or refuse over) it. The old row survives in the earlier
            # session's file; this session gets a fresh chance at a verdict.
            reused = None
        if reused is not None:
            recorded = str(reused.get("baselineWas") or "")
            expected = "A" if baseline_is_a else "B"
            if recorded and recorded != expected:
                raise ReusedJudgmentMismatch(
                    f"cell {cell} was judged with the baseline as "
                    f"'{recorded}' but this run presents it as "
                    f"'{expected}' — the earlier verdict answered a "
                    "differently-blinded question and cannot be reused")
            outcome = reused.get("outcome")
            if outcome not in ("baseline", "variant", "tie"):
                raise ReusedJudgmentMismatch(
                    f"cell {cell} carries outcome {outcome!r}, which is not "
                    "a recorded verdict — refusing to reuse it")
            judgments.append(reused)
            reused_count += 1
            agg = tally.setdefault(
                pair["condition"],
                {"baselineWins": 0, "variantWins": 0, "ties": 0, "n": 0})
            agg["n"] += 1
            agg[{"baseline": "baselineWins", "variant": "variantWins",
                 "tie": "ties"}[outcome]] += 1
            if on_judgment is not None:
                on_judgment(reused)
            continue
        a, b = ((pair["baseline"], pair["variant"]) if baseline_is_a
                else (pair["variant"], pair["baseline"]))
        # Canonical judge contract (2026-07-19): the judge sees the TASK
        # PROMPT the responses answered — evaluate now matches the sweep
        # paths (a judge with the prompt and one without can reasonably
        # reach different verdicts). The verdict's winner is VALIDATED
        # (retry once, then refuse — `valid_verdict`): a missing winner was
        # silently recorded as a tie, and an out-of-vocabulary one was
        # counted as a WIN for one side — both invented data.
        try:
            verdict = valid_verdict(
                judge, judge_model, judge_rubric, a, b, structured_prompt,
                task_prompt=pair.get("prompt") or None,
                judge_label=f"'{judge_model}'",
                item_label=f"pair {pair['condition']}/{pair['promptID']}",
                on_invalid=on_invalid)
        except JudgeNoncompliant as exc:
            # A judge that ANSWERS but will not produce a verdict for THIS
            # pair no longer kills the whole evaluation (Christian,
            # 2026-08-09): after hours of generation and hundreds of good
            # judgments, one flaky refusal was aborting entire runs. The
            # refusal-to-invent stands — no winner is recorded — but the
            # failure becomes a per-pair ROW, kept for later examination
            # and classification, excluded from every tally and agreement
            # statistic. Systemic failure is different (the cap check after
            # the loop), and TRANSPORT failure is different too — a plain
            # RuntimeError still fails the session, which targeted retry
            # resumes without re-paying completed judgments.
            judgment = {**{k: pair[k] for k in ("promptID", "sampleIndex",
                                                "condition", "baselineSeed",
                                                "variantSeed")},
                        "baselineWas": "A" if baseline_is_a else "B",
                        "outcome": None,
                        "noncompliant": True,
                        "noncomplianceReason": str(exc)[:2000],
                        "judgment": None}
            judgments.append(judgment)
            noncompliant_count += 1
            if on_judgment is not None:
                on_judgment(judgment)
            continue
        winner = verdict.get("winner")
        if winner == "tie":
            outcome = "tie"
        else:
            baseline_won = (winner == "A") == baseline_is_a
            outcome = "baseline" if baseline_won else "variant"
        judgment = {**{k: pair[k] for k in ("promptID", "sampleIndex",
                                            "condition", "baselineSeed",
                                            "variantSeed")},
                    "baselineWas": "A" if baseline_is_a else "B",
                    "outcome": outcome, "confidence": (verdict or {}).get("confidence"),
                    "judgment": verdict}
        if verdict.get("reasoningTruncated"):
            # Salvage visibility (2026-08-06): a verdict rescued by the
            # truncation-salvage regex is a legible winner but NOT a clean
            # verdict — a retry that "succeeds" via salvage is worse than a
            # visible failure if nothing marks it. Stamped on the row (not
            # only inside the nested verdict) so report readers separate
            # clean columns from salvaged ones without re-parsing.
            judgment["verdictSalvaged"] = True
        judgments.append(judgment)
        if on_judgment is not None:
            # Persist-then-continue: the caller writes this row before the
            # NEXT judge call can fail, which is the whole point.
            on_judgment(judgment)
        agg = tally.setdefault(pair["condition"],
                               {"baselineWins": 0, "variantWins": 0, "ties": 0, "n": 0})
        agg["n"] += 1
        agg[{"baseline": "baselineWins", "variant": "variantWins", "tie": "ties"}[outcome]] += 1

    if noncompliant_count and noncompliant_count / len(pairs) > NONCOMPLIANCE_CAP:
        # A few flaky refusals are survivable; a judge that fails a quarter
        # of its column is broken, and a "completed" evaluation built on it
        # would be worse than a failed one. Every row — compliant and not —
        # was already persisted through on_judgment, so nothing is lost.
        raise RuntimeError(
            f"judge {judge_model!r} was noncompliant on {noncompliant_count} "
            f"of {len(pairs)} pairs (> {NONCOMPLIANCE_CAP:.0%} cap) — this "
            "is systemic judge failure, not flakiness. All rows (including "
            "the noncompliant ones, with raw reasons) were persisted; fix "
            "or swap the judge, then re-run evaluate")
    report = {"conditions": tally, "pairs": len(pairs), "judgeModel": judge_model}
    if noncompliant_count:
        # Loud, nonzero-only: these pairs have NO verdict from this judge —
        # excluded from tallies and agreement, kept as rows for later
        # classification. A reader must see the column is incomplete.
        report["noncompliantJudgments"] = noncompliant_count
    # Token provenance for the whole column (cross-engine keys with Swift's
    # per-judge `judgeUsage` map). Present only when some judgment reported
    # usage; nothing reads it to gate anything.
    report.update(sum_judgment_usage(judgments))
    salvaged = sum(1 for j in judgments
                   if j.get("verdictSalvaged")
                   or (j.get("judgment") or {}).get("reasoningTruncated"))
    if salvaged:
        # How many of this column's verdicts were truncation-salvaged
        # rather than cleanly parsed (2026-08-06) — including rows reused
        # from an earlier session, which carry their own stamps. Present
        # only when nonzero, like reusedJudgments; nothing gates on it,
        # but a column that is mostly salvage is not the same evidence as
        # a clean one and the report must say so.
        report["salvagedVerdicts"] = salvaged
    if reused_count:
        # How much of this judge's column came from an earlier session is
        # PROVENANCE, not a footnote: an external judge's model can change
        # between sessions, so a reader has to be able to see that a column
        # was not produced in one sitting.
        report["reusedJudgments"] = reused_count
        report["freshJudgments"] = len(pairs) - reused_count
    return judgments, report
