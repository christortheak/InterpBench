"""Chat-template capabilities DERIVED from the pinned template, recorded per
model (2026-09-05). Swift twin: ``SteeringKit.ModelCapabilities``.

Until this module, every family rule the renderers branched on was a
substring test on the model id: ``"qwen" in id`` meant "has a thinking mode
and reads ``reasoning_effort``", ``"gemma" in id`` meant "no system role". Two
of those were false in ways a frozen manifest could not see. Qwen/Qwen3-14B's
template reads ``enable_thinking`` but IGNORES ``reasoning_effort`` — a study
declaring ``medium`` ran at the template's default while its manifest asserted
medium — and a template can be re-vendored under the same id with different
variables. So the capabilities are now PROBED: the engine's own renderer is
asked, once per (model id, revision), what the template actually does, and the
answer is a hashed JSON record in the workspace that every gate, renderer,
run stamp and preregistration reads.

THE PROBES (``probe``). Each is a render comparison, never a regex over the
template source:

- **systemRole** — render ``[user]`` and ``[system, user]``. ``systemTurn``
  when the system text lands outside the user turn; ``foldedIntoUser`` when
  the second render is the first with the system text prepended inside the
  user turn (Gemma 3: ``system + "\\n\\n" + user``, and the separator is
  recorded); ``unsupported`` when the template raises on, or silently drops,
  the system turn.
- **thinkingSwitch** — render with ``enable_thinking=True`` and ``False``;
  ``supported`` iff the renders differ. ``thinkOpenInPrompt`` records whether
  the thinking-on generation prompt ENDS with an opening ``<think>`` tag
  (Qwen3.8 does; Qwen3 leaves the model to write it), which decides whether
  decoded output begins with the tag.
- **effortLevels** — for each candidate in :data:`EFFORT_CANDIDATES`, render
  thinking-on with ``reasoning_effort=<candidate>``. Whether the template
  READS the variable at all is settled first, with a value no template could
  accept: a raise or a changed render on the bogus value proves it is read.
  Then a candidate is ``accepted`` when the template reads the variable and
  renders it, ``rejected`` when the template raises on it, and ``ignored``
  when the variable is never read — a template whose default happens to equal
  a candidate is therefore still ``accepted``, which a plain "did the render
  change" comparison would have misfiled as ignored.
- **thinkTokens** — the single-token ids of ``</think>`` and ``<think>`` in
  the tokenizer, or null when the vocabulary has no single token for one.
- **architecture** — layer count, hidden size and per-layer attention types
  from the repo config (``text_config``-nested for the multimodal Gemma 3s).

THE RECORD (``prompts/models/<owner>--<repo>@<revision>.json``) carries the
probe results under ``detected``, the template's sha256 and the
``tokenizer_config.json`` sha256, which engine probed and when, an optional
human ``overrides`` block (``{field: {value, reason, setAt}}`` — displayed
beside the detected value and stamped into runs, never silent), and its own
``recordHash`` over the canonical JSON of everything else. ``effective()`` is
detected-with-overrides-applied, and is what every consumer reads.

THE FALLBACK. With no tokenizer to probe (an authoring client, a test
double, an offline gate before the model was ever installed) the old id
heuristics still answer, but as a record whose ``source`` is ``heuristic``
and whose ``advisories`` say so — never as something a reader could mistake
for a probed fact. Every consumer that has a tokenizer in hand probes it.

Import discipline: stdlib only at module level. ``jinja2`` is imported
inside :func:`render_template_text` (the fixture/test renderer) and
``transformers``/``huggingface_hub`` inside :func:`probe_model` (the
weights-free engine probe), so the authoring client never pays for either.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import paths

SCHEMA_VERSION = 1

#: Where a workspace keeps its records, under ``prompts/`` because a record
#: is recipe-side data (git-tracked, hashed, shared by both engines) rather
#: than a run output.
DIRECTORY = os.path.join("prompts", "models")

#: The reasoning-effort levels the probe tries, in the fixed cross-engine
#: order. Every level a template could accept is a probe candidate whether or
#: not any known template accepts it today; ``high`` is here because a probe
#: that never asked could never record the answer.
EFFORT_CANDIDATES: tuple[str, ...] = ("low", "medium", "high", "xhigh")

#: The value no template accepts, used to settle whether ``reasoning_effort``
#: is read at all before any real candidate is judged.
EFFORT_PROBE_VALUE = "steerlab-probe-effort"

SYSTEM_TURN = "systemTurn"
FOLDED_INTO_USER = "foldedIntoUser"
UNSUPPORTED = "unsupported"
SYSTEM_ROLES = (SYSTEM_TURN, FOLDED_INTO_USER, UNSUPPORTED)

SUPPORTED = "supported"
THINKING_SWITCHES = (SUPPORTED, UNSUPPORTED)

ACCEPTED = "accepted"
REJECTED = "rejected"
IGNORED = "ignored"
#: A heuristic record's verdict on a level it never probed but the old id
#: rule assumed — allowed by the gates with an advisory, never silently.
ASSUMED = "assumed"
EFFORT_VERDICTS = (ACCEPTED, REJECTED, IGNORED, ASSUMED)

SOURCE_PROBE = "probe"
SOURCE_HEURISTIC = "heuristic"

#: The fields a human may override, each with the closed vocabulary its value
#: must come from (``None`` = a boolean).
OVERRIDABLE_FIELDS: dict[str, tuple | None] = {
    "systemRole": SYSTEM_ROLES,
    "thinkingSwitch": THINKING_SWITCHES,
    "thinkOpenInPrompt": None,
}

#: Probe texts: single ASCII words no template transforms, distinct from each
#: other and from anything a template emits on its own.
PROBE_USER_TEXT = "steerlab-probe-user"
PROBE_SYSTEM_TEXT = "steerlab-probe-system"
THINK_OPEN_TOKEN = "<think>"
THINK_CLOSE_TOKEN = "</think>"


class RecordError(ValueError):
    """A malformed record, or a verb argument the record cannot take."""


# --- canonical form and hashing --------------------------------------------

def canonical_json(value) -> str:
    """The canonical text both engines hash: sorted keys (by code point —
    which is byte order for the record's ASCII keys), compact separators,
    raw UTF-8. Swift twin: ``ModelCapabilities.canonicalJSON``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def record_hash(record: dict) -> str:
    """sha256 over the canonical JSON of the record minus ``recordHash``."""
    payload = {k: v for k, v in record.items() if k != "recordHash"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_sha256(chat_template) -> str | None:
    """The hash of the template SOURCE the engine rendered with: a string
    as-is; a named-template list/dict in canonical JSON; None when absent."""
    if chat_template is None:
        return None
    if isinstance(chat_template, str):
        return sha256_text(chat_template)
    return sha256_text(canonical_json(chat_template))


# --- the record ------------------------------------------------------------

@dataclass(frozen=True)
class Capabilities:
    """A record's EFFECTIVE view — detected facts with overrides applied —
    plus the provenance every consumer stamps or displays.

    Read this through :func:`effective`; the raw record dict is the
    on-disk/wire form and keeps ``detected`` and ``overrides`` apart.
    """
    model_id: str
    revision: str | None
    source: str
    system_role: str
    fold_separator: str | None
    thinking_switch: str
    think_open_in_prompt: bool | None
    effort_variable_read: bool | None
    effort_levels: dict | None
    think_tokens: dict
    architecture: dict | None
    template_sha256: str | None
    tokenizer_config_sha256: str | None
    overrides: dict = field(default_factory=dict)
    advisories: tuple = ()
    record_hash: str | None = None
    #: Workspace-relative record path when the view came from a file.
    path: str | None = None

    @property
    def is_probed(self) -> bool:
        return self.source == SOURCE_PROBE

    @property
    def has_system_role(self) -> bool:
        return self.system_role == SYSTEM_TURN

    @property
    def system_prompt_deliverable(self) -> bool:
        return self.system_role != UNSUPPORTED

    @property
    def has_thinking_switch(self) -> bool:
        return self.thinking_switch == SUPPORTED

    @property
    def accepted_efforts(self) -> list[str]:
        levels = self.effort_levels or {}
        return [v for v in EFFORT_CANDIDATES if levels.get(v) == ACCEPTED]

    def effort_verdict(self, effort: str) -> str | None:
        """``accepted`` / ``rejected`` / ``ignored`` / ``assumed`` for one
        level, or None when this record never judged it."""
        return (self.effort_levels or {}).get(effort)

    def stamp(self) -> dict:
        """The compact provenance a run's ``config.json`` carries under
        ``notes.modelCapabilities`` — enough to know what the render did and
        which record said so, without copying the record. Swift twin:
        ``ModelCapabilities.stamp``."""
        return {
            "source": self.source,
            "record": self.path,
            "recordHash": self.record_hash,
            "templateSha256": self.template_sha256,
            "systemRole": self.system_role,
            "thinkingSwitch": self.thinking_switch,
            "thinkOpenInPrompt": self.think_open_in_prompt,
            "effortVariableRead": self.effort_variable_read,
            "effortLevels": dict(self.effort_levels) if self.effort_levels else None,
            "overrides": {k: v.get("value") for k, v in (self.overrides or {}).items()},
        }

    def summary_lines(self) -> list[str]:
        """The preregistration's account: one line per fact, in the words
        both engines print. Swift twin: ``ModelCapabilities.summaryLines``."""
        levels = self.effort_levels or {}
        effort_text = (", ".join(f"{v} {levels[v]}" for v in EFFORT_CANDIDATES
                                 if v in levels) or "none")
        override_text = ", ".join(
            f"{k}={v.get('value')!s} ({v.get('reason')})"
            for k, v in sorted((self.overrides or {}).items())) or "none"
        return [
            f"- **Model capabilities:** source {self.source}"
            + (f", record `{self.path}`" if self.path else "")
            + (f" (hash `{self.record_hash}`)" if self.record_hash else ""),
            f"- **System role:** {self.system_role}"
            + (f" (separator {self.fold_separator!r})"
               if self.system_role == FOLDED_INTO_USER else ""),
            f"- **Thinking switch:** {self.thinking_switch}"
            + (f", opening tag in prompt {str(self.think_open_in_prompt).lower()}"
               if self.thinking_switch == SUPPORTED
               and self.think_open_in_prompt is not None else ""),
            f"- **Reasoning effort levels:** {effort_text}",
            f"- **Capability overrides:** {override_text}",
        ]


def effective(record: dict, *, path: str | None = None) -> Capabilities:
    """The effective view of a record dict (overrides applied)."""
    detected = dict(record.get("detected") or {})
    overrides = dict(record.get("overrides") or {})
    for key, entry in overrides.items():
        if isinstance(entry, dict) and "value" in entry:
            detected[key] = entry["value"]
    template = record.get("template") or {}
    return Capabilities(
        model_id=str(record.get("modelID") or ""),
        revision=record.get("revision"),
        source=str(record.get("source") or SOURCE_HEURISTIC),
        system_role=str(detected.get("systemRole") or SYSTEM_TURN),
        fold_separator=detected.get("foldSeparator"),
        thinking_switch=str(detected.get("thinkingSwitch") or UNSUPPORTED),
        think_open_in_prompt=detected.get("thinkOpenInPrompt"),
        effort_variable_read=detected.get("effortVariableRead"),
        effort_levels=(dict(detected["effortLevels"])
                       if isinstance(detected.get("effortLevels"), dict) else None),
        think_tokens=dict(detected.get("thinkTokens") or {"open": None, "close": None}),
        architecture=(dict(detected["architecture"])
                      if isinstance(detected.get("architecture"), dict) else None),
        template_sha256=template.get("sha256"),
        tokenizer_config_sha256=template.get("tokenizerConfigSha256"),
        overrides=overrides,
        advisories=tuple(record.get("advisories") or ()),
        record_hash=record.get("recordHash"),
        path=path)


def validate_record(record: dict) -> list[str]:
    """Shape problems of a record dict, as sentences; empty when well-formed.
    The same checks the pinned JSON schema states, applied without a schema
    library so the authoring client stays dependency-free."""
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    if record.get("schemaVersion") != SCHEMA_VERSION:
        problems.append(f"schemaVersion must be {SCHEMA_VERSION}")
    if not isinstance(record.get("modelID"), str) or not record["modelID"]:
        problems.append("modelID must be a non-empty string")
    if record.get("source") not in (SOURCE_PROBE, SOURCE_HEURISTIC):
        problems.append("source must be probe or heuristic")
    detected = record.get("detected")
    if not isinstance(detected, dict):
        return problems + ["detected must be an object"]
    if detected.get("systemRole") not in SYSTEM_ROLES:
        problems.append("detected.systemRole must be one of " + ", ".join(SYSTEM_ROLES))
    if detected.get("thinkingSwitch") not in THINKING_SWITCHES:
        problems.append("detected.thinkingSwitch must be supported or unsupported")
    levels = detected.get("effortLevels")
    if levels is not None:
        if not isinstance(levels, dict):
            problems.append("detected.effortLevels must be an object or null")
        else:
            for key, value in levels.items():
                if key not in EFFORT_CANDIDATES or value not in EFFORT_VERDICTS:
                    problems.append(
                        f"detected.effortLevels.{key}={value!r} is not a probe "
                        "candidate with a known verdict")
    for key, entry in (record.get("overrides") or {}).items():
        if key not in OVERRIDABLE_FIELDS:
            problems.append(f"overrides.{key} is not an overridable field")
            continue
        if not isinstance(entry, dict) or not entry.get("reason"):
            problems.append(f"overrides.{key} needs a value and a reason")
            continue
        vocabulary = OVERRIDABLE_FIELDS[key]
        value = entry.get("value")
        if vocabulary is None:
            if not isinstance(value, bool):
                problems.append(f"overrides.{key}.value must be a boolean")
        elif value not in vocabulary:
            problems.append(f"overrides.{key}.value must be one of " + ", ".join(vocabulary))
    if "recordHash" in record and record.get("recordHash") != record_hash(record):
        problems.append("recordHash does not match the record's canonical bytes")
    return problems


# --- the probe -------------------------------------------------------------

class _Raised(Exception):
    """The template refused a render (``raise_exception`` or any error)."""


def _render_or_raised(render, messages: list[dict], **context):
    try:
        text = render(messages, **context)
    except Exception as exc:  # noqa: BLE001 - any refusal is the fact recorded
        return _Raised(str(exc))
    if not isinstance(text, str):
        return _Raised(f"renderer returned {type(text).__name__}, not text")
    return text


def probe(render, *, model_id: str, revision: str | None = None,
          think_token_id=None, architecture: dict | None = None,
          template_sha256: str | None = None,
          tokenizer_config_sha256: str | None = None,
          engine: str | None = None, engine_version: str | None = None,
          probed_at: str | None = None) -> dict:
    """Derive a record from a renderer.

    ``render(messages, **context) -> str`` renders a chat WITH a generation
    prompt under extra template variables, raising on a template refusal —
    the engine's own ``apply_chat_template`` on the server, the swift-
    transformers renderer on the Mac, a Jinja fixture in tests.
    ``think_token_id(token) -> int | None`` answers the single-token id of a
    tag, or None. Pure over its arguments: the same renderer yields the same
    record bytes (the ``probedBy.at`` stamp aside), which is what lets a
    fixture pin the outcome on both engines.
    """
    user = [{"role": "user", "content": PROBE_USER_TEXT}]
    with_system = [{"role": "system", "content": PROBE_SYSTEM_TEXT}] + user
    detected: dict = {}

    # -- systemRole -------------------------------------------------------
    plain = _render_or_raised(render, user)
    if isinstance(plain, _Raised):
        raise RecordError(
            f"the chat template of {model_id} refuses a plain user turn: {plain}")
    framed = _render_or_raised(render, with_system)
    fold_separator = None
    if isinstance(framed, _Raised):
        # Engine-neutral on purpose: the two Jinja engines spell the
        # exception differently, and the detail is pinned cross-engine.
        system_role, detail = UNSUPPORTED, "the template raises on a system turn"
    elif PROBE_SYSTEM_TEXT not in framed:
        system_role, detail = UNSUPPORTED, "the template silently drops a system turn"
    else:
        user_index = plain.find(PROBE_USER_TEXT)
        prefix = plain[:user_index] if user_index >= 0 else ""
        folded = (bool(prefix) and framed.startswith(prefix)
                  and framed.find(PROBE_SYSTEM_TEXT) >= len(prefix) - 0
                  and framed.find(PROBE_SYSTEM_TEXT) < framed.find(PROBE_USER_TEXT))
        if folded:
            start = framed.find(PROBE_SYSTEM_TEXT) + len(PROBE_SYSTEM_TEXT)
            end = framed.find(PROBE_USER_TEXT)
            fold_separator = framed[start:end]
            system_role, detail = FOLDED_INTO_USER, None
            # The fold must be exactly what a hand fold produces, or the
            # renderer cannot reproduce it by prepending: check the round
            # trip so a template that also rewrites the text is not misfiled.
            hand = _render_or_raised(
                render, [{"role": "user",
                          "content": PROBE_SYSTEM_TEXT + fold_separator + PROBE_USER_TEXT}])
            if hand != framed:
                system_role, detail = SYSTEM_TURN, (
                    "system text appears inside the user turn but not as a "
                    "plain prefix; treated as a system turn")
                fold_separator = None
        else:
            system_role, detail = SYSTEM_TURN, None
    detected["systemRole"] = system_role
    detected["systemRoleDetail"] = detail
    detected["foldSeparator"] = fold_separator

    # -- thinkingSwitch ---------------------------------------------------
    on = _render_or_raised(render, user, enable_thinking=True)
    off = _render_or_raised(render, user, enable_thinking=False)
    switch = (SUPPORTED if isinstance(on, str) and isinstance(off, str) and on != off
              else UNSUPPORTED)
    detected["thinkingSwitch"] = switch
    detected["thinkOpenInPrompt"] = (
        on.rstrip().endswith(THINK_OPEN_TOKEN) if switch == SUPPORTED else None)

    # -- effortLevels -----------------------------------------------------
    if switch == SUPPORTED:
        bogus = _render_or_raised(render, user, enable_thinking=True,
                                  reasoning_effort=EFFORT_PROBE_VALUE)
        variable_read = isinstance(bogus, _Raised) or bogus != on
        levels: dict = {}
        for candidate in EFFORT_CANDIDATES:
            if not variable_read:
                levels[candidate] = IGNORED
                continue
            rendered = _render_or_raised(render, user, enable_thinking=True,
                                         reasoning_effort=candidate)
            levels[candidate] = REJECTED if isinstance(rendered, _Raised) else ACCEPTED
        detected["effortVariableRead"] = variable_read
        detected["effortLevels"] = levels
    else:
        detected["effortVariableRead"] = None
        detected["effortLevels"] = None

    # -- thinkTokens ------------------------------------------------------
    detected["thinkTokens"] = {
        "open": think_token_id(THINK_OPEN_TOKEN) if think_token_id else None,
        "close": think_token_id(THINK_CLOSE_TOKEN) if think_token_id else None,
    }
    detected["architecture"] = _architecture_block(architecture)

    record = {
        "schemaVersion": SCHEMA_VERSION,
        "modelID": model_id,
        "revision": revision,
        "source": SOURCE_PROBE,
        "probedBy": {
            "engine": engine or _this_engine(),
            "version": engine_version or _this_engine_version(),
            "at": probed_at or _now(),
        },
        "template": {
            "sha256": template_sha256,
            "tokenizerConfigSha256": tokenizer_config_sha256,
        },
        "detected": detected,
        "overrides": {},
        "advisories": [],
    }
    record["recordHash"] = record_hash(record)
    return record


def _architecture_block(architecture: dict | None) -> dict | None:
    if not architecture:
        return None
    layer_types = architecture.get("layerTypes")
    return {
        "layerCount": architecture.get("layerCount"),
        "hiddenSize": architecture.get("hiddenSize"),
        "layerTypes": list(layer_types) if isinstance(layer_types, (list, tuple)) else None,
    }


def _this_engine() -> str:
    from ..steering.vector_store import SUBSTRATE
    return SUBSTRATE


def _this_engine_version() -> str:
    from ..build_identity import engine_version
    return engine_version()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the heuristic fallback --------------------------------------------------

def heuristic_advisory(model_id: str) -> str:
    return (f"model capabilities for {model_id} derive from the model id, not "
            f"from its chat template — no probed record under {DIRECTORY}/; "
            "the declaration was gated on the old family rule (qwen → "
            "thinking switch + effort levels assumed, gemma → system text "
            "folded into the first user turn). Probe the pinned template: "
            "steerlab-server model capabilities <modelID> --probe, or on the "
            "Mac steerlab-cli model capabilities <modelID> --probe")


def heuristic(model_id: str, revision: str | None = None) -> dict:
    """Today's id heuristics, as a record that SAYS it is one.

    Reproduces the pre-record rules exactly — ``"qwen"`` ⇒ a thinking switch
    whose low/medium/xhigh levels are ASSUMED accepted (the renderer passed
    every one of them to every Qwen); ``"gemma"`` ⇒ system text folded with
    ``"\\n\\n"``; anything else ⇒ a system turn and no thinking mode — so a
    consumer with no tokenizer behaves as it always did, and its record
    stamps ``source: heuristic`` with the advisory.
    """
    lowered = (model_id or "").lower()
    is_qwen = "qwen" in lowered
    is_gemma = "gemma" in lowered
    record = {
        "schemaVersion": SCHEMA_VERSION,
        "modelID": model_id,
        "revision": revision,
        "source": SOURCE_HEURISTIC,
        "probedBy": None,
        "template": None,
        "detected": {
            "systemRole": FOLDED_INTO_USER if is_gemma else SYSTEM_TURN,
            "systemRoleDetail": "assumed from the model id" ,
            "foldSeparator": "\n\n" if is_gemma else None,
            "thinkingSwitch": SUPPORTED if is_qwen else UNSUPPORTED,
            "thinkOpenInPrompt": None,
            "effortVariableRead": None,
            "effortLevels": ({"low": ASSUMED, "medium": ASSUMED, "xhigh": ASSUMED}
                             if is_qwen else None),
            "thinkTokens": {"open": None, "close": None},
            "architecture": None,
        },
        "overrides": {},
        "advisories": [heuristic_advisory(model_id)],
    }
    record["recordHash"] = record_hash(record)
    return record


# --- workspace records -------------------------------------------------------

def directory(root: str | None = None) -> str:
    return os.path.join(root or paths.project_root(), DIRECTORY)


def record_filename(model_id: str, revision: str | None) -> str:
    """``<owner>--<repo>@<revision>.json``; ``unpinned`` stands in for an
    unknown revision so the name is always well-formed."""
    slug = model_id.strip().strip("/").replace("/", "--")
    slug = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in slug)
    return f"{slug}@{revision or 'unpinned'}.json"


def record_relpath(model_id: str, revision: str | None) -> str:
    return os.path.join(DIRECTORY, record_filename(model_id, revision))


def record_path(model_id: str, revision: str | None, root: str | None = None) -> str:
    return os.path.join(directory(root), record_filename(model_id, revision))


def write_record(record: dict, root: str | None = None) -> str:
    """Write (or overwrite) a record; returns the workspace-relative path.
    The hash is recomputed so a caller cannot write a stale one."""
    problems = validate_record({**record, "recordHash": record_hash(record)})
    if problems:
        raise RecordError("; ".join(problems))
    record = dict(record)
    record["recordHash"] = record_hash(record)
    path = record_path(record["modelID"], record.get("revision"), root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return record_relpath(record["modelID"], record.get("revision"))


def read_record(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    problems = validate_record(record)
    if problems:
        raise RecordError(f"{path}: " + "; ".join(problems))
    return record


def list_records(root: str | None = None) -> list[tuple[str, dict]]:
    """Every readable record in the workspace as ``(relpath, record)``,
    sorted by path. Unreadable files are skipped: a listing is a display
    surface, and the verb that READS a specific record names the problem."""
    base = directory(root)
    out: list[tuple[str, dict]] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        if not name.endswith(".json"):
            continue
        try:
            record = read_record(os.path.join(base, name))
        except (OSError, ValueError, RecordError):
            continue
        out.append((os.path.join(DIRECTORY, name), record))
    return out


def lookup(model_id: str, revision: str | None = None,
           root: str | None = None) -> Capabilities | None:
    """The workspace record for (model id, revision), as an effective view.

    Exact match first. Otherwise the model id's most recently probed record
    at ANY revision, with an advisory naming the substitution — a template
    rarely changes between revisions, but the reader is told which revision
    actually answered. None when the workspace holds nothing for the id.
    """
    exact = record_path(model_id, revision, root)
    if revision and os.path.isfile(exact):
        record = read_record(exact)
        return effective(record, path=record_relpath(model_id, revision))
    candidates = [(relpath, record) for relpath, record in list_records(root)
                  if record.get("modelID") == model_id]
    if not candidates:
        return None
    if revision is None and len(candidates) == 1:
        relpath, record = candidates[0]
        return effective(record, path=relpath)
    relpath, record = max(
        candidates,
        key=lambda item: ((item[1].get("probedBy") or {}).get("at") or ""))
    view = effective(record, path=relpath)
    note = (f"no capability record for {model_id} at revision "
            f"{revision or 'unpinned'}; using the record for revision "
            f"{record.get('revision') or 'unpinned'} ({relpath})")
    return _with_advisory(view, note)


def _with_advisory(view: Capabilities, note: str) -> Capabilities:
    from dataclasses import replace
    return replace(view, advisories=tuple(view.advisories) + (note,))


def resolve(model_id: str, revision: str | None = None,
            root: str | None = None) -> Capabilities:
    """The record a gate reads: the workspace record when one exists, else
    the heuristic (which carries its own advisory)."""
    found = lookup(model_id, revision, root)
    if found is not None:
        return found
    return effective(heuristic(model_id, revision))


def diff(old: dict, new: dict) -> list[str]:
    """One line per detected fact that changed between two records — what
    a re-probe after a template change prints."""
    lines: list[str] = []
    a = old.get("detected") or {}
    b = new.get("detected") or {}
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            lines.append(f"{key}: {a.get(key)!r} → {b.get(key)!r}")
    old_sha = (old.get("template") or {}).get("sha256")
    new_sha = (new.get("template") or {}).get("sha256")
    if old_sha != new_sha:
        lines.insert(0, f"template sha256: {old_sha} → {new_sha}")
    return lines


def set_override(record: dict, field_name: str, value, reason: str) -> dict:
    """A copy of ``record`` with one override set (``value=None`` clears it).
    Values are parsed from their wire spelling: the closed vocabulary for
    ``systemRole``/``thinkingSwitch``, ``true``/``false`` for the boolean."""
    if field_name not in OVERRIDABLE_FIELDS:
        raise RecordError(
            f"'{field_name}' is not an overridable capability — one of "
            + ", ".join(sorted(OVERRIDABLE_FIELDS)))
    updated = dict(record)
    overrides = dict(record.get("overrides") or {})
    if value is None or value == "":
        overrides.pop(field_name, None)
    else:
        vocabulary = OVERRIDABLE_FIELDS[field_name]
        if vocabulary is None:
            text = str(value).strip().lower()
            if text not in ("true", "false"):
                raise RecordError(f"{field_name} takes true or false — got {value!r}")
            parsed: object = text == "true"
        else:
            if value not in vocabulary:
                raise RecordError(
                    f"{field_name} takes one of " + ", ".join(vocabulary)
                    + f" — got {value!r}")
            parsed = value
        if not (reason or "").strip():
            raise RecordError(
                "an override needs a --reason: it is displayed beside the "
                "detected value and stamped into every run")
        overrides[field_name] = {"value": parsed, "reason": reason.strip(),
                                 "setAt": _now()}
    updated["overrides"] = overrides
    updated["recordHash"] = record_hash(updated)
    return updated


# --- the engine probe (tokenizer in hand) ------------------------------------

def _tokenizer_render(tokenizer):
    def render(messages, **context):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **context)
    return render


def _tokenizer_think_id(tokenizer):
    def think_token_id(token: str):
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception:  # noqa: BLE001 - a double that cannot answer
            return None
        ids = list(ids) if ids is not None else []
        return int(ids[0]) if len(ids) == 1 else None
    return think_token_id


def looks_probeable(tokenizer) -> bool:
    """Whether this tokenizer is a real template renderer rather than a test
    double: it carries a chat template of its own. A double that only fakes
    ``apply_chat_template`` keeps today's heuristic behaviour."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return bool(template.strip())
    return isinstance(template, (dict, list)) and bool(template)


def architecture_from_config(config) -> dict | None:
    """Layer count, hidden size and per-layer attention types from an HF
    config object (or a plain dict), reading the nested ``text_config`` of a
    multimodal repo when the top level has none."""
    if config is None:
        return None

    def get(source, name):
        if isinstance(source, dict):
            return source.get(name)
        return getattr(source, name, None)

    text = get(config, "text_config")
    out: dict = {"layerCount": None, "hiddenSize": None, "layerTypes": None}
    for source in (config, text):
        if source is None:
            continue
        for key, names in (("layerCount", ("num_hidden_layers", "n_layer", "num_layers")),
                           ("hiddenSize", ("hidden_size", "d_model", "n_embd")),
                           ("layerTypes", ("layer_types",))):
            if out[key] is not None:
                continue
            for name in names:
                value = get(source, name)
                if key == "layerTypes":
                    if isinstance(value, (list, tuple)) and value:
                        out[key] = [str(v) for v in value]
                        break
                elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    out[key] = value
                    break
    if all(v is None for v in out.values()):
        return None
    return out


def tokenizer_config_sha256(model_id: str, revision: str | None) -> str | None:
    """sha256 of the cached ``tokenizer_config.json`` bytes, or None when the
    hub cache holds no such file for the id (a local directory, a double)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        kwargs = {"revision": revision} if revision else {}
        found = try_to_load_from_cache(model_id, "tokenizer_config.json", **kwargs)
    except Exception:  # noqa: BLE001 - cache lookups must never sink a probe
        return None
    if not isinstance(found, str) or not os.path.isfile(found):
        return None
    with open(found, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def raw_config(model_id: str, revision: str | None) -> dict | None:
    """The repo's ``config.json`` as the FILE says it, from the hub cache —
    never a transformers config object, which DERIVES fields the file does
    not carry (``layer_types`` on Qwen3 and Gemma 3) and would make the two
    engines record different architectures for one snapshot. None when the
    cache holds no such file for the id."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        kwargs = {"revision": revision} if revision else {}
        found = try_to_load_from_cache(model_id, "config.json", **kwargs)
    except Exception:  # noqa: BLE001 - cache lookups must never sink a probe
        return None
    if not isinstance(found, str) or not os.path.isfile(found):
        return None
    try:
        with open(found, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def cached_revision(model_id: str, revision: str | None) -> str | None:
    """The 40-hex commit the hub cache resolves for ``revision`` (or
    ``main``), read off the cached ``tokenizer_config.json`` path."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return revision
    try:
        kwargs = {"revision": revision} if revision else {}
        found = try_to_load_from_cache(model_id, "tokenizer_config.json", **kwargs)
    except Exception:  # noqa: BLE001
        return revision
    if not isinstance(found, str):
        return revision
    parts = found.split(os.sep)
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts) and len(parts[index + 1]) == 40:
            return parts[index + 1]
    return revision


def probe_tokenizer(tokenizer, *, model_id: str, revision: str | None = None,
                    config=None) -> dict:
    """Probe a loaded tokenizer (and its config) into a record. The
    architecture block reads the cached ``config.json`` FILE when the hub
    cache holds one — the same bytes the Mac reads — and the given config
    object only as the fallback for a model loaded from a plain directory."""
    return probe(
        _tokenizer_render(tokenizer), model_id=model_id, revision=revision,
        think_token_id=_tokenizer_think_id(tokenizer),
        architecture=architecture_from_config(raw_config(model_id, revision) or config),
        template_sha256=template_sha256(getattr(tokenizer, "chat_template", None)),
        tokenizer_config_sha256=tokenizer_config_sha256(model_id, revision))


def probe_model(model_id: str, revision: str | None = None, *,
                local_files_only: bool = True) -> dict:
    """The weights-free engine probe: ``AutoTokenizer`` + ``AutoConfig`` on
    the pinned revision, no GPU, no model slot. Resolves the revision the
    cache actually holds so the record is named by a commit, never ``main``."""
    from transformers import AutoConfig, AutoTokenizer
    kwargs: dict = {"revision": revision} if revision else {}
    if local_files_only:
        kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    try:
        config = AutoConfig.from_pretrained(model_id, **kwargs)
    except Exception:  # noqa: BLE001 - the architecture block is optional
        config = None
    resolved = cached_revision(model_id, revision)
    return probe_tokenizer(tokenizer, model_id=model_id, revision=resolved,
                           config=config)


_MEMO: dict[tuple, Capabilities] = {}


def for_tokenizer(tokenizer, *, model_id: str, revision: str | None = None,
                  config=None) -> Capabilities:
    """The capabilities of a tokenizer in hand: probed and memoized per
    (model id, template sha256) when it carries a real template, else the
    heuristic. This is what the renderers call when no record was passed —
    the template itself answers, so a render can never disagree with the
    template it renders through."""
    if not looks_probeable(tokenizer):
        return effective(heuristic(model_id, revision))
    key = (model_id, template_sha256(getattr(tokenizer, "chat_template", None)))
    cached = _MEMO.get(key)
    if cached is not None:
        return cached
    try:
        record = probe_tokenizer(tokenizer, model_id=model_id, revision=revision,
                                 config=config)
    except RecordError:
        return effective(heuristic(model_id, revision))
    view = effective(record)
    _MEMO[key] = view
    return view


def forget_memo() -> None:
    _MEMO.clear()


def ensure_record(tokenizer, *, model_id: str, revision: str | None,
                  config=None, root: str | None = None, log=None) -> Capabilities:
    """The record a run reads AND leaves behind: probe the loaded tokenizer;
    write the workspace record when none exists; when one exists whose
    template hash differs from the live template, re-probe, print the diff
    loudly and overwrite (overrides survive). Returns the effective view with
    its workspace path, so the run stamps the file it agrees with."""
    _log = log or (lambda _line: None)
    if not looks_probeable(tokenizer):
        view = effective(heuristic(model_id, revision))
        for line in view.advisories:
            _log(f"ADVISORY: {line}")
        return view
    fresh = probe_tokenizer(tokenizer, model_id=model_id, revision=revision, config=config)
    path = record_path(model_id, revision, root)
    relpath = record_relpath(model_id, revision)
    if os.path.isfile(path):
        try:
            existing = read_record(path)
        except (OSError, ValueError, RecordError) as exc:
            _log(f"ADVISORY: capability record {relpath} is unreadable ({exc}); rewriting it")
            existing = None
        if existing is not None:
            same_template = ((existing.get("template") or {}).get("sha256")
                             == (fresh.get("template") or {}).get("sha256"))
            if same_template:
                return effective(existing, path=relpath)
            fresh["overrides"] = dict(existing.get("overrides") or {})
            fresh["recordHash"] = record_hash(fresh)
            _log(f"ADVISORY: the chat template of {model_id} changed since "
                 f"{relpath} was probed — re-probed; "
                 + "; ".join(diff(existing, fresh)))
    written = write_record(fresh, root)
    _log(f"probed model capabilities for {model_id} → {written}")
    return effective(read_record(path), path=written)


# --- template-text rendering (fixtures, tests, the regeneration script) ------

def render_template_text(template: str, messages: list[dict], *,
                         add_generation_prompt: bool = True,
                         special_tokens: dict | None = None, **context) -> str:
    """Render a chat-template SOURCE the way ``apply_chat_template`` does —
    transformers' own sandbox shape (``raise_exception``, ``tojson``,
    ``strftime_now``, loop controls, trimmed blocks) — with no tokenizer. The
    fixture families render through this on the Python side and through
    swift-jinja on the Mac, which is how the probe logic is pinned across
    engines without vendoring a model."""
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, **kwargs):
        return json.dumps(value, **kwargs)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = strftime_now
    compiled = environment.from_string(template)
    variables = dict(special_tokens or {})
    variables.update(context)
    return compiled.render(messages=messages,
                           add_generation_prompt=add_generation_prompt,
                           **variables)


def template_renderer(template: str, special_tokens: dict | None = None):
    """A ``render(messages, **context)`` closure over a template source."""
    def render(messages, **context):
        return render_template_text(template, messages,
                                    special_tokens=special_tokens, **context)
    return render
