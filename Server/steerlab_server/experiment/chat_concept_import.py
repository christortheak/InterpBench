"""Import a concept from two row-aligned chat-format JSONL files.

A concept's stimulus files are one ``{"text": …}`` per line
(``prompts/concepts/<name>/{positive,negative}.jsonl``). Much published
contrast data comes as CHAT rows instead — ``{"messages": [{"role": "user",
…}, {"role": "assistant", …}]}`` — in two files, one per pole, whose rows
answer the same prompts in the same order. This module turns such a pair of
files into a concept, and it is the line-for-line twin of
``Sources/ExperimentKit/ChatConceptImport.swift``: the same input and seed
produce the same bytes on both engines, pinned by the cross-engine fixture
``Tests/Fixtures/cross-engine/chat-concept-import.json``.

The rule, stated once (it is also written into every ``provenance.json``):

* Row *i* of the positive file is paired with row *i* of the negative file.
  Rows are the non-blank lines, counted from 1; a leading UTF-8 byte-order
  mark is ignored.
* A row must hold exactly one ``user`` message followed by one ``assistant``
  message; ``system`` messages are allowed and ignored, and any other role is
  refused. A row that is not valid UTF-8 or JSON is refused by file and line.
* A pair is DROPPED, not refused, when its two user prompts differ, when its
  two assistant replies are identical, or when either reply is empty (only
  spaces, tabs, and line breaks). Each drop is counted and its row recorded.
* The kept pairs' assistant replies are written verbatim to the same line of
  ``positive.jsonl`` and ``negative.jsonl``. Optionally, N kept pairs are held
  out instead as ``validation.jsonl`` rows (``{"text": …, "expresses": true}``
  then ``false``, per pair), chosen by the seeded draw the neutral token bank
  already shares across engines (SplitMix64, partial Fisher–Yates,
  :func:`~steerlab_server.steering.token_bank_downsampling.selected_indices`).
* ``concept.json`` and a ``provenance.json`` naming both source files, their
  SHA-256, the counts, the dropped rows, and the seed are written beside them.

**Why a strict parser.** Byte parity is decided as much by which rows are
REFUSED as by what is written. Foundation's JSON reader accepts trailing
commas and a byte-order mark mid-file; Python's accepts lone surrogates,
``NaN``, and numbers that overflow to infinity. Both engines therefore apply
the same narrower rule: RFC 8259 JSON with no lone surrogates, no non-finite
numbers, nesting at most :data:`MAX_DEPTH` deep, and the LAST of any repeated
key. Text is compared code point by code point, never by Unicode
equivalence.

Nothing here is bundled data. The importer reads files a researcher points it
at and writes into their workspace; it downloads nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone

from . import paths
from ..steering.token_bank_downsampling import selected_indices

#: The importer's identity, stamped into ``provenance.json``. A change to what
#: this module writes for the same input is a new identifier.
IMPORTER = "chat-pairs-v1"
#: How the held-out pairs are drawn — the neutral token bank's shared draw.
DRAW = "splitmix64-partial-fisher-yates"
#: The largest seed: every integer up to it is exact in a JSON reader that
#: holds numbers as doubles, so the recorded seed is the seed that was used.
MAX_SEED = (1 << 53) - 1
MAX_NAME_LENGTH = 64
#: Arrays and objects nested deeper than this make a row unreadable.
MAX_DEPTH = 64

PROVENANCE_ARTIFACT = "steerlab-concept-import"
CONCEPT_ARTIFACT = "steerlab-concept"

RULE = ("Row i of the positive file is paired with row i of the negative "
        "file, counting non-blank lines from 1. A pair is dropped when its "
        "two user prompts differ, when its two assistant replies are "
        "identical, or when either reply is empty. Each kept pair's assistant "
        "replies are written to the same line of positive.jsonl and "
        "negative.jsonl; held-out pairs go to validation.jsonl instead, the "
        "positive reply then the negative one.")

ROW_SHAPE = ('each line must be one JSON object with a "messages" list holding '
             'one {"role": "user", "content": "…"} message and then one '
             '{"role": "assistant", "content": "…"} message; "system" '
             'messages are allowed and ignored')

_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_BLANK_BYTES = frozenset(b" \t\r")
_EMPTY_REPLY = frozenset(" \t\n\r")
_DIGITS = re.compile(r"[0-9]+")
_BOM = b"\xef\xbb\xbf"


class ChatImportRefusal(Exception):
    """A typed refusal: ``code`` and ``state`` for the envelope, a plain
    ``reason``, and a ``repair_action`` the reader can carry out.

    ``state`` is ``refused`` (65) for data the importer declines,
    ``notFound`` (66) for a missing source file, and ``blocked`` (64) for a
    malformed request (a name or a flag value)."""

    def __init__(self, code: str, reason: str, repair_action: str,
                 state: str = "refused") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.repair_action = repair_action
        self.state = state


# --- formatting shared with the Swift twin ----------------------------------


def timestamp(epoch: float) -> str:
    """UTC, whole seconds: the ``createdAt`` shape every concept.json has."""
    return datetime.fromtimestamp(math.floor(epoch), tz=timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


# Line-separating characters a reader may split on (Python's
# ``str.splitlines`` breaks on all three; JSON allows them raw). Escaped, so
# every row stays one line whatever reads it.
_LINE_SEPARATORS = {"\u0085": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def json_string(text: str) -> str:
    """One JSON string literal, non-ASCII kept as UTF-8 — the escaping both
    engines write (``"``, ``\\``, ``\\b\\f\\n\\r\\t``, other controls as
    lower-case ``\\u00xx``, and U+0085, U+2028, and U+2029 as ``\\u0085``,
    ``\\u2028``, ``\\u2029``)."""
    literal = json.dumps(text, ensure_ascii=False)
    for raw, escaped in _LINE_SEPARATORS.items():
        literal = literal.replace(raw, escaped)
    return literal


def text_row(text: str) -> str:
    return '{"text": ' + json_string(text) + "}\n"


def validation_row(text: str, expresses: bool) -> str:
    return ('{"text": ' + json_string(text) + ', "expresses": '
            + ("true" if expresses else "false") + "}\n")


def concept_document(name: str, created_at: str) -> str:
    """The concept primitive, in the exact shape the app writes it."""
    return ("{\n"
            f'  "artifactType" : "{CONCEPT_ARTIFACT}",\n'
            f'  "createdAt" : "{created_at}",\n'
            f'  "name" : "{name}",\n'
            '  "schemaVersion" : 1\n'
            "}")


def provenance_document(provenance: dict) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the request ---------------------------------------------------------------


def suggested_name(raw: str) -> str:
    """A usable spelling of ``raw``, or "" when nothing of it survives."""
    kept = []
    for ch in raw:
        if "A" <= ch <= "Z":
            kept.append(ch.lower())
        elif "a" <= ch <= "z" or "0" <= ch <= "9":
            kept.append(ch)
        elif ch in " _-.":
            kept.append("-")
    text = re.sub("-+", "-", "".join(kept)).strip("-")
    return text[:MAX_NAME_LENGTH].strip("-")


def validate_name(raw: str) -> str:
    if len(raw) <= MAX_NAME_LENGTH and _NAME_RE.fullmatch(raw):
        return raw
    suggestion = suggested_name(raw)
    repair = ("use lowercase letters, digits, and single hyphens, at most "
              f"{MAX_NAME_LENGTH} characters")
    if suggestion and suggestion != raw:
        repair += f", e.g. {suggestion}"
    raise ChatImportRefusal(
        "usage", f"'{raw}' is not a usable concept name", repair,
        state="blocked")


def parse_count(text: str) -> int:
    """``--validation-pairs``: a whole number."""
    if not _DIGITS.fullmatch(text or "") or len(text) > 9:
        raise ChatImportRefusal(
            "usage",
            f"--validation-pairs expects a whole number, got '{text}'",
            "--validation-pairs <n>, the number of pairs to hold out",
            state="blocked")
    return int(text)


def parse_seed(text: str) -> int:
    """``--seed``: a whole number from 0 to :data:`MAX_SEED`."""
    if (not _DIGITS.fullmatch(text or "") or len(text) > 16
            or int(text) > MAX_SEED):
        raise ChatImportRefusal(
            "usage",
            f"--seed expects a whole number from 0 to {MAX_SEED}, got '{text}'",
            f"--seed <n>, a whole number from 0 to {MAX_SEED}",
            state="blocked")
    return int(text)


def check_holdout(validation_pairs: int, seed: int | None) -> None:
    """A held-out draw needs its seed, and a seed needs a draw."""
    if validation_pairs > 0 and seed is None:
        raise ChatImportRefusal(
            "usage",
            "--validation-pairs needs --seed: the held-out pairs are drawn "
            "with a seed that is recorded in provenance.json",
            f"add --seed <n>, a whole number from 0 to {MAX_SEED}",
            state="blocked")
    if validation_pairs == 0 and seed is not None:
        raise ChatImportRefusal(
            "usage",
            "--seed draws the held-out pairs, and no pairs are held out",
            "add --validation-pairs <n>, or leave out --seed",
            state="blocked")


# --- reading rows ---------------------------------------------------------------


class _Invalid(Exception):
    pass


def _ignore_int(_text: str) -> int:
    # Integers are validated by the grammar and never used. Not converting
    # them keeps a 5000-digit literal from tripping CPython's int-size limit,
    # which the Swift twin has no counterpart for.
    return 0


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Invalid()
    return value


def _no_constant(_text: str):
    raise _Invalid()


def _too_deep(text: str) -> bool:
    """Whether arrays/objects nest past :data:`MAX_DEPTH` — scanned BEFORE the
    parse, so a hostile row is refused rather than recursed into."""
    depth = 0
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                return True
        elif ch in "]}":
            depth -= 1
    return False


def _strings_are_scalars(value) -> bool:
    """False when any string holds a lone surrogate (``"\\ud800"``), which
    has no UTF-8 form and which Foundation's reader refuses."""
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _parse(text: str):
    if _too_deep(text):
        raise _Invalid()
    try:
        value = json.loads(text, parse_int=_ignore_int,
                           parse_float=_finite_float,
                           parse_constant=_no_constant)
    except (ValueError, RecursionError, _Invalid):
        raise _Invalid() from None
    if not _strings_are_scalars(value):
        raise _Invalid()
    return value


def _unreadable(file: str, line: int, problem: str) -> ChatImportRefusal:
    return ChatImportRefusal(
        "chatRowUnreadable", f"{file} line {line} {problem}",
        f"fix {file} line {line} ({ROW_SHAPE}), then import again")


def _roles(file: str, line: int, reason: str) -> ChatImportRefusal:
    return ChatImportRefusal(
        "chatRowRoles", reason,
        f"fix {file} line {line} ({ROW_SHAPE}), then import again")


def read_rows(data: bytes, file: str) -> tuple[list[tuple[str, str]], int]:
    """``([(user prompt, assistant reply)], rows carrying a system message)``,
    refusing the first unreadable or wrongly shaped row."""
    if data.startswith(_BOM):
        data = data[len(_BOM):]
    rows: list[tuple[str, str]] = []
    with_system = 0
    for index, raw in enumerate(data.split(b"\n")):
        line = index + 1
        if all(byte in _BLANK_BYTES for byte in raw):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise _unreadable(file, line, "is not UTF-8 text") from None
        try:
            value = _parse(text)
        except _Invalid:
            raise _unreadable(file, line, "is not valid JSON") from None
        if not isinstance(value, dict):
            raise _unreadable(file, line, "is not a JSON object")
        messages = value.get("messages")
        if not isinstance(messages, list):
            raise _unreadable(file, line, 'has no "messages" list')
        prompt = reply = None
        users = assistants = 0
        user_at = assistant_at = 0
        has_system = False
        for position, message in enumerate(messages, 1):
            if (not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)):
                raise ChatImportRefusal(
                    "chatRowUnreadable",
                    f'{file} line {line}: message {position} needs a "role" '
                    'string and a "content" string',
                    f"fix {file} line {line} ({ROW_SHAPE}), then import again")
            role = message["role"]
            if role == "user":
                users += 1
                user_at = position
                prompt = message["content"]
            elif role == "assistant":
                assistants += 1
                assistant_at = position
                reply = message["content"]
            elif role == "system":
                has_system = True
            else:
                raise _roles(
                    file, line,
                    f"{file} line {line}: message {position} has role "
                    f'{json_string(role)}; only "system", "user", and '
                    '"assistant" are read')
        if users != 1 or assistants != 1:
            raise _roles(
                file, line,
                f"{file} line {line} has {users} user and {assistants} "
                "assistant messages; each row needs exactly one of each")
        if assistant_at < user_at:
            raise _roles(
                file, line,
                f"{file} line {line} puts its assistant message before its "
                "user message")
        rows.append((prompt, reply))
        if has_system:
            with_system += 1
    return rows, with_system


def _is_empty(reply: str) -> bool:
    return all(ch in _EMPTY_REPLY for ch in reply)


# --- the import -------------------------------------------------------------


def prepare(name: str, positive: bytes, negative: bytes, *,
            positive_file: str, negative_file: str,
            validation_pairs: int = 0, seed: int | None = None,
            created_at: str) -> dict:
    """Everything the import writes, computed and checked, nothing written.

    Returns ``{"name", "files": [(filename, bytes)], "provenance",
    "stimulusSetHash", "validationHash"}``; ``files`` is in write order.
    """
    name = validate_name(name)
    check_holdout(validation_pairs, seed)
    positive_rows, positive_system = read_rows(positive, positive_file)
    negative_rows, negative_system = read_rows(negative, negative_file)
    if len(positive_rows) != len(negative_rows):
        raise ChatImportRefusal(
            "rowCountMismatch",
            f"{positive_file} has {len(positive_rows)} rows and "
            f"{negative_file} has {len(negative_rows)}; the two files must "
            "pair up row by row",
            "import two files that answer the same prompts in the same order, "
            "one pair per row: check that neither file was cut short and that "
            "both come from the same release")

    dropped = {"promptMismatch": [], "identicalReplies": [], "emptyReply": []}
    kept: list[tuple[int, str, str]] = []
    for row, (pos, neg) in enumerate(zip(positive_rows, negative_rows), 1):
        if pos[0] != neg[0]:
            dropped["promptMismatch"].append(row)
        elif pos[1] == neg[1]:
            dropped["identicalReplies"].append(row)
        elif _is_empty(pos[1]) or _is_empty(neg[1]):
            dropped["emptyReply"].append(row)
        else:
            kept.append((row, pos[1], neg[1]))
    dropped_count = sum(len(rows) for rows in dropped.values())

    if not kept:
        raise ChatImportRefusal(
            "noUsablePairs",
            f"no usable pairs: {len(positive_rows)} rows read, "
            f"{len(dropped['promptMismatch'])} with different user prompts, "
            f"{len(dropped['identicalReplies'])} with identical replies, "
            f"{len(dropped['emptyReply'])} with an empty reply",
            "check that the two files are the positive and negative halves "
            "of one dataset, in the same row order, with the concept-present "
            "replies in the positive file")
    if validation_pairs >= len(kept):
        raise ChatImportRefusal(
            "validationTooLarge",
            f"holding out {validation_pairs} pairs leaves none for "
            f"extraction: {len(kept)} pairs are usable",
            f"set --validation-pairs below {len(kept)}, or leave it out")

    held = set(selected_indices(len(kept), validation_pairs, seed)
               if validation_pairs > 0 else [])
    training = [pair for index, pair in enumerate(kept) if index not in held]
    held_out = [pair for index, pair in enumerate(kept) if index in held]

    positive_bytes = "".join(text_row(p) for _, p, _ in training).encode()
    negative_bytes = "".join(text_row(n) for _, _, n in training).encode()
    files = [("concept.json", concept_document(name, created_at).encode()),
             ("positive.jsonl", positive_bytes),
             ("negative.jsonl", negative_bytes)]
    validation_hash = None
    if held_out:
        validation_bytes = "".join(
            validation_row(p, True) + validation_row(n, False)
            for _, p, n in held_out).encode()
        files.append(("validation.jsonl", validation_bytes))
        validation_hash = sha256(validation_bytes)

    def source(data: bytes, file: str, rows: int, system: int) -> dict:
        return {"bytes": len(data), "file": file, "rows": rows,
                "sha256": sha256(data), "systemMessages": system}

    provenance = {
        "artifactType": PROVENANCE_ARTIFACT,
        "concept": name,
        "dropped": dropped,
        "importedAt": created_at,
        "importer": IMPORTER,
        "pairs": {"dropped": dropped_count,
                  "extraction": len(training),
                  "kept": len(kept),
                  "rows": len(positive_rows),
                  "validation": len(held_out)},
        "rule": RULE,
        "schemaVersion": 1,
        "sources": {
            "negative": source(negative, negative_file, len(negative_rows),
                               negative_system),
            "positive": source(positive, positive_file, len(positive_rows),
                               positive_system),
        },
        "validation": ({"draw": DRAW, "pairs": len(held_out),
                        "rows": [row for row, _, _ in held_out],
                        "seed": seed} if held_out else None),
    }
    files.append(("provenance.json",
                  provenance_document(provenance).encode()))
    return {"name": name, "files": files, "provenance": provenance,
            "stimulusSetHash": sha256(positive_bytes + negative_bytes),
            "validationHash": validation_hash}


def _exists(name: str) -> ChatImportRefusal:
    return ChatImportRefusal(
        "conceptExists", f"prompts/concepts/{name}/ already exists",
        "choose a concept name that is not already in prompts/concepts/ — "
        "the importer never adds to or replaces an existing concept")


def _read(path: str, side: str) -> bytes:
    if not os.path.exists(path):
        raise ChatImportRefusal(
            "notFound", f"no file at {os.path.abspath(path)}",
            f"check the path to the {side} chat file", state="notFound")
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        raise ChatImportRefusal(
            "chatFileUnreadable", f"could not read {os.path.abspath(path)}",
            f"check that the {side} path names a readable file") from None


def import_concept(name: str, positive_path: str, negative_path: str, *,
                   root: str, validation_pairs: int = 0,
                   seed: int | None = None, now: float | None = None) -> dict:
    """Read, check, and write one concept. Refuses before writing anything;
    the files land together by one rename of a staging directory."""
    name = validate_name(name)
    check_holdout(validation_pairs, seed)
    concepts = paths.concepts_directory(root)
    destination = os.path.join(concepts, name)
    if os.path.lexists(destination):
        raise _exists(name)
    positive = _read(positive_path, "positive")
    negative = _read(negative_path, "negative")
    prepared = prepare(
        name, positive, negative,
        positive_file=os.path.basename(positive_path),
        negative_file=os.path.basename(negative_path),
        validation_pairs=validation_pairs, seed=seed,
        created_at=timestamp(time.time() if now is None else now))

    os.makedirs(concepts, exist_ok=True)
    staging = os.path.join(concepts,
                           f".steerlab-chat-import-{uuid.uuid4().hex}")
    os.mkdir(staging)
    try:
        for filename, data in prepared["files"]:
            with open(os.path.join(staging, filename), "wb") as handle:
                handle.write(data)
        if os.path.lexists(destination):
            raise _exists(name)
        try:
            os.rename(staging, destination)
        except OSError:
            if os.path.lexists(destination):
                raise _exists(name) from None
            raise
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)

    relative = f"prompts/concepts/{name}"
    prepared["directory"] = relative
    prepared["writtenFiles"] = [f"{relative}/{filename}"
                                for filename, _ in prepared["files"]]
    return prepared


def result_payload(outcome: dict) -> dict:
    """The envelope's ``result`` — the same keys on both clients."""
    return {"concept": outcome["name"],
            "directory": outcome["directory"],
            "files": outcome["writtenFiles"],
            "provenance": outcome["provenance"],
            "stimulusSetHash": outcome["stimulusSetHash"],
            "validationHash": outcome["validationHash"]}


def summary_lines(outcome: dict) -> list[str]:
    """What a person at a terminal reads. The first line is the envelope's
    message."""
    pairs = outcome["provenance"]["pairs"]
    dropped = outcome["provenance"]["dropped"]
    name = outcome["name"]
    lines = [f"imported '{name}' from chat pairs: {pairs['extraction']} for "
             f"extraction, {pairs['validation']} held out for validation, "
             f"{pairs['dropped']} dropped",
             f"wrote {outcome['directory']}/ ("
             + ", ".join(filename for filename, _ in outcome["files"]) + ")"]
    if pairs["dropped"]:
        lines.append(
            f"dropped {pairs['dropped']} of {pairs['rows']} pairs: "
            f"{len(dropped['promptMismatch'])} with different user prompts, "
            f"{len(dropped['identicalReplies'])} with identical replies, "
            f"{len(dropped['emptyReply'])} with an empty reply (their rows "
            "are listed in provenance.json)")
    if not pairs["validation"]:
        lines.append(
            f"no validation.jsonl was written: author "
            f"{outcome['directory']}/validation.jsonl before freezing a study "
            "that uses this concept, or import again under a new name with "
            "--validation-pairs and --seed")
    return lines
