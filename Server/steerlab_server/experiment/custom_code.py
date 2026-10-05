"""Custom code a study carries, and the recorded acknowledgement before it runs.

Everything a workspace holds is data, with one exception: an intervention
policy may carry an *expert provider*, Python source that the engine runs with
``exec`` while the study runs (``policy_execution.Execution``). Its author's
warning appears at authoring, but a study shared as a pack or a bundle carries
that code to a machine whose owner never saw the warning. This module is the
notice for that recipient:

* :func:`providers` finds every expert provider anywhere in a study document,
  including inside the exact policy bytes an agent embeds, and names each one
  by the SHA-256 of its source text, computed here rather than trusted from the
  document.
* :func:`acknowledge` records who acknowledged which source hash, and when, in
  ``custom-code-acknowledgements.json`` at the workspace root, so the notice is
  not repeated for the same code and a run can show that it was acknowledged.
* :func:`run_refusal` is the one gate: a step that can execute the study's
  agents does not start while a provider in it is unacknowledged.

Nothing here sandboxes anything, and nothing here may say it does. The code
runs with the permissions of whoever runs the study.

Swift twin: ``Sources/ExperimentKit/CustomCodeNotice.swift``. The file name,
the record keys, the notice sentence, and the executing verbs are duplicated
there on purpose; ``test_custom_code.py`` and ``CustomCodeNoticeTests`` hold
the two copies to the same literals.
"""
from __future__ import annotations

from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import manifest_files
from .manifest_errors import ExperimentStoreError

#: The workspace-root file both clients and the app read and write.
FILENAME = "custom-code-acknowledgements.json"
SCHEMA_VERSION = 1

#: The notice, verbatim on every surface that shows it.
NOTICE = ("This study contains custom code from its author. It runs with your "
          "permissions when the study runs. Run it only if you trust the "
          "source.")

#: Study steps that can generate text with the study's agents, and therefore
#: execute an expert provider. Every other step (extract, validate, evaluate,
#: analyze, verify) runs no agent policy and is never held by this gate.
EXECUTING_VERBS: tuple[str, ...] = ("pipeline", "run", "sweep")

#: The gate id a refusal carries. ``missingPrerequisite`` because the repair is
#: "supply the missing thing, then run again"; the specific facts travel in
#: the refusal's ``customCode`` block.
GATE = "missingPrerequisite"

_MAX_DEPTH = 64


class CustomCodeError(ExperimentStoreError):
    """A refusal about custom code or its acknowledgement record."""

    def __init__(self, message: str, *, repair: str, facts: dict | None = None):
        super().__init__(message, gate=GATE, repair=repair)
        self.facts = dict(facts or {})


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_digest(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def providers(document) -> list[dict]:
    """Every expert provider in ``document``, one entry per source hash.

    ``[{"sha256": <hex>, "policyNames": [<sorted names>]}]`` sorted by hash.
    Scans nested agents and panels, and parses each attached policy's exact
    ``json`` text, because that is where a study carries its agents'
    policies. The hash is computed from ``sourceText``; a document that
    declares a different ``sourceSHA256`` is refused by the engine anyway, and
    an acknowledgement must name the code itself."""
    found: dict[str, set] = {}
    _scan(document, found, 0)
    return [{"sha256": digest, "policyNames": sorted(found[digest])}
            for digest in sorted(found)]


def _scan(value, found: dict, depth: int) -> None:
    if depth > _MAX_DEPTH:
        return
    if isinstance(value, dict):
        provider = value.get("provider")
        if isinstance(provider, dict) and isinstance(provider.get("sourceText"), str):
            names = found.setdefault(_digest(provider["sourceText"]), set())
            if isinstance(value.get("name"), str) and value["name"]:
                names.add(value["name"])
        attached = value.get("interventionPolicies")
        if isinstance(attached, list):
            for item in attached:
                if isinstance(item, dict) and isinstance(item.get("json"), str):
                    try:
                        parsed = json.loads(item["json"])
                    except ValueError:
                        continue
                    _scan(parsed, found, depth + 1)
        for child in value.values():
            _scan(child, found, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _scan(child, found, depth + 1)


def provider_sources(document) -> dict[str, str]:
    """``{sha256: sourceText}`` for every provider, so a person can read the
    code before acknowledging it."""
    sources: dict[str, str] = {}

    def visit(value, depth):
        if depth > _MAX_DEPTH:
            return
        if isinstance(value, dict):
            provider = value.get("provider")
            if isinstance(provider, dict) and isinstance(provider.get("sourceText"), str):
                sources.setdefault(_digest(provider["sourceText"]), provider["sourceText"])
            for item in value.get("interventionPolicies") or []:
                if isinstance(item, dict) and isinstance(item.get("json"), str):
                    try:
                        visit(json.loads(item["json"]), depth + 1)
                    except ValueError:
                        pass
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    visit(document, 0)
    return sources


def record_path(root) -> Path:
    return Path(root) / FILENAME


def _unreadable(detail: str) -> CustomCodeError:
    return CustomCodeError(
        f"The custom code acknowledgement record ({FILENAME}) cannot be read: "
        f"{detail}. Nothing was run.",
        repair=(f"Restore {FILENAME} from the workspace history (it is an "
                "ordinary tracked file), or move it aside and acknowledge the "
                "study's custom code again."))


def records(root) -> list[dict]:
    """The recorded acknowledgements, oldest first. A missing file is an
    empty record; a damaged one is refused rather than read as empty, because
    an empty reading would silently re-show every notice."""
    try:
        raw = record_path(root).read_bytes()
    except FileNotFoundError:
        return []
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise _unreadable(f"it is not valid JSON ({exc})") from None
    if (not isinstance(document, dict)
            or document.get("schemaVersion") != SCHEMA_VERSION
            or not isinstance(document.get("acknowledgements"), list)):
        raise _unreadable("it is not a version-1 acknowledgement record")
    return [entry for entry in document["acknowledgements"]
            if isinstance(entry, dict) and _is_digest(entry.get("providerSHA256"))]


def acknowledged(root) -> dict[str, dict]:
    """``{providerSHA256: first acknowledgement}``."""
    result: dict[str, dict] = {}
    for entry in records(root):
        result.setdefault(entry["providerSHA256"], entry)
    return result


def status(document, root) -> list[dict]:
    """Each provider with whether, when, and by whom it was acknowledged."""
    seen = acknowledged(root)
    rows = []
    for provider in providers(document):
        entry = seen.get(provider["sha256"])
        rows.append({**provider, "acknowledged": entry is not None,
                     "acknowledgedAt": entry.get("acknowledgedAt") if entry else None,
                     "acknowledgedBy": entry.get("acknowledgedBy") if entry else None})
    return rows


def acknowledge_command(study: str, hashes, *, program: str) -> str:
    return (f"{program} experiment acknowledge-custom-code {study} "
            + " ".join(f"--sha256 {digest}" for digest in hashes)).strip()


def review_command(study: str, *, program: str) -> str:
    return f"{program} experiment acknowledge-custom-code {study}"


def notice(document, root, *, study: str, program: str) -> dict | None:
    """The ``customCode`` block a surface attaches when ``document`` carries
    custom code, or ``None`` when it carries none.

    ``notice`` is the sentence while any provider is unacknowledged and
    ``None`` once all are, which is what keeps the notice from repeating for
    code someone already acknowledged."""
    rows = status(document, root)
    if not rows:
        return None
    pending = [row["sha256"] for row in rows if not row["acknowledged"]]
    return {"notice": NOTICE if pending else None,
            "providers": rows,
            "acknowledged": not pending,
            "recordFile": FILENAME,
            "reviewCommand": review_command(study, program=program) if pending else None,
            "acknowledgeCommand": (acknowledge_command(study, pending, program=program)
                                   if pending else None)}


def notice_lines(block: dict | None) -> list[str]:
    """Human-mode lines for a ``customCode`` block (empty once acknowledged)."""
    if not block or not block.get("notice"):
        return []
    lines = [block["notice"]]
    for row in block["providers"]:
        if row["acknowledged"]:
            continue
        named = ", ".join(row["policyNames"]) or "unnamed policy"
        lines.append(f"  custom code SHA-256 {row['sha256']} (policy: {named})")
    lines.append(f"Read it: {block['reviewCommand']}")
    lines.append(f"If you trust the source: {block['acknowledgeCommand']}")
    return lines


def _account() -> str:
    try:
        return getpass.getuser() or "unknown"
    except Exception:   # noqa: BLE001 — no account name is not a reason to refuse
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def acknowledge(root, document, hashes, *, study: str, client: str,
                program: str, account: str | None = None,
                now: str | None = None) -> dict:
    """Record an acknowledgement for each named source hash the study carries.

    Idempotent: a hash already acknowledged keeps its first record. A hash the
    study does not carry is refused, so a typo can never acknowledge code
    nobody looked at."""
    carried = {row["sha256"]: row for row in providers(document)}
    if not carried:
        raise CustomCodeError(
            f"'{study}' carries no custom code, so there is nothing to "
            "acknowledge.",
            repair=f"{program} experiment verify {study}")
    requested = list(dict.fromkeys(hashes))
    unknown = [digest for digest in requested if digest not in carried]
    if unknown or not requested:
        raise CustomCodeError(
            (f"'{study}' carries no custom code with SHA-256 "
             + ", ".join(unknown) + ". " if unknown else
             "Name the custom code to acknowledge by its SHA-256. ")
            + "The custom code it carries: " + ", ".join(sorted(carried)) + ".",
            repair=review_command(study, program=program),
            facts={"providers": list(carried.values())})
    path = record_path(root)
    with manifest_files.transaction(str(path), workspace_root=str(root)):
        existing = records(root)
        seen = {entry["providerSHA256"] for entry in existing}
        added = []
        stamp = now or _now()
        who = account or _account()
        for digest in requested:
            if digest in seen:
                continue
            added.append({"providerSHA256": digest, "acknowledgedAt": stamp,
                          "acknowledgedBy": who, "client": client,
                          "study": study,
                          "policyNames": carried[digest]["policyNames"]})
        if added:
            _write(path, existing + added)
    return {"acknowledged": added,
            "alreadyAcknowledged": [d for d in requested if d in seen],
            "recordFile": FILENAME,
            "providers": status(document, root)}


def _write(path: Path, entries: list[dict]) -> None:
    payload = json.dumps({"schemaVersion": SCHEMA_VERSION,
                          "acknowledgements": entries},
                         indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".custom-code-",
        suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def run_refusal(document, root, *, study: str, verb: str | None,
                program: str) -> CustomCodeError | None:
    """The refusal for a step that would execute unacknowledged custom code,
    or ``None``. ``verb=None`` means the step is not known yet, which is
    treated as one that may execute."""
    if verb is not None and verb not in EXECUTING_VERBS:
        return None
    block = notice(document, root, study=study, program=program)
    if block is None or block["acknowledged"]:
        return None
    pending = [row for row in block["providers"] if not row["acknowledged"]]
    named = "; ".join(
        f"SHA-256 {row['sha256']} (policy: {', '.join(row['policyNames']) or 'unnamed'})"
        for row in pending)
    return CustomCodeError(
        f"'{study}' was not sent to run: it contains custom code from its "
        f"author that nobody has acknowledged in this workspace ({named}). "
        f"{NOTICE}",
        repair=(f"Read the code with `{block['reviewCommand']}`; if you trust "
                f"the source, run `{block['acknowledgeCommand']}`, then run "
                "the study again."),
        facts={"customCode": block})
