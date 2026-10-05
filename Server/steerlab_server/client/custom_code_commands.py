"""The client's custom-code notice and acknowledgement verb.

The owner is :mod:`steerlab_server.experiment.custom_code`; this module only
attaches its ``customCode`` block to the surfaces that bring a study into a
workspace (``pack apply``, ``bundle import``, ``experiment attach-agent``) and
adapts ``experiment acknowledge-custom-code`` to the command line.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..cli_envelope import CLIResult, VerbSpec

PROGRAM = "steerlab"
CLIENT = "steerlab"

VERB_SPECS = (
    VerbSpec("experiment", "acknowledge-custom-code", positional="<name>",
             purpose="Show the custom code (intervention-policy expert "
                     "providers) a study carries; with --sha256, record that "
                     "you trust it, so the study can run. The code runs with "
                     "your permissions; it is not sandboxed.",
             value_flags=frozenset({"--sha256"})),
)
EXPERIMENT_VERBS = frozenset(s.verb for s in VERB_SPECS if s.family == "experiment")


def attach_notice(payload: dict, document, root, *, study: str) -> dict:
    """Add the ``customCode`` block to ``payload`` when ``document`` carries
    custom code, and print the notice in human mode (stdout carries nothing
    else under ``--json``, where the block is the notice)."""
    from ..experiment import custom_code
    block = custom_code.notice(document, root, study=study, program=PROGRAM,
                               after_write=True)
    if block is not None:
        payload["customCode"] = block
        for line in custom_code.notice_lines(block):
            print(line)
    return payload


def imported_studies(result: dict) -> list[tuple[str, dict, str]]:
    """``(study, document, root)`` for each study manifest a bundle import
    landed, read from the files it extracted."""
    root = result.get("targetRoot")
    found = []
    for relative in result.get("extracted") or []:
        parts = Path(relative).parts
        if len(parts) != 3 or parts[0] != "experiments" or parts[2] != "experiment.json":
            continue
        try:
            with open(os.path.join(root, relative), encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(document, dict):
            found.append((parts[1], document, root))
    return found


def attach_bundle_notice(result: dict) -> dict:
    """The ``customCode`` block for a bundle import: one entry per imported
    study that carries custom code."""
    from ..experiment import custom_code
    blocks = {}
    for study, document, root in imported_studies(result):
        block = custom_code.notice(document, root, study=study, program=PROGRAM,
                                   after_write=True)
        if block is not None:
            blocks[study] = block
            for line in custom_code.notice_lines(block):
                print(line)
    if len(blocks) == 1:
        result["customCode"] = next(iter(blocks.values()))
    elif blocks:
        result["customCode"] = {"studies": blocks}
    return result


def run(invocation) -> CLIResult:
    from ..client_cli import ClientRefusal
    from ..experiment import custom_code, experiment_store as store
    from . import authoring_files as files

    spec = invocation.spec
    args = invocation.positionals
    if len(args) != 1:
        raise ClientRefusal(code="usage", reason=f"{spec.label} needs {spec.positional}.",
                            repair_action=f"{PROGRAM} {spec.label} --help")
    name = args[0]
    root = files.root_path()
    files.study_path(root, name)
    document = dict(store.load_raw(name, str(root)))
    hashes = invocation.all("--sha256")
    if not hashes:
        rows = custom_code.status(document, root)
        sources = custom_code.provider_sources(document, root)
        payload = {"study": name, "providers": [{**row, "sourceText": sources.get(row["sha256"])}
                                                for row in rows],
                   "recordFile": custom_code.FILENAME}
        pending = [row["sha256"] for row in rows if not row["acknowledged"]]
        if not rows:
            line = f"'{name}' carries no custom code; nothing to acknowledge."
        elif pending:
            payload["notice"] = custom_code.NOTICE
            payload["acknowledgeCommand"] = custom_code.acknowledge_command(
                name, pending, program=PROGRAM)
            line = (f"'{name}' carries {len(rows)} piece(s) of custom code; "
                    f"{len(pending)} not yet acknowledged.")
        else:
            line = f"'{name}' carries {len(rows)} piece(s) of custom code, all acknowledged."
        print(line)
        for row in payload["providers"]:
            state = (f"acknowledged {row['acknowledgedAt']} by {row['acknowledgedBy']}"
                     if row["acknowledged"] else "not acknowledged")
            print(f"--- custom code SHA-256 {row['sha256']} "
                  f"(policy: {', '.join(row['policyNames']) or 'unnamed'}; {state})")
            print(row["sourceText"] or "")
        if pending:
            print(custom_code.NOTICE)
            print(f"If you trust the source: {payload['acknowledgeCommand']}")
        return CLIResult(message=line, payload=payload)
    try:
        result = custom_code.acknowledge(root, document, hashes, study=name,
                                         client=CLIENT, program=PROGRAM)
    except custom_code.CustomCodeError as exc:
        raise ClientRefusal(code=custom_code.GATE, gate=custom_code.GATE,
                            reason=str(exc), repair_action=exc.repair_action,
                            state="refused", payload=exc.facts) from None
    added = len(result["acknowledged"])
    line = (f"recorded {added} acknowledgement(s) for '{name}' in "
            f"{custom_code.FILENAME}" if added else
            f"every named piece of custom code in '{name}' was already acknowledged")
    print(line)
    return CLIResult(message=line, changed=bool(added),
                     payload={"study": name, **result})
