"""``--help``, rendered from the declarative verb table (WP0 step 11).

Swift twin: ``Sources/ExperimentKit/CLIHelp.swift``. ``docs/CLI-REFERENCE.md``
recorded "Neither CLI has ``--help``. This document is the substitute" — which,
for an agent driving a shipped binary with no checkout to read, is the
substitution the wrong way round. The declarative table step 8 built as DATA
(:data:`steerlab_server.cli_envelope.VERB_SPECS`) makes the manual free, and
generating BOTH ``--help`` and the reference document's marked regions from the
same table is what makes them incapable of disagreeing.

Everything in this module is CONTRACT TEXT: neutral, imperative, and readable
by a caller with no prior context. No institution names, no study names, no
incident history — those belong in the reference document's prose, which the
generation deliberately leaves alone.
"""

from __future__ import annotations

from .cli_envelope import (HELP_FLAG, JSON_FLAG, OUT_FLAG, VERB_SPECS, VerbSpec,
                           spec_for)

#: The binary's name in every synopsis line.
PROGRAM = "steerlab-server"

#: The flags every agent-path verb carries.
SHARED_FLAGS = frozenset({HELP_FLAG, JSON_FLAG, OUT_FLAG})

#: What a value flag's argument is called. Boolean flags have none. A flag with
#: no entry renders ``<value>`` — the honest fallback, never an invented shape.
METAVARS: dict = {
    "--adjudicated-endpoint": "<file>",
    "--agent": "<name-or-path>",
    "--agent-name": "<name>",
    "--cell": "<layer>:<alpha>",
    "--concept": "<name>",
    "--deltas": "<d1,d2>",
    "--dependency": "<spec>",
    "--device": "<device>",
    "--dir": "<path>",
    "--dtype": "<dtype>",
    "--executor": "<local|slurm>",
    "--expect-artifact": "<runDir/name>",
    "--expect-artifact-hash": "<sha256>",
    "--expect-cell": "<layer>:<alpha>",
    "--expect-epoch": "<sha256>",
    "--gres": "<spec>",
    "--job-name": "<name>",
    "--mem": "<size>",
    "--out": "<file>",
    "--parallel": "<n>",
    "--parallel-jobs": "<n>",
    "--partition": "<partition>",
    "--prompts": "<path>",
    "--qualification": "<path>",
    "--reason": "<text>",
    "--resume": "<run-dir>",
    "--resume-from": "<run-dir>",
    "--shard": "<k/K>",
    "--source": "<run-dir>",
    "--sweep-run": "<run-dir>",
    "--target": "<root>",
    "--threshold": "<ratio>",
    "--verb": "<verb>",
    "--walltime": "<hh:mm:ss>",
}

#: One line per flag. Imperative, and about the effect rather than the history.
FLAG_PURPOSES: dict = {
    "--adjudicated-endpoint":
        "Substitute an extraction campaign's verified per-record endpoint "
        "values; requires --source.",
    "--agent": "The promoted agent the policy perturbs.",
    "--agent-name": "Name the minted variant artifact.",
    "--allow-unverified-epoch":
        "Accept a legacy run that carries no experiment-hash stamp.",
    "--cell": "Override the sweep-selected cell, loudly.",
    "--concept":
        "The mirrored pole's concept name (required) — it must differ from "
        "the source's, because the negated direction points at the opposite "
        "pole.",
    "--deltas": "Perturbation deltas around the anchor cell (default 0.2).",
    "--dependency":
        "Hold the submission behind a Slurm dependency, e.g. afterok:12345.",
    "--device": "Torch device the verb loads the model onto.",
    "--dir": "Directory the template reads instead of its default.",
    "--dry-run": "Print what would be submitted and submit nothing.",
    "--dtype": "Torch dtype the model is loaded in.",
    "--executor": "Where the job runs.",
    "--expect-artifact": "Refuse unless the sweep produced this artifact.",
    "--expect-artifact-hash": "Refuse unless the artifact hashes to this.",
    "--expect-cell": "Refuse unless the selected cell is this one.",
    "--expect-epoch": "Refuse unless the manifest hash is this one.",
    "--force": "Submit despite a failing preflight verdict, recorded on the job.",
    "--gres": "Scheduler GPU resource request.",
    "--help": "Print this surface and run nothing.",
    "--job-name": "Scheduler job name.",
    "--json": "Print exactly one machine-readable envelope on stdout.",
    "--mem": "Scheduler memory request.",
    "--no-control": "Omit the control condition.",
    "--no-evidence": "Skip packaging an evidence bundle for the job.",
    "--out": "Write the same document to this file.",
    "--output-name":
        "File name for the new artifact inside its run directory (default: "
        "the mirrored concept name).",
    "--parallel": "Shard a Slurm run across N GPU jobs.",
    "--parallel-jobs": "Cap how many shard jobs are submitted at once.",
    "--partition": "Scheduler partition.",
    "--prompts": "Override the pinned task-prompt file (pin-checked when frozen).",
    "--qualification": "Qualification evidence recorded on the promotion.",
    "--reason": "Why the manual cell was chosen; recorded on the certificate.",
    "--resume": "Continue a checkpointed run in this directory.",
    "--skip-model-fixtures":
        "Skip the checks that need a tokenizer from the local model cache.",
    "--resume-from": "Continue judging from this run directory.",
    "--shard":
        "Generate only shard k of K of the run's records; resume a shard "
        "partial with the same k/K.",
    "--source": "The source run directory; absent, the newest completed run.",
    "--sweep-run": "The sweep run whose recommendation is read.",
    "--target": "Remote root the job executes against.",
    "--threshold": "Minimum cosine below which the comparison refuses.",
    "--verb": "The experiment verb the job runs (defaults to run).",
    "--walltime": "Scheduler wall-time request.",
}

#: The state vocabulary, in one line, on every page — an agent reading a help
#: page is about to read an exit code. Swift twin:
#: ``ExperimentCLIHelp.exitCodeLine``.
EXIT_CODE_LINE = (
    "exit codes: 0 ok · 64 malformed invocation · 65 refused · 66 not found "
    "· 70 failed  (--json: the envelope's `state` is authoritative)")


def metavar(flag: str) -> str:
    return METAVARS.get(flag, "<value>")


def flag_purpose(flag: str) -> str:
    return FLAG_PURPOSES.get(flag, "")


def takes_value(spec: VerbSpec, flag: str) -> bool:
    if flag in spec.value_flags:
        return True
    if flag in spec.boolean_flags:
        return False
    return flag == OUT_FLAG


def spelling(spec: VerbSpec, flag: str) -> str:
    return f"{flag} {metavar(flag)}" if takes_value(spec, flag) else flag


def synopsis(spec: VerbSpec, *, include_program: bool = True,
             include_shared_flags: bool = True) -> str:
    """``steerlab-server experiment run <name> [--dtype <dtype>] …``"""
    parts = [PROGRAM] if include_program else []
    parts.append(spec.label)
    if spec.positional:
        parts.append(spec.positional)
    for flag in spec.declared_flags:
        # The reference document lists the three shared agent flags once per
        # region rather than on every line; `--help` prints them, because a
        # caller reading one page has no region note.
        if not include_shared_flags and flag in SHARED_FLAGS:
            continue
        text = spelling(spec, flag)
        parts.append(text if flag in spec.required_flags else f"[{text}]")
    return " ".join(parts)


def verb_text(spec: VerbSpec) -> str:
    """One verb's ``--help`` page."""
    lines = ["usage: " + synopsis(spec)]
    if spec.purpose:
        lines += ["", spec.purpose]
    flags = spec.declared_flags
    if flags:
        lines += ["", "flags:"]
        spellings = [spelling(spec, flag) for flag in flags]
        width = min(max(len(text) for text in spellings), 34)
        for text, flag in zip(spellings, flags):
            purpose = flag_purpose(flag)
            if flag in spec.required_flags:
                purpose = (purpose + "  (required)") if purpose else "Required."
            padding = " " * max(2, width + 2 - len(text))
            lines.append("  " + text + (padding + purpose if purpose else ""))
    lines += ["", EXIT_CODE_LINE]
    return "\n".join(lines) + "\n"


def family_text(family: str) -> str:
    """One family's ``--help`` page: the verbs of that family that speak the
    agent contract.

    The families carry more verbs than these — ``experiment attach-artifact``,
    ``jobs reconcile``, and the rest run exactly as they always did. Saying so
    is the honest page: a caller must not read this list as the family's whole
    surface.
    """
    specs = [spec for spec in VERB_SPECS if spec.family == family]
    if not specs:
        return (f"usage: {PROGRAM} <family> <verb> …\n  agent-path families: "
                + " | ".join(sorted({spec.family for spec in VERB_SPECS}))
                + "\n")
    lines = [f"usage: {PROGRAM} {family} <verb> …", ""]
    width = min(max(len(spec.verb) for spec in specs), 22)
    for spec in specs:
        padding = " " * max(2, width + 2 - len(spec.verb))
        lines.append("  " + spec.verb + padding + spec.purpose)
    lines += [
        "",
        f"{PROGRAM} {family} <verb> --help prints one verb's arguments.",
    ]
    other = _verbs_outside_the_envelope(family, {spec.verb for spec in specs})
    if other:
        lines.append(
            f"Other {family} verbs run unchanged and answer no envelope: "
            + " | ".join(other) + ".")
    else:
        lines.append(
            f"Other {family} verbs exist and run unchanged; these are the "
            "ones that answer in the agent envelope.")
    lines.append(EXIT_CODE_LINE)
    return "\n".join(lines) + "\n"


def _verbs_outside_the_envelope(family: str, declared: set) -> list:
    """The family's dispatched verbs that do NOT answer in the agent envelope.

    Read from the SAME list the family's refusal roster prints
    (``cli.EXPERIMENT_VERBS``), so the two surfaces cannot name different
    sets. Gate-5 dry run #2 (P3): ``experiment --help`` listed ten verbs, the
    bare-``experiment`` refusal listed sixteen, and nothing related the two —
    a caller reading only the help page could not tell that ``rescore-style``
    or ``pipeline`` exist, and an unlisted verb is indistinguishable from an
    absent one. ``EXPERIMENT_VERBS`` stays the superset the dispatch is
    asserted against (``test_the_printed_experiment_verb_list_is_complete``);
    this page now spends it rather than paraphrasing it.

    Imported lazily: ``cli`` imports this module inside a function, and a
    module-level import back would close the cycle.
    """
    if family != "experiment":
        return []
    from .cli import EXPERIMENT_VERBS
    return [verb for verb in EXPERIMENT_VERBS if verb not in declared]


def verb_payload(spec: VerbSpec) -> dict:
    """The same page as data, for ``--json``: a machine caller should not have
    to parse the columns a human reads."""
    flags = []
    for flag in spec.declared_flags:
        entry = {"flag": flag, "takesValue": takes_value(spec, flag),
                 "purpose": flag_purpose(flag)}
        if takes_value(spec, flag):
            entry["value"] = metavar(flag)
        flags.append(entry)
    return {"verb": spec.label, "purpose": spec.purpose,
            "positional": spec.positional, "synopsis": synopsis(spec),
            "flags": flags}


def family_payload(family: str) -> dict:
    return {"family": family,
            "verbs": [{"verb": spec.label, "purpose": spec.purpose,
                       "synopsis": synopsis(spec)}
                      for spec in VERB_SPECS if spec.family == family]}


def text_for(family: str, verb: str | None) -> str:
    """The page one invocation asked for."""
    spec = spec_for(family, verb)
    return verb_text(spec) if spec else family_text(family)


def payload_for(family: str, verb: str | None) -> dict:
    spec = spec_for(family, verb)
    return verb_payload(spec) if spec else family_payload(family)
