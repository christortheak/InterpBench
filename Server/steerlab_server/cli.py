"""Headless runner (parallel to Swift ``steerlab-cli``).

Same verbs, so a cluster operator drives the engine exactly as on the Mac:

    steerlab-server --config path.json          # smoke-test | toy-concept
    steerlab-server serve [--port N] [--root D] # FastAPI service (token auth
                                                # by default; --dev-open-loopback
                                                # opts into the open local tier)
    steerlab-server experiment <verb> <name> …  # extract/validate/sweep/run/
                                                # evaluate/analyze/verify/list

``--root <dir>`` is accepted by EVERY verb (it exports ``STEERLAB_ROOT``
before the app or any path helper is constructed), so a cluster operator can
point one binary at any workspace tree without touching the environment.

Anything needed for the paper must work from here, not just the UI — the GUI/web
client call into this layer, never the reverse.
"""

from __future__ import annotations

import json
import os
import sys
import time


class RootFlagError(Exception):
    """A ``--root`` flag that cannot possibly point at an artifact tree."""


_ROOT_FROM_FLAG = False


def _apply_root_flag(args: list[str]) -> list[str]:
    """Pop a global ``--root <path>`` (valid on every verb) and export it as
    ``STEERLAB_ROOT`` **before** anything imports the app or touches
    ``experiment.paths`` — the whole artifact tree (prompts/, experiments/,
    runs/) hangs off that env var. The directory must exist: a typo'd root
    would otherwise silently point every path at nothing. Returns the
    remaining args; raises ``RootFlagError`` on a missing/non-directory path.
    """
    global _ROOT_FROM_FLAG
    out = list(args)
    while "--root" in out:
        i = out.index("--root")
        if i + 1 >= len(out):
            raise RootFlagError("--root requires a directory path")
        raw = out[i + 1]
        root = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
        if not os.path.isdir(root):
            raise RootFlagError(f"--root {raw!r} is not a directory")
        os.environ["STEERLAB_ROOT"] = root
        _ROOT_FROM_FLAG = True
        out = out[:i] + out[i + 2:]
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _apply_root_flag(args)
    except RootFlagError as exc:
        sys.stderr.write(f"steerlab-server: {exc}\n")
        return 64

    if args and args[0] == "serve":
        return _serve(args[1:])
    # The AGENT-PATH families (WP0 step 8): the fourteen verbs of the audit's
    # §2.1 table answer in the shared `SteerLabCLIEnvelope` under `--json`,
    # with strict flag parsing and the cross-engine state/exit vocabulary.
    # Every other verb in these families passes through untouched.
    if args and args[0] in _AGENT_FAMILY_ORDER:
        return _run_agent_family(args[0], args[1:])
    if args and args[0] == "docs":
        return _docs(args[1:])
    if args and args[0] == "profile":
        return _profile(args[1:])
    if args and args[0] == "bundle":
        return _bundle(args[1:])
    if args and args[0] == "finetune":
        return _finetune(args[1:])
    if args and args[0] == "housekeeping":
        return _housekeeping(args[1:])
    if args and args[0] == "battery":
        return _battery(args[1:])
    if args and args[0] == "panel":
        return _panel(args[1:])
    if args and args[0] == "jlens":
        return _jlens(args[1:])
    if args and args[0] == "optvec":
        return _optvec(args[1:])
    if args and args[0] == "sae":
        return _sae(args[1:])
    if args and args[0] == "gemmascope":
        return _gemmascope(args[1:])
    if len(args) >= 2 and args[0] == "--config":
        return _run_config(args[1])

    # `steerlab-server --help` with no family: the same page the usage error
    # prints, on stdout, exit 0 (WP0 step 11). Asking what the families are was
    # never an error.
    if args and args[0] in ("--help", "-h"):
        sys.stdout.write(_usage_text())
        return 0

    sys.stderr.write(_usage_text())
    return 64


def _usage_text() -> str:
    return (
        "usage: steerlab-server --config <path.json> "
        "| serve [--port N] [--root DIR] [--host H] [--dev-open-loopback] "
        "| experiment <verb> <name> … | profile show|validate "
        "| jobs list|reconcile <dir> | bundle run|evidence|inspect|import|create|submit … "
        "| study submit <experiment> … "
        "| finetune execute <job-dir> [--record rec.json] | finetune plan|train <config.json> "
        "| finetune submit <finetune-request.json> [--plan-only] "
        "[--confirm-plan HASH] [--dry-run] [--force] "
        "| housekeeping status [--refresh] | housekeeping maintenance set --file <json> "
        "| vectors compare <a.safetensors> <b.safetensors> [--json OUT] [--threshold T] "
        "| vectors backfill-norms <runDir/name> [--corpus <path>] "
        "[--output-name N] [--redenominate] "
        "| site qualify [--json OUT] [--skip-model-fixtures] "
        "| site node-scratch-wrapper [--metadata-root DIR] [--print] "
        "| data check optvec [--dir DIR] [--json] "
        "| data check lora [<package-manifest-or-dir>] [--json] "
        "| battery lint <path> [--json] "
        "| battery generation-prompt [--count N] [--avoid <domain text>] "
        "[--out <file>] "
        "| panel list | panel check <path-or-name> "
        "| jlens list|supported|acquire|import|inspect|support|qualify|g0|report "
        "[<model-or-lens-id>] "
        "| optvec train|eval|interpret|family|jspace --config <path.json> "
        "| optvec geometry <artifact> <artifact> … "
        "| optvec campaign materialize|submit|status … "
        "| sae candidates check <path> [--json] "
        "| sae candidates pin <experiment> <path> "
        "| sae family-report --config <path.json> "
        "| sae qualification record --inputs <json> --artifact <runDir/name> "
        "| sae qualification show <path> [--json]\n"
        "| gemmascope import-id --model M --release R --sae-id S --feature N "
        "--label L --residual-norm-artifact <runDir/name>\n"
        "| docs cli-reference [--check|--write] [--path <file>]\n"
        "(--root DIR sets the artifact root — STEERLAB_ROOT — for any verb)\n"
        "(`steerlab-server <family> --help` lists a family's agent-path verbs; "
        "`<family> <verb> --help` prints one verb's arguments)\n")


#: The verb families whose declared verbs are on the agent path (audit §2.1).
#: Tuple rather than a set so the dispatch order in ``main`` is readable.
_AGENT_FAMILY_ORDER = ("experiment", "jobs", "study", "vectors", "data", "site")

#: Every ``experiment`` verb this engine dispatches, in lifecycle order.
#:
#: ONE list, printed at both refusal sites. The audit's drift finding D9 was
#: those two hand-written lists disagreeing with each other and with the
#: dispatch: the first omitted ``pipeline`` and ``judge-worker``, the second
#: omitted those two plus ``preflight-endpoints``. An unlisted verb is
#: indistinguishable from an absent one, so the list is now derived in one
#: place and asserted against the dispatch by
#: ``test_cli_reference.py::test_the_printed_experiment_verb_list_is_complete``.
EXPERIMENT_VERBS = (
    "list", "verify", "attach-artifact", "extract", "validate", "sweep", "run",
    "pipeline", "evaluate", "judge-worker", "complete-judgment", "analyze",
    "rescore-style", "promote", "confirm", "preflight-endpoints",
)

_EXPERIMENT_VERB_LINE = "verbs: " + " | ".join(EXPERIMENT_VERBS) + "\n"


#: The argument surface of every ``experiment`` verb that does NOT answer in
#: the agent envelope, as ``{flag: takes-a-value}`` (open-issues §16 repair 3).
#:
#: The envelope's declared verbs (``cli_envelope.VERB_SPECS``) are parsed
#: strictly in BOTH modes already — an undeclared flag there is 64 before the
#: verb does any work. The pass-through verbs had the opposite behaviour: the
#: ``_flag`` scanner reads the flags it knows and says nothing about the rest,
#: so a correct-looking command line was quietly half-executed. That is the
#: trap class §16 is about (`experiment pipeline … --shard 0/2` ran the WHOLE
#: matrix, unsharded), and a leftover ``--…`` token now refuses in the same
#: shape the envelope uses.
#:
#: Kept as DATA beside the verb list it completes: ``test_shard_cli.py``
#: asserts this table covers exactly the verbs ``VERB_SPECS`` does not, so a
#: new pass-through verb cannot join the family without declaring its surface.
_EXPERIMENT_PASSTHROUGH_FLAGS: dict = {
    "attach-artifact": {"--artifact": True, "--source-concept": True,
                        "--eval-run": True},
    "pipeline": {"--resume": True, "--dtype": True, "--device": True,
                 # Accepted so it can be REFUSED with a reason (see the
                 # pipeline arm) rather than silently dropped.
                 "--shard": True},
    "judge-worker": {"--awaiting-run": True, "--model": True, "--out": True,
                     "--record": True, "--revision": True, "--dtype": True,
                     "--device": True},
    "complete-judgment": {"--awaiting-run": True, "--judgments": True},
    "rescore-style": {"--source": True, "--allow-unverified-epoch": False},
    "preflight-endpoints": {"--baseline-run": True, "--out": True,
                            "--json": False, "--band": True,
                            "--min-cell-items": True, "--min-items": True},
}


def _unknown_experiment_flag(verb: str, rest: list[str]) -> str | None:
    """The first flag ``experiment <verb>`` does not accept, or ``None``.

    Scoped to the pass-through verbs: a verb the envelope declares was already
    parsed strictly before this dispatch ran, and an unrecognised verb answers
    with the roster rather than with a flag complaint.
    """
    known = _EXPERIMENT_PASSTHROUGH_FLAGS.get(verb)
    if known is None:
        return None
    index = 0
    while index < len(rest):
        token = rest[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token not in known:
            return token
        # A value flag's argument is consumed, so a value that itself looks
        # like a flag (`--reason --3`) is never mistaken for one.
        index += 2 if known[token] else 1
    return None


def _mac_authority_refusal(label: str, repair: str, *, note: str = "",
                           exit_code: int = 64):
    """A verb that lives on the Mac and will not be added here.

    The table is :data:`cli_envelope.MAC_AUTHORITY_VERBS` — declared there
    because the parser has to know it too (it is the one unrecognised-verb
    class that must still answer a ``--json`` document).

    Gate-5 dry run #2 (P3) measured what typing one of these used to cost.
    With a positional it fell through to the verb ROSTER, which says the verb
    is unknown but not that it EXISTS somewhere else — so §7's standing promise
    ("when a verb is unavailable here the refusal says so; do not emulate it")
    was unkept, and an agent's only reading was "I typed something wrong".
    Without a positional it never even reached the roster: it got
    ``usage: experiment attach <name>``, which asserts the verb is real here.

    ``state: refused`` (65 in ``--json``) rather than ``blocked``: the request
    is well formed, and a policy declined it. The human exit code is unchanged
    from what these invocations always returned, per the audit's compatibility
    posture — the DOCUMENT is where the distinction lives.
    """
    from .cli_envelope import CLIResult, MAC_AUTHORITY_CODE

    reason = (
        f"'{label}' is not a verb of this engine and will not become one — "
        "authoring (create/attach, the pin-*/declare-*/set-* verbs, freeze, "
        "duplicate) is Mac-authority by policy: the Mac workspace is the "
        "source of truth and this engine is a runner and a cache")
    if note:
        reason += f". {note}"
    sys.stderr.write(f"{reason}\n  {repair}\n")
    return CLIResult(state="refused", exit_code=exit_code,
                     code=MAC_AUTHORITY_CODE, message=reason,
                     repair_action=repair)

#: True while an agent-path verb is running under ``--json``.
#:
#: Read by the handful of verbs that CATCH their own refusal to print an
#: ``ERROR:`` line and return 1 (``promote``, ``confirm``). Human mode must
#: keep that line and that code byte for byte; JSON mode needs the exception
#: itself so ``_exception_envelope`` can read its gate id, so those sites
#: re-raise instead. One flag rather than threading a mode through every
#: signature: the alternative is a parameter on functions the HTTP layer also
#: calls.
_JSON_MODE_ACTIVE = False


_DOCS_USAGE = ("usage: steerlab-server docs cli-reference [--check | --write] "
               "[--path <file>]\n")


def _docs(args: list[str]) -> int:
    """Regenerate — or check — this engine's marked regions in
    ``docs/CLI-REFERENCE.md`` (audit §5.3).

    Generation is deliberately NOT a build step: the document must be readable
    on GitHub with no toolchain, so the generated text is COMMITTED and a test
    compares. ``--check`` is the same comparison from a command line and is the
    DEFAULT — a verb that rewrote a committed document because a flag was
    forgotten would be the wrong default by a wide margin.

    Exit codes: 0 = in sync (or rewritten); 65 = drift; 64 = usage;
    66 = no document to read.
    """
    from . import cli_reference
    if not args or args[0] != "cli-reference":
        sys.stderr.write(_DOCS_USAGE)
        return 64
    path = _flag(args, "--path") or cli_reference.committed_path()
    if not path:
        sys.stderr.write(
            "steerlab-server docs: no docs/CLI-REFERENCE.md next to this "
            "install — pass --path <file>\n")
        return 66
    try:
        with open(path, encoding="utf-8") as handle:
            document = handle.read()
        drifted = cli_reference.drift(document)
        if "--write" in args:
            rewritten = cli_reference.rewrite(document)
            if rewritten == document:
                print(f"{path}: already in sync")
                return 0
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(rewritten)
            print(f"rewrote {len(drifted)} region(s) in {path}: "
                  + ", ".join(drifted))
            return 0
        if drifted:
            for region in drifted:
                sys.stderr.write(f"DRIFT: region {region} is out of date\n")
            sys.stderr.write(
                "repair: steerlab-server docs cli-reference --write, then "
                "commit the document\n")
            return 65
        print(f"{path}: {len(cli_reference.REGIONS)} generated region(s) match "
              "the verb table")
        return 0
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"steerlab-server docs cli-reference: {exc}\n")
        return 66 if isinstance(exc, OSError) else 65


def _run_agent_family(family: str, args: list[str]) -> int:
    """Run one agent-path family invocation: parse strictly, run the verb, and
    answer in the shared envelope when ``--json`` was asked for.

    The compatibility posture, stated once because every rule below follows
    from it:

    * **Human mode is byte-stable.** The verb prints exactly what it always
      printed and returns exactly the code it always returned — including an
      uncaught exception, which still propagates with its traceback. An agent
      asked for a document; a human did not, and changing what a human's
      ``set -e`` wrapper sees is not this step's business.
    * **``--json`` mode is a NEW surface** and is born speaking the state
      vocabulary: a gate refusal is 65, a missing file 66, an operational
      failure 70, a malformed invocation 64. Exactly one document on stdout;
      every diagnostic — including everything ``tasks.py`` prints — on stderr.
    * **An undeclared flag is 64 in BOTH modes**, before the verb does any
      work. A malformed invocation was never a refusal, and a refusal after
      the first concept is pinned is not much better than no refusal at all.
    """
    from . import cli_envelope as envelope

    handlers = {"experiment": _experiment, "data": _data, "vectors": _vectors,
                "jobs": _jobs, "study": _study, "site": _site}
    handler = handlers[family]

    try:
        invocation = envelope.parse(family, args)
    except envelope.UsageError as exc:
        # `--json` is honoured even when parsing itself fails: an agent that
        # asked for machine output must not get prose back on the one path it
        # cannot parse (audit §2.2).
        sys.stderr.write(
            f"steerlab-server {family}: {exc.reason}\n  {exc.repair_action}\n")
        document = envelope.refusal(
            exc.verb, code=envelope.UsageError.code, reason=exc.reason,
            repair_action=exc.repair_action, state="blocked")
        envelope.emit(document, json_mode=(envelope.JSON_FLAG in args),
                      out_path=None)
        return 64

    # `--help` runs NOTHING and exits 0 in both modes (WP0 step 11). The page
    # goes to stdout in human mode; in `--json` mode the page travels as the
    # envelope's `result`, so the one-document rule holds and a machine caller
    # never parses the columns a human reads.
    if invocation.help:
        from . import cli_help
        page = cli_help.text_for(family, invocation.verb)
        spec = envelope.spec_for(family, invocation.verb)
        if invocation.json:
            sys.stderr.write(page)
            document = envelope.success(
                spec.label if spec else family,
                spec.purpose if spec else
                f"{family} verbs that answer in the agent envelope",
                result=cli_help.payload_for(family, invocation.verb))
            envelope.emit(document, json_mode=True,
                          out_path=invocation.out_path)
            return 0
        sys.stdout.write(page)
        return 0

    for notice in invocation.deprecations:
        sys.stderr.write(notice + "\n")

    global _JSON_MODE_ACTIVE
    json_mode = invocation.json
    document_stream = sys.stdout
    if json_mode:
        sys.stdout = sys.stderr
        _JSON_MODE_ACTIVE = True
    try:
        outcome = handler(invocation.args)
    except Exception as exc:     # noqa: BLE001 — re-raised in human mode
        sys.stdout = document_stream
        if not json_mode:
            raise
        sys.stderr.write(f"steerlab-server {family}: {exc}\n")
        document = _exception_envelope(invocation, exc)
        envelope.emit(document, json_mode=True, out_path=invocation.out_path,
                      stream=document_stream)
        return document.exit_code
    finally:
        sys.stdout = document_stream
        _JSON_MODE_ACTIVE = False

    document = _outcome_envelope(invocation, outcome)
    human_code = (outcome.exit_code
                  if isinstance(outcome, envelope.CLIResult) else int(outcome))
    envelope.emit(document, json_mode=json_mode, out_path=invocation.out_path,
                  stream=document_stream)
    return document.exit_code if json_mode else human_code


def _outcome_envelope(invocation, outcome):
    """The envelope for a verb that returned rather than raised."""
    from . import cli_envelope as envelope

    label = invocation.label
    if isinstance(outcome, envelope.CLIResult):
        if outcome.state in ("ready", "okWithAdvisories"):
            return envelope.success(
                label, outcome.message, changed=outcome.changed,
                advisories=outcome.advisories, result=outcome.payload or None,
                next_action_=outcome.next_action)
        return envelope.refusal(
            label, code=outcome.code or "refused", gate=outcome.gate,
            reason=outcome.message, repair_action=outcome.repair_action,
            state=outcome.state, result=outcome.payload or None,
            next_action_=outcome.next_action)
    # A verb this step has not converted still answers in the vocabulary
    # rather than in an integer an agent would have to guess at.
    code = int(outcome)
    state = envelope.state_for_legacy_exit(code)
    if state in ("ready", "planned", "running", "okWithAdvisories", "pending"):
        return envelope.success(label, f"{label} completed", state=state)
    return envelope.failure(
        label, code="usage" if code == 64 else "verbFailed",
        reason=f"{label} exited {code} — see the diagnostics on stderr",
        repair_action=f"steerlab-server {invocation.family}  (verb list)",
        state=state)


def _implicit_case_family_advisories(name: str, root: str | None) -> list:
    """The envelope half of the ``caseFamily`` deprecation (2026-08-18): the
    same sentence ``tasks`` logs, carried as the closed-vocabulary code
    ``deprecatedImplicitSelection`` so an agent can switch on it instead of
    grepping a log.

    Tolerant by design, like every other advisory helper here: a manifest that
    cannot be read is a problem the verb itself has already reported properly,
    and an advisory must never be the thing that fails a verb. Advisories never
    change the exit code.
    """
    from .cli_envelope import advisory
    from .experiment.manifest import (IMPLICIT_CASE_FAMILY_ADVISORY, Manifest,
                                      implicit_case_family_endpoint)
    try:
        manifest = Manifest.load(name, root)
    except Exception:  # noqa: BLE001
        return []
    if not implicit_case_family_endpoint(manifest):
        return []
    return [advisory("deprecatedImplicitSelection",
                     IMPLICIT_CASE_FAMILY_ADVISORY)]


def _defaulted_selection_advisory(manifest, root: str | None) -> str | None:
    """The sweep's defaulted-criterion advisory, computed from the manifest and
    its pinned task set alone (punch list #1, P3).

    Tolerant by design: a study whose prompts cannot be loaded is a problem the
    sweep itself will report properly — an advisory must never be the thing
    that fails a verb.
    """
    from .experiment import sweep_selection, tasks
    spec = (manifest.raw.get("sweep") or {}).get("selection")
    try:
        prompts = tasks._load_prompts(manifest, None, root)
    except Exception:   # noqa: BLE001 - see the docstring
        return None
    choice = sum(1 for prompt in prompts if prompt.get("options"))
    return sweep_selection.defaulted_selection_advisory(
        spec, choice, len(prompts))


def _exception_envelope(invocation, exc: BaseException):
    """The envelope for a raised refusal or failure.

    Three carriers, in the order the audit's §2.4 census names them: a FREEZE
    gate (``ExperimentStoreError.gate``, the closed seven), a LIFECYCLE gate
    (the closed eighteen — a lifecycle refusal's ``code`` IS its gate id), and
    everything else. A missing file is ``notFound`` (66) rather than a failure:
    it is the commonest agent mistake and was indistinguishable from a crash.
    """
    from . import cli_envelope as envelope
    from .experiment import lifecycle_gates

    label = invocation.label
    reason = str(exc)
    freeze_gate = getattr(exc, "gate", None)
    freeze_gates = list(getattr(exc, "gates", ()) or ())
    lifecycle_gate = lifecycle_gates.gate_of(exc)

    if lifecycle_gate:
        return envelope.refusal(
            label, code=lifecycle_gate, gate=lifecycle_gate, reason=reason,
            repair_action=lifecycle_gates.repair_of(exc) or _UNTYPED_REPAIR)
    if freeze_gate:
        from .experiment.experiment_store import FORCED_GATE_IDS
        if freeze_gate in FORCED_GATE_IDS:
            return envelope.refusal(
                label, code="freezeGateFailed", gate=freeze_gate,
                gates=freeze_gates, reason=reason,
                repair_action=lifecycle_gates.repair_of(exc) or (
                    f"satisfy the '{freeze_gate}' gate  (this engine has no "
                    "freeze verb — freezing, forced or not, is Mac-authority)"))
    if isinstance(exc, FileNotFoundError):
        missing = getattr(exc, "filename", None)
        # Gate-5 dry run #2 (P3): the commonest instance by far is a MISTYPED
        # EXPERIMENT NAME, and `reason` was Python's own
        # "[Errno 2] No such file or directory: '/…/experiments/typo/
        # experiment.json'" — an absolute path, with the name the caller
        # actually typed nowhere in the document. Say what is missing.
        experiment = _experiment_name_in_missing_path(missing)
        if experiment:
            return envelope.refusal(
                label, code="notFound", state="notFound",
                reason=f"experiment '{experiment}' not found in this workspace",
                repair_action=(
                    "steerlab-server experiment list  (the experiments this "
                    "workspace holds), then re-run with a name from it — or "
                    "author it on the Mac: steerlab-cli experiment create "
                    "<name> --model <id>"))
        return envelope.refusal(
            label, code="notFound", reason=reason, state="notFound",
            repair_action=(f"no file at {missing}" if missing else
                           "check the name against `steerlab-server "
                           "experiment list`"))
    return envelope.failure(
        label, code="verbFailed", reason=reason,
        repair_action=_UNTYPED_REPAIR)


#: What an UNTYPED throw honestly offers. Gate-5 dry run #2 (P3): "read the
#: reason and repair the named input" claimed a repair had been derived and an
#: input named, and on the paths that reached it — a bare ``RuntimeError``,
#: the re-freeze guard — neither was true. Every throw that DOES know its
#: repair carries one now (``LifecycleRefusalMixin``, ``ExperimentStoreError``
#: with ``repair=``) and never reaches this string. Byte-identical to the
#: Swift twin in ``ExperimentCLIRunner``.
_UNTYPED_REPAIR = (
    "this was not a typed refusal — read the reason; if it names a file or a "
    "pin, repair that, otherwise the verb failed operationally and the reason "
    "is all there is")


def _experiment_name_in_missing_path(path: str | None) -> str | None:
    """The experiment a missing-file path is about, when the path is a
    manifest.

    BOTH manifest layouts this engine reads: the nested
    ``…/experiments/<name>/experiment.json`` and the flat legacy
    ``…/experiments/<name>.json`` that ``Manifest.load`` falls back to — the
    fallback is what a MISSING study actually reports, so matching only the
    nested shape would have left the commonest case unimproved.

    Shape-matched at the CLASSIFICATION site rather than threaded down from
    each verb: the loaders are called from everywhere and raise Python's own
    ``FileNotFoundError``, so this is the one place that sees every instance.
    A path that is not a manifest keeps the file-shaped answer — inventing an
    experiment name for a missing rubric would be worse than the raw path.
    """
    if not path:
        return None
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-1] == "experiment.json" \
            and parts[-3] == "experiments" and parts[-2]:
        return parts[-2]
    if len(parts) >= 2 and parts[-2] == "experiments" \
            and parts[-1].endswith(".json"):
        name = parts[-1][:-len(".json")]
        return name or None
    return None


def _print_support(readout: dict, run_dir: str) -> None:
    """Human-readable support readout.

    The energy figure is printed only next to its null, and never as a headline:
    a norm-matched random direction scores comparably in this dictionary, so the
    tokens are the finding and the fraction is context for them.
    """
    vector = readout["vector"]
    print(f"{vector['name']}  ({vector.get('extractionMethod') or '?'} on "
          f"{readout['modelID']})  lens {readout['lensID']} "
          f"[{readout.get('lensFitPrompts')} prompts, "
          f"{readout.get('lensFitCorpus') or 'unknown corpus'}]")
    for report in readout["layers"]:
        margin = report["energyOverNull"]
        print(f"\nlayer {report['layer']}  "
              f"reconstructed {100 * report['energyFraction']:.1f}% "
              f"vs {100 * report['nullEnergyFraction']:.1f}% for a matched-norm "
              f"random direction ({margin:+.1%})"
              + (f"  [cone exhausted at k={report['coneExhaustedAt']}]"
                 if report["coneExhaustedAt"] else ""))
        for rank, row in enumerate(report["support"], start=1):
            print(f"  {rank:3d}. {row['piece']:<24s} {row['share']:6.1%}  "
                  f"(id {row['tokenID']})")
    print(f"\n{run_dir}")


def _int_list(value: str | None) -> list[int] | None:
    """``"20,26,32"`` → ``[20, 26, 32]``; absent stays absent.

    None and the empty list mean different things everywhere they are used
    (absent = "use the default", empty = "none"), so an unsupplied flag must
    not collapse into an empty selection.
    """
    if value is None:
        return None
    return [int(piece) for piece in value.split(",") if piece.strip()]


def _alpha_ladder(value: str | None) -> tuple[float, ...] | None:
    """A dose ladder from ``"0.04,0.08,0.12"`` or the range form ``"LO:HI"``.

    The range form expands to three rungs (low, midpoint, high) because a
    two-point "ladder" cannot distinguish a monotone response from a straight
    line through two arbitrary numbers.
    """
    if value is None:
        return None
    text = value.strip()
    if ":" in text:
        low, high = (float(x) for x in text.split(":", 1))
        return (low, (low + high) / 2.0, high)
    return tuple(float(piece) for piece in text.split(",") if piece.strip())


def _print_jlens_probe(report: dict) -> None:
    """The trajectory, readable at a terminal.

    Prints RANKS rather than top-k membership for the pinned tokens: the top of
    this distribution is tilted toward tokens with large unembedding norms
    (including untrained ones), so rank is the measurement and top-k is the
    illustration. Lower rank = more strongly poised to say it.
    """
    config = report["configuration"]
    print(f"{report['runID']}  lens {report['lens']['lensID']} "
          f"[{report['evidenceTier']}]  {report['claim']}")
    print(f"{report['prompt']['tokenCount']} token(s), layers "
          f"{config['armedLayers']}, {config['projections']} projection(s)")
    pinned = config["pinned"]
    directions = [d["name"] for d in config["directions"]]
    if not pinned and not directions:
        print("\nno pinned tokens or directions — see probe-topk.csv for the "
              "discovery view")
    for layer in config["armedLayers"]:
        rows = [r for r in report["trajectory"] if r["layer"] == layer]
        if not rows:
            continue
        print(f"\nlayer {layer}")
        for token, piece in sorted(pinned.items()):
            ranks = [(r["position"], (r.get("pinnedRanks") or {}).get(token))
                     for r in rows]
            ranks = [(p, v) for p, v in ranks if v is not None]
            if not ranks:
                continue
            best = min(ranks, key=lambda pair: pair[1])
            print(f"  {piece!r:<16} best rank {best[1]} at position {best[0]}"
                  f"  (first {ranks[0][1]} → last {ranks[-1][1]})")
        for name in directions:
            values = [(r["position"],
                       (r.get("directions") or {}).get(name, {}).get(
                           "transportedCosine"))
                      for r in rows]
            values = [(p, v) for p, v in values if v is not None]
            if not values:
                continue
            peak = max(values, key=lambda pair: pair[1])
            print(f"  {name:<16} peak transported cos {peak[1]:+.3f} at "
                  f"position {peak[0]}")
    print(f"\n{report['runDirectory']}")


def _print_jlens_report(report: dict) -> None:
    """The cross-condition roll-up, readable at a terminal.

    Prints the J-lens number and its logit-lens companion together, always:
    the companion is the control that says whether transport did any work, and
    a table that showed one without the other would invite reading the lens as
    having found something the residual already said.
    """
    identity = report["identity"]
    print(f"{report['runDirectory']}  lens {identity.get('lensID')} "
          f"[{identity.get('evidenceTier')}]  "
          f"{identity.get('modelID')}@{(identity.get('modelRevision') or '')[:12]}")
    counts = report["completeness"]
    print(f"{counts['rowsUsed']}/{counts['rowsRead']} trace row(s) complete"
          + (f"; excluded {counts['incompleteByCondition']}"
             if counts["incompleteByCondition"] else ""))
    for name, conditions in sorted(report["watchlist"].items()):
        print(f"\ntoken set '{name}'")
        for condition, block in sorted(conditions.items()):
            score = block["score"]
            print(f"  {condition:<28s} {('—' if score is None else f'{score:+.3f}')}"
                  f"  [{block['convention']}]  "
                  f"{block['scoredObservations']} scored / "
                  f"{block['excludedObservations']} excluded")
    deltas = report["deltas"]
    if not deltas.get("available"):
        print(f"\nno deltas: {deltas.get('reason')}")
        return
    print(f"\ntop-k movers vs '{deltas['baseline']}'")
    for condition, layers in sorted(deltas["topK"].items()):
        for layer, tokens in sorted(layers.items(), key=lambda kv: int(kv[0])):
            movers = sorted(tokens.items(),
                            key=lambda kv: -abs(kv[1]["occupancyDelta"]))[:5]
            for token, cell in movers:
                print(f"  {condition:<24s} L{layer:<3s} id {token:<8s} "
                      f"occupancy {cell['baselineOccupancy']:.2f} → "
                      f"{cell['conditionOccupancy']:.2f} "
                      f"({cell['occupancyDelta']:+.2f}"
                      + (", new" if cell["newInCondition"] else "")
                      + f"; n={cell['conditionCount']})")


def _jlens(args: list[str]) -> int:
    """J-lens reading instruments — server-only, Gemma-only (CLAUDE.md).

    Two operations, deliberately distinct (plan §11.0.1): `acquire` puts the
    published bytes in the HF cache and needs egress; `import` converts them
    once into the workspace lens store and is offline. Everything a study
    depends on must be reachable here, not only through the UI.
    """
    from .jlens import acquire as acquire_mod
    from .jlens import importer, lens_store
    from .jlens.schemas import JLensError

    verb = args[0] if args else None

    try:
        if verb == "supported":
            for model_id in importer.supported_models():
                entry = importer.SUPPORTED[model_id]
                print(f"{model_id}\t{entry['tier']}\t{entry['folder']}/{entry['tensor']}")
            return 0

        if verb == "list":
            records = lens_store.list_lenses()
            if not records:
                print("no imported lenses — steerlab-server jlens import <model>")
                return 0
            for rec in records:
                converted = "converted" if rec.converted else "NOT CONVERTED"
                print(f"{rec.lensID}\t{rec.fit.modelID or '?'}\t"
                      f"layers {rec.sourceLayers[0]}..{rec.sourceLayers[-1]} "
                      f"target {rec.targetLayer}\t{converted}\t"
                      f"{len(rec.qualifications)} qualification(s)")
            return 0

        if verb == "acquire" and len(args) >= 2:
            # STDOUT POLLUTION FIX (WP0 step 8, named by audit §7 row 8): the
            # download's progress lines went to STDOUT, ahead of the JSON
            # document this verb ends with — so `jlens acquire | jq` failed on
            # the one verb whose output a script is most likely to pipe. The
            # lines now go to stderr, exactly as `jlens support`, `qualify`,
            # and `g0` already send theirs. Nothing else about the verb
            # changes: same document, same exit code, same log text.
            snapshot = acquire_mod.acquire(
                args[1], log=lambda m: print(m, file=sys.stderr, flush=True))
            print(json.dumps({"ok": True, "snapshot": snapshot}))
            return 0

        if verb == "import" and len(args) >= 2:
            record = importer.import_lens(args[1])
            print(json.dumps({
                "ok": True, "lensID": record.lensID,
                "sourceLayers": [record.sourceLayers[0], record.sourceLayers[-1]],
                "targetLayer": record.targetLayer, "dModel": record.dModel,
                "converted": record.converted.path if record.converted else None,
                "convertedDtype": record.converted.dtype if record.converted else None,
            }))
            return 0

        if verb == "inspect" and len(args) >= 2:
            print(json.dumps(lens_store.resolve(args[1]).to_dict(),
                             indent=2, sort_keys=True))
            return 0

        if verb == "support" and len(args) >= 3:
            from .jlens import decompose
            layers = _flag(args, "--layers")
            readout = decompose.decompose(
                lens_id=args[1],
                vector_directory=os.path.dirname(args[2]),
                vector_name=os.path.basename(args[2]),
                layers=[int(x) for x in layers.split(",")] if layers else None,
                budget=int(_flag(args, "--k") or decompose.DEFAULT_BUDGET),
                device=_flag(args, "--device"),
                progress=lambda m: print(m, file=sys.stderr, flush=True))
            run_dir = decompose.write_readout(readout)
            if "--json" in args:
                print(json.dumps(readout, indent=2, sort_keys=True))
                return 0
            _print_support(readout, run_dir)
            return 0

        if verb == "token-options" and len(args) >= 3:
            from transformers import AutoTokenizer

            from .jlens import token_options
            model_id = args[1]
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print(json.dumps(token_options.options_for(
                tokenizer, args[2],
                include_case_variants="--case-variants" in args), indent=2))
            return 0

        if verb == "qualify" and len(args) >= 3:
            # Stage 4. The first verb that spends 27B node time, and the one
            # that makes the freeze gate satisfiable: until it exists, no
            # J-lens study can freeze non-force by any supported path.
            from .jlens import qualification

            result = qualification.qualify(
                args[1], args[2],
                revision=_flag(args, "--revision"),
                layers=_int_list(_flag(args, "--layers")),
                alpha_range=_alpha_ladder(_flag(args, "--alpha-range")),
                token_id=(int(_flag(args, "--token-id"))
                          if _flag(args, "--token-id") else None),
                watchlist=_int_list(_flag(args, "--watchlist")),
                prompts_path=_flag(args, "--prompts"),
                battery_path=_flag(args, "--battery"),
                device=_flag(args, "--device"),
                dtype=_flag(args, "--dtype"),
                log=lambda m: print(m, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2, sort_keys=True))
            # A failed qualification is a STORED verdict, not a crash — but
            # the exit code must not report success for a lens that did not
            # qualify, or a script would seat it.
            return 0 if result["passed"] else 3

        if verb == "g0" and len(args) >= 2:
            from .jlens import g0

            report = g0.run(
                args[1],
                lens_id=_flag(args, "--lens"),
                revision=_flag(args, "--revision"),
                layers=_int_list(_flag(args, "--layers")),
                watchlist=_int_list(_flag(args, "--watchlist")),
                token_id=(int(_flag(args, "--token-id"))
                          if _flag(args, "--token-id") else None),
                piece=_flag(args, "--piece"),
                endpoint_path=_flag(args, "--endpoint"),
                alpha_range=_alpha_ladder(_flag(args, "--alpha-range")),
                top_k=int(_flag(args, "--top-k") or 10),
                band_stride=int(_flag(args, "--band-stride")
                                if _flag(args, "--band-stride") is not None
                                else g0.DEFAULT_BAND_STRIDE),
                prompts_path=_flag(args, "--prompts"),
                device=_flag(args, "--device"),
                dtype=_flag(args, "--dtype"),
                log=lambda m: print(m, file=sys.stderr, flush=True))
            print(json.dumps(report, indent=2, sort_keys=True))
            # Mechanics are the blocking half; the scientific verdict and the
            # arm licences are the gate's OUTPUT, not its success criterion.
            return 0 if report["mechanical"]["verdict"] == "pass" else 3

        if verb == "probe" and len(args) >= 2:
            from .jlens import probe as probe_mod

            text = _flag(args, "--prompt")
            prompt_file = _flag(args, "--prompt-file")
            if prompt_file:
                with open(prompt_file, encoding="utf-8") as handle:
                    text = handle.read()
            if not text:
                sys.stderr.write(
                    "steerlab-server jlens probe: --prompt or --prompt-file "
                    "is required — a probe reads ONE prompt at every armed "
                    "layer and position.\n")
                return 64
            pins = _flag(args, "--pin")
            report = probe_mod.probe(
                args[1],
                prompt=text,
                lens_id=_flag(args, "--lens"),
                revision=_flag(args, "--revision"),
                layers=_int_list(_flag(args, "--layers")),
                top_k=int(_flag(args, "--top-k") or 10),
                pin_words=[w.strip() for w in pins.split(",")] if pins else (),
                pin_ids=_int_list(_flag(args, "--pin-id")) or (),
                variant_path=_flag(args, "--variant"),
                prompt_mode=_flag(args, "--prompt-mode"),
                system_prompt=_flag(args, "--system-prompt"),
                directions=[d.strip()
                            for d in (_flag(args, "--directions") or "").split(",")
                            if d.strip()],
                position_stride=int(_flag(args, "--stride")
                                    or probe_mod.DEFAULT_POSITION_STRIDE),
                max_tokens=(int(_flag(args, "--max-tokens"))
                            if _flag(args, "--max-tokens") else None),
                device=_flag(args, "--device"), dtype=_flag(args, "--dtype"),
                log=lambda m: print(m, file=sys.stderr, flush=True))
            if "--json" in args:
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0
            _print_jlens_probe(report)
            return 0

        if verb == "report" and len(args) >= 2:
            from .jlens import report as report_mod

            built = report_mod.report(
                args[1],
                baseline=_flag(args, "--baseline")
                         or report_mod.BASELINE_CONDITION,
                band=_int_list(_flag(args, "--band")),
                bands=int(_flag(args, "--bands") or report_mod.DEFAULT_BANDS))
            if "--json" in args:
                print(json.dumps(built, indent=2, sort_keys=True))
                return 0
            _print_jlens_report(built)
            return 0

        if verb == "derive" and len(args) >= 3:
            from .jlens import derive
            lens_id = args[1]
            token = _flag(args, "--token-id")
            if token is None:
                sys.stderr.write(
                    "steerlab-server jlens derive: --token-id is required. A "
                    "direction is indexed by an exact vocabulary token, and "
                    "resolving a word here would be the silent mis-selection "
                    "the token-options verb exists to prevent.\n")
                return 64
            model_id = args[2]
            print(json.dumps(derive.derive_direction(
                lens_id, int(token), model_id=model_id,
                revision=_flag(args, "--revision"),
                name=_flag(args, "--name"),
                piece=_flag(args, "--piece")), indent=2))
            return 0
    except JLensError as exc:
        sys.stderr.write(f"steerlab-server jlens: {exc}\n")
        return 1
    except RuntimeError as exc:          # the installer's actionable failures
        sys.stderr.write(f"steerlab-server jlens: {exc}\n")
        return 1

    sys.stderr.write(
        "usage: steerlab-server jlens supported\n"
        "       steerlab-server jlens acquire <model-id>   # bytes -> HF cache (needs egress)\n"
        "       steerlab-server jlens import <model-id>    # convert -> workspace (offline)\n"
        "       steerlab-server jlens list\n"
        "       steerlab-server jlens inspect <lens-id>\n"
        "       steerlab-server jlens support <lens-id> <runDir>/<vectorName> "
        "[--layers 5,17,29] [--k 25] [--json]\n"
        "                                                  # what vocabulary is "
        "this vector made of\n"
        "       steerlab-server jlens token-options <model-id> <text> [--case-variants]\n"
        "       steerlab-server jlens derive <lens-id> <model-id> --token-id N "
        "[--piece P] [--name N] [--revision R]\n"
        "       steerlab-server jlens qualify <lens-id> <model-id> "
        "[--revision R] [--layers 20,26,32]\n"
        "                                                  [--alpha-range "
        "0.04:0.12 | 0.04,0.08,0.12] [--token-id N]\n"
        "                                                  [--watchlist "
        "id,id] [--prompts P] [--battery P] [--dtype D] [--device D]\n"
        "                                                  # Stage 4: accept "
        "this lens against ONE exact runtime (exit 3 = did not qualify)\n"
        "       steerlab-server jlens g0 <model-id> [--lens L] [--revision R] "
        "[--endpoint <rows.jsonl>] [--layers …]\n"
        "                                                  [--watchlist id,id] "
        "[--alpha-range …] [--top-k K] [--token-id N] [--piece P]\n"
        "                                                  [--band-stride N "
        "| 0 to skip the readable-band sweep] [--prompts P]\n"
        "                                                  [--device D] "
        "[--dtype T]\n"
        "                                                  # the G0 1b "
        "feasibility gate: two arms, verdicted separately (exit 3 = "
        "mechanics failed)\n"
        "       steerlab-server jlens probe <model-id> --prompt-file P | "
        "--prompt TEXT\n"
        "                                                  [--layers 31,40,48] "
        "[--pin \' therefore, however\'] [--pin-id N,N]\n"
        "                                                  [--lens L] "
        "[--revision R] [--directions <runDir>/<name>,…] [--top-k K]\n"
        "                                                  [--stride N] "
        "[--max-tokens N] [--device D] [--dtype T] [--json]\n"
        "                                                  [--variant "
        "<runDir>/<agent>.json] [--prompt-mode M] [--system-prompt S]\n"
        "                                                  # position x layer "
        "readout over ONE prompt: rank trajectories, direction cosines, top-k\n"
        "       steerlab-server jlens report <runDir> [--baseline NAME] "
        "[--band 20,26] [--bands N] [--json]\n"
        "                                                  # cross-condition "
        "J-Space roll-up into the run directory\n")
    return 64


def _panel(args: list[str]) -> int:
    """Panel scripts under prompts/panels/.

    Authoring is plain JSON editing now that panels are git-versioned recipes
    (plan B1); what a hand-editor needs is a way to FIND them and a way to be
    told what is wrong before a run does it silently. Cross-engine twin of
    `steerlab-cli panel`.
    """
    from . import cli_envelope as cli_envelope_module
    from .experiment import multi_agent

    verb = args[0] if args else None
    # Answered before anything else, so `panel compile` gets the redirect
    # rather than a usage line that implies the verb does not exist anywhere
    # (open-issues §18; the shape `data check <experiment>` already uses).
    # Returned as an int, not a CLIResult: `panel` is not an agent family here,
    # so nothing downstream would serialize an envelope for it.
    mac_authority = cli_envelope_module.MAC_AUTHORITY_VERBS.get("panel", {})
    if verb in mac_authority:
        refusal = _mac_authority_refusal(
            f"panel {verb}", mac_authority[verb],
            note="casting a panel binds a study's model and sampling settings "
                 "to a seat assignment, writes the compiled scenario as a "
                 "workspace input, and pins it into a draft manifest — all "
                 "authoring. Cast and freeze on the Mac, then submit the "
                 "frozen study here")
        return refusal.exit_code

    if verb == "list":
        panels = multi_agent.list_scenarios()
        if not panels:
            sys.stderr.write("no panels found\n")
            return 0
        for panel in panels:
            print(f"{panel['name']}\t{panel['path']}")
            print(f"  {panel['agents']} agents, {panel['turns']} turns")
        return 0

    if verb == "check" and len(args) >= 2:
        target = args[1]
        if not os.path.exists(target):
            match = next((p for p in multi_agent.list_scenarios()
                          if p["name"] == target
                          or os.path.basename(p["path"]) == target), None)
            if match is None:
                sys.stderr.write(
                    f"no panel '{target}' — try `panel list`\n")
                return 66
            target = match["path"]
        scenario, _ = multi_agent.load_scenario(target)
        try:
            multi_agent.validate(scenario)
        except multi_agent.ScenarioError as exc:
            sys.stderr.write(f"invalid: {exc}\n")
            return 65
        print(f"valid: {len(scenario.agents)} agents, "
              f"{len(scenario.turns)} turns — {target}")
        notes = multi_agent.advisories(scenario)
        for note in notes:
            print(f"ADVISORY: {note}")
        if notes:
            print(f"{len(notes)} advisory(ies) — these do not block a run; they "
                  "make prompts quietly wrong, so fix them before measuring.")
        return 0

    sys.stderr.write("usage: panel list | panel check <path-or-name>\n")
    return 64


def _apply_serve_posture(host: str, dev_open_flag: bool) -> int:
    """Resolve, apply, and announce the serve-time auth posture (WP-S).

    Returns 0 to continue, non-zero to refuse the start. The decision logic
    lives in ``api.posture.resolve_posture`` (pure, unit-tested); this wrapper
    is only environment writes and printing.
    """
    from .api import posture as _posture

    decision = _posture.resolve_posture(
        os.environ, host=host, dev_open_flag=dev_open_flag)
    for note in decision.notes:
        print(note, file=sys.stderr)
    if decision.refusal is not None:
        print(decision.refusal, file=sys.stderr)
        return 64
    os.environ["STEERLAB_AUTH_MODE"] = decision.auth_mode
    outcome = _posture.hydrate_token(decision)
    for note in outcome.notes:
        print(note, file=sys.stderr)
    if decision.auth_mode == "token":
        print(f"  token file: {outcome.path}  (the VALUE is never printed)",
              file=sys.stderr)
        print(_posture.authentication_hint(outcome.path), file=sys.stderr)
        if not outcome.present:
            print("  no token is configured — every /api route will answer "
                  "503 until one is.", file=sys.stderr)
    return 0


def _serve(args: list[str]) -> int:
    port = None  # explicit --port wins; role-specific defaults resolve below
    if "--port" in args:
        i = args.index("--port")
        if i + 1 < len(args):
            port = int(args[i + 1])
    host = os.environ.get("STEERLAB_BIND", "127.0.0.1")
    if "--host" in args:
        i = args.index("--host")
        if i + 1 < len(args):
            host = args[i + 1]
    # WP-S: the single-user open tier is now OPT-IN and named on the argv (the
    # env spelling exists so a launcher/app can set it without argv plumbing).
    dev_open_flag = "--dev-open-loopback" in args

    from .api.profile import server_role
    if server_role() == "gpu-session":
        # GPU session worker (GPU-SESSION-PLAN Wave 1). It binds non-loopback
        # so the controller can proxy to it over the cluster network — the
        # same bind decision as controller-job.sbatch.template, with the same
        # compensating control enforced HERE in Python (no launch path can
        # skip a bash guard that was never run): token mode on every route,
        # and a refusal to open a non-loopback socket without a token.
        #
        # The token is hydrated from a FILE (STEERLAB_AUTH_TOKEN_FILE,
        # default the bootstrap-written ~/.steerlab-token) so the secret
        # never has to ride through the sbatch bundle, whose run.sbatch and
        # bundle.json are durable on-disk artifacts.
        # Hydrate-only (never generate): a freshly minted secret the
        # controller has not been handed is a silent mismatch, not a working
        # worker — the refusal below is the honest outcome.
        from .api import posture as _posture
        _posture.hydrate_token(
            _posture.PostureDecision(auth_mode="token", source="explicit",
                                     hydrate_token=True))
        if (host not in _posture.LOOPBACK_HOSTS
                and not os.environ.get("STEERLAB_AUTH_TOKEN")):
            print("steerlab-server: refusing to start the GPU session worker:"
                  f" binding {host} (non-loopback) without STEERLAB_AUTH_TOKEN"
                  " would expose an open instrument to every user on the "
                  "cluster network. Set STEERLAB_AUTH_TOKEN, or point "
                  "STEERLAB_AUTH_TOKEN_FILE at the token file (bootstrap.sh "
                  "writes ~/.steerlab-token).", file=sys.stderr)
            return 64
        if port is None:
            # No explicit --port: honor a numeric STEERLAB_SESSION_PORT, else
            # derive an allocation-unique port from SLURM_JOB_ID ("auto", the
            # session-start default since 2026-07-17 — a fixed 8081 collided
            # when two session jobs landed on one multi-GPU node).
            from .api.gpu_session import derive_session_port
            raw = (os.environ.get("STEERLAB_SESSION_PORT") or "").strip()
            port = int(raw) if raw.isdigit() else derive_session_port()
        # The worker's discovery record needs the real serving port; the env
        # var is how the app module learns it (set here — resolved to the
        # ACTUAL number, never "auto" — for both manual and scheduled runs).
        os.environ["STEERLAB_SESSION_PORT"] = str(port)
    if port is None:
        port = 8080

    # ── Security posture (WP-S) ───────────────────────────────────────────
    # Resolved HERE, before the app is built, and EXPORTED so the per-request
    # ServerProfile.from_env sees an explicit decision instead of a default.
    # ``from_env`` itself stays a pure reader defaulting to "none": tests and
    # embedders construct the app directly and must keep pinning what they
    # pin. Nothing below returns a secret to stdout — only the file path.
    if _apply_serve_posture(host, dev_open_flag) != 0:
        return 64

    # The artifact root is --root, STEERLAB_ROOT, or the CURRENT DIRECTORY —
    # starting from the wrong cwd (e.g. Server/) silently points prompts/
    # experiments/runs at a tree that doesn't exist. Announce it instead.
    from .experiment import paths
    root = os.path.realpath(paths.project_root())
    origin = ("--root" if _ROOT_FROM_FLAG
              else "STEERLAB_ROOT" if os.environ.get("STEERLAB_ROOT")
              else "current directory")
    print("=" * 60, file=sys.stderr)
    print(f"artifact root: {root}", file=sys.stderr)
    print(f"  (resolved from {origin}; prompts/, experiments/ and runs/ "
          "live here)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    missing = [name for name in ("prompts", "experiments")
               if not os.path.isdir(os.path.join(root, name))]
    if missing:
        print(f"WARNING: artifact root {root!r} has no "
              f"{' or '.join(missing)} directory — start the server from the "
              "project root, pass serve --root, or set STEERLAB_ROOT to the "
              "tree that holds prompts/ and experiments/", file=sys.stderr)
    if paths.looks_like_source_checkout(root):
        print("WARNING: artifact root looks like the SteerLab SOURCE "
              "CHECKOUT, not a data workspace — server-side builds/authoring "
              "will write into the code repo.", file=sys.stderr)
        print("         Pass serve --root <workspace> (or set STEERLAB_ROOT) "
              "to pair the server with the app's data workspace.",
              file=sys.stderr)
    # The controller's own launching script (open-issues §1 field report,
    # 2026-08-20). A daemon-in-a-job controller whose script exports no chain
    # marker was launched from a PRE-CHAIN rendered copy and will not resubmit
    # itself at walltime. Loud, and only that: a chain-less controller serves
    # perfectly well, it just needs manual cycling — refusing to start would
    # cost the researcher a working daemon over a resubmission convenience.
    # Silent for every non-controller and non-Slurm serve.
    from . import controller_render
    from .api.profile import ServerProfile as _ServerProfile
    for line in controller_render.boot_warning_lines(_ServerProfile.from_env()):
        print(line, file=sys.stderr)
    if not _hf_token_configured() and os.environ.get("HF_HUB_OFFLINE") != "1":
        # Suppressed under HF_HUB_OFFLINE=1: an offline server (the GPU-
        # session worker) loads from cache only — warning about download
        # auth there reads as if the server might try the Hub (2026-07-17).
        print("INFO: no Hugging Face token found (HF_TOKEN / "
              "HUGGING_FACE_HUB_TOKEN unset, no cached login) — unauthenticated "
              "Hub requests are rate-limited and gated repos (e.g. "
              "google/gemma-3-*) will fail to download. Set HF_TOKEN to a read "
              "token if you need them.", file=sys.stderr)
    import uvicorn
    uvicorn.run("steerlab_server.api.app:app", host=host, port=port, log_level="info")
    return 0


def _hf_token_configured() -> bool:
    """True when HF Hub auth is available: env token or a huggingface-cli
    login stored under the HF cache."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    cache = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.isfile(os.path.join(cache, "token"))


def _profile(args: list[str]) -> int:
    from .api.profile import ServerProfile, capability_snapshot, validate_profile
    verb = args[0] if args else "show"
    if verb not in {"show", "validate"}:
        sys.stderr.write("usage: profile show|validate [--json]\n")
        return 64
    profile = ServerProfile.from_env()
    data = capability_snapshot()
    if "--json" in args:
        print(json.dumps(validate_profile() if verb == "validate" else data,
                         indent=2, sort_keys=True))
    elif verb == "show":
        print(f"profile: {profile.profile}")
        print(f"topology: {profile.launch_topology}")
        print(f"executor: {profile.executor}")
        print(f"auth: {profile.auth_mode}")
        print(f"root: {profile.root}")
        for warning in profile.warnings():
            print(f"WARNING: {warning}")
    if verb == "validate":
        report = validate_profile()
        if "--json" not in args:
            for check in report["checks"]:
                print(f"{check['status'].upper():4} {check['name']}: {check['message']}")
            print(f"{report['failures']} failure(s), {report['warnings']} warning(s)")
        return 0 if report["ok"] else 1
    return 0


def _jobs(args: list[str]):
    from .api.jobs import JobManager
    if not args or args[0] == "list":
        from .cli_envelope import CLIResult
        mgr = JobManager()
        jobs = [j.to_dict() for j in mgr.list()]
        # Human mode keeps the exact raw document it always wrote; JSON mode
        # gets the same array inside `result.jobs`, under a versioned envelope
        # (audit §2.1: "raw JSON, unversioned").
        print(json.dumps({"jobs": jobs}, indent=2, sort_keys=True))
        running = sum(1 for job in jobs if job.get("status") == "running")
        return CLIResult(
            message=f"{len(jobs)} job(s), {running} running",
            payload={"count": len(jobs), "runningCount": running,
                     "jobs": jobs})
    if args[0] == "reconcile" and len(args) >= 2:
        mgr = JobManager()
        print(json.dumps({"reconciled": mgr.reconcile(args[1])}, sort_keys=True))
        return 0
    sys.stderr.write("usage: jobs list | jobs reconcile <records-dir>\n")
    return 64


def _bundle(args: list[str]) -> int:
    from .experiment import bundles

    from .api.executors import SlurmExecutor, SlurmResources
    if not args or args[0] not in {
        "run", "evidence", "inspect", "import", "execute", "create", "submit"
    }:
        sys.stderr.write(
            "usage:\n"
            "  bundle run <experiment> [--out path]\n"
            "  bundle evidence <run-dir> [--out path]\n"
            "  bundle inspect <bundle.tar.gz>\n"
            "  bundle import <bundle.tar.gz> [--target root] [--overwrite]\n"
            "  bundle execute <bundle.tar.gz> --verb <verify|extract|validate|sweep|run|evaluate|analyze|pipeline> [--target root] [--shard k/K] [--resume <run-dir>]\n"
            "  bundle create|submit <bundle-dir> [--gres A100] [--walltime HH:MM:SS] -- <cmd...>\n")
        return 64
    verb = args[0]
    if verb == "run":
        if len(args) < 2:
            sys.stderr.write("experiment name required\n")
            return 64
        print(json.dumps(
            bundles.package_experiment(args[1], output_path=_flag(args, "--out")),
            indent=2, sort_keys=True))
        return 0
    if verb == "evidence":
        if len(args) < 2:
            sys.stderr.write("run directory required\n")
            return 64
        # Same contract as POST /api/bundles/evidence: a ledger-only
        # failure record answers a structured skip, never an error.
        skip = bundles.failure_record_skip(args[1])
        print(json.dumps(
            skip if skip is not None else
            bundles.package_evidence(args[1], output_path=_flag(args, "--out")),
            indent=2, sort_keys=True))
        return 0
    if verb == "inspect":
        if len(args) < 2:
            sys.stderr.write("bundle path required\n")
            return 64
        print(json.dumps(bundles.inspect_bundle(args[1]), indent=2, sort_keys=True))
        return 0
    if verb == "import":
        if len(args) < 2:
            sys.stderr.write("bundle path required\n")
            return 64
        print(json.dumps(
            bundles.import_bundle(
                args[1], target_root=_flag(args, "--target"),
                allow_overwrite=("--overwrite" in args)),
            indent=2, sort_keys=True))
        return 0
    if verb == "execute":
        if len(args) < 2:
            sys.stderr.write("bundle path required\n")
            return 64
        task_verb = _flag(args, "--verb")
        if not task_verb:
            sys.stderr.write("--verb required for bundle execute\n")
            return 64
        # Checkpoint-on-signal for the sbatch child (WS2): the rendered script
        # traps USR1/TERM and forwards to this process; the study loop parks
        # the run (fsync + resume-state.json) and we exit 85 so the scheduler
        # record distinguishes "checkpointed, resumable" from a failure.
        # Installed for the verbs with a checkpoint consumer — run, and
        # pipeline (whose run STAGE polls the same flag); the other verbs
        # have none, and a handler nothing polls would swallow SIGTERM.
        from .experiment import resume as resume_mod
        flag = (resume_mod.CheckpointFlag().install()
                if task_verb in ("run", "pipeline") else None)
        try:
            result = bundles.execute_run_bundle(
                args[1], verb=task_verb, target_root=_flag(args, "--target"),
                dtype=_flag(args, "--dtype") or "auto",
                device=_flag(args, "--device"),
                prompts_path=_flag(args, "--prompts"),
                source_path=_flag(args, "--source"),
                package_evidence_on_complete=("--no-evidence" not in args),
                record_path=_flag(args, "--record"),
                # Targeted retry through the BUNDLE path (2026-07-24): the
                # Slurm case is the one that matters most, since that is
                # where a failed judged evaluate is expensive to redo.
                resume_from=_flag(args, "--resume-from"),
                # `--resume <run-dir>` continues a PARKED run/sweep/pipeline
                # through the bundle path (2026-08-23). A relative directory
                # is resolved against `--target`, so the flag behaves the same
                # whether the child was started by hand or by the renderer,
                # which cd's into its own slurm directory before srun.
                resume_directory=_flag(args, "--resume"),
                checkpoint=flag,
                # Multi-GPU fan-out: generate ONLY the records in shard k of
                # K (0-based, balanced contiguous ranges of the run's
                # deterministic record order). Byte-identical by record
                # independence; the submitting server merges the partials.
                shard=_flag(args, "--shard"))
        except resume_mod.CheckpointRequested as exc:
            sys.stderr.write(f"bundle execute checkpointed: {exc}\n")
            return resume_mod.CHECKPOINT_EXIT_CODE
        except Exception as exc:  # noqa: BLE001 - CLI needs a useful process error
            sys.stderr.write(f"bundle execute failed: {type(exc).__name__}: {exc}\n")
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if len(args) < 2:
        sys.stderr.write("bundle directory required\n")
        return 64
    bundle_dir = args[1]
    rest = args[2:]
    if "--" not in rest:
        sys.stderr.write("command separator '--' required\n")
        return 64
    sep = rest.index("--")
    opts, command = rest[:sep], rest[sep + 1:]
    if not command:
        sys.stderr.write("command required after '--'\n")
        return 64
    resources = SlurmResources.from_env(job_name=_flag(opts, "--job-name") or "steerlab")
    resources.gres = _flag(opts, "--gres") or resources.gres
    resources.walltime = _flag(opts, "--walltime") or resources.walltime
    resources.memory = _flag(opts, "--mem") or resources.memory
    if _flag(opts, "--partition"):
        resources.partition = _flag(opts, "--partition")
    bundle = SlurmExecutor().create_bundle(bundle_dir, command, resources=resources)
    if verb == "submit":
        slurm_id = SlurmExecutor().submit(bundle)
        print(json.dumps({"slurmJobID": slurm_id, "bundle": bundle.to_dict()},
                         indent=2, sort_keys=True))
    else:
        print(json.dumps({"bundle": bundle.to_dict()}, indent=2, sort_keys=True))
    return 0


def _study(args: list[str]):
    if not args or args[0] != "submit" or len(args) < 2:
        sys.stderr.write(
            "usage: study submit <experiment> --verb <run|validate|extract|sweep|evaluate|analyze|verify|pipeline> "
            "[--executor local|slurm] [--dry-run] [--force] [--parallel N] "
            "[--gres A100] [--partition P] [--mem M] [--walltime HH:MM:SS] "
            "[--job-name N]\n"
            "  [--target root] [--dtype D] [--device DEV] [--prompts P] "
            "[--source S] [--resume <run-dir>] [--dependency <spec>] "
            "[--no-evidence]\n"
            "(--force overrides a failing preflight verdict — recorded loudly "
            "on the job)\n"
            "(--parallel N shards a Slurm 'run' across N GPU jobs, cap 64; the "
            "RUNNING server's reconciler merges the partials)\n"
            "(GPU type is chosen by --gres, e.g. --gres A100; unset falls back "
            "to STEERLAB_SLURM_GRES from the site profile)\n")
        return 64
    from .api.jobs import JobManager
    from .api.submissions import SubmissionRefusal, submit_study

    experiment = args[1]
    verb = _flag(args, "--verb") or "run"
    # `--parallel-jobs` is accepted as an alias for the API's `parallelJobs`
    # spelling: a flag the CLI silently ignored would run one job while the
    # researcher believed they had fanned out.
    parallel_named = ("--parallel" in args) or ("--parallel-jobs" in args)
    parallel_raw = _flag(args, "--parallel")
    if parallel_raw is None:
        parallel_raw = _flag(args, "--parallel-jobs")
    if parallel_raw is None and parallel_named:
        # `_flag` reads "the next argument" and yields None at end of line;
        # a trailing `--parallel` must not resolve to a silent single job.
        sys.stderr.write("--parallel requires a number of shard jobs\n")
        return 64
    if parallel_raw is None:
        parallel = 1
    else:
        try:
            parallel = int(parallel_raw)
        except ValueError:
            sys.stderr.write(
                f"--parallel expects an integer number of shard jobs "
                f"(got {parallel_raw!r})\n")
            return 64
        if parallel < 1:
            sys.stderr.write("--parallel must be at least 1\n")
            return 64
    resources = {
        key: value for key, value in {
            "gres": _flag(args, "--gres"),
            "walltime": _flag(args, "--walltime"),
            "memory": _flag(args, "--mem"),
            "partition": _flag(args, "--partition"),
            "jobName": _flag(args, "--job-name"),
        }.items() if value is not None
    }
    try:
        # ONE jobs DB, two writer processes. `JobManager()` here opens the very
        # same `jobs.sqlite` the daemon uses (STEERLAB_METADATA_ROOT, default
        # <root>/.steerlab) — so run this CLI with the SAME STEERLAB_ROOT /
        # STEERLAB_METADATA_ROOT as the server, or the parent + shard records
        # land in a store the daemon's reconciler never reads and the partials
        # are never merged. Cross-process safety is SQLite's own file locking
        # plus the store's 30 s busy timeout (the in-process RLock only
        # serializes this process); every write is a single short statement
        # inside one transaction, so contention with the 15 s reconcile tick
        # resolves inside the timeout rather than raising "database is locked".
        jobs = JobManager()
        submission = submit_study(
            experiment, verb=verb, jobs=jobs,
            executor=_flag(args, "--executor"), dry_run=("--dry-run" in args),
            target_root=_flag(args, "--target"), dtype=_flag(args, "--dtype") or "auto",
            device=_flag(args, "--device"), prompts_path=_flag(args, "--prompts"),
            source_path=_flag(args, "--source"),
            # The two flags that close the reasons an operator hand-rolls an
            # sbatch (2026-08-23): a parked run continues through the RENDERER
            # (node-scratch gres + cleanup trap), and a job can be chained
            # behind another with Slurm's own dependency vocabulary.
            resume_directory=_flag(args, "--resume"),
            dependency=_flag(args, "--dependency"),
            package_evidence=("--no-evidence" not in args), resources=resources,
            force=("--force" in args), parallel_jobs=parallel)
    except SubmissionRefusal as exc:
        # A TYPED refusal, not a failure: the request is well formed and a
        # policy/precondition declined it, so `--json` carries the reason and
        # the repair rather than "exited 1 — see the diagnostics on stderr"
        # (which is how the ledger's operator ended up with no usable words).
        # Human mode is unchanged: the same stderr shape, the same exit 1.
        from .cli_envelope import CLIResult

        sys.stderr.write(f"study submit refused: {exc}\n  {exc.repair_action}\n")
        return CLIResult(
            state="refused", exit_code=1, code=exc.code, message=str(exc),
            repair_action=exc.repair_action,
            payload={"experiment": experiment, "verb": verb})
    except Exception as exc:  # noqa: BLE001 - command-line setup error
        sys.stderr.write(f"study submit failed: {type(exc).__name__}: {exc}\n")
        return 1
    # An aborted sharded fan-out is reported through the PARENT JOB RECORD,
    # not an exception: `_submit_sharded_bundle` scancels the submitted
    # siblings, fails the parent, and returns a submission with no shard ids
    # so the app can render the abort from the record. Exiting 0 on that
    # shape hid five QOSMaxSubmitJobPerUserLimit-refused submissions on
    # 2026-08-09 — read the record back and put sbatch's own words on the
    # terminal.
    from .cli_envelope import CLIResult, next_action
    parent = jobs.get(submission.job_id)
    if parent is not None and parent.status == "failed":
        print(json.dumps(submission.to_dict(), indent=2, sort_keys=True))
        sys.stderr.write(
            f"study submit failed: parent job {submission.job_id} is marked "
            f"failed — {parent.error or 'no error text recorded on the job'}\n")
        return CLIResult(
            state="failed", exit_code=1, code="submissionFailed",
            message=(f"parent job {submission.job_id} is marked failed — "
                     f"{parent.error or 'no error text recorded on the job'}"),
            repair_action=(
                "read `squeue`/`sacct` for the scheduler's own words, then "
                f"steerlab-server study submit {experiment} --verb {verb}"),
            payload={"experiment": experiment, "verb": verb,
                     "submission": submission.to_dict()})
    if submission.shard_job_ids:
        # The merge is NOT performed by this process: `merge_shard_runs` runs
        # only from `JobManager._reconcile_shard_parents`, driven by the
        # monitor thread that `steerlab-server serve` starts. Without a running
        # server the shards will complete and the partials will sit unmerged.
        sys.stderr.write(
            f"fanned out across {len(submission.shard_job_ids)} shard jobs "
            f"under parent {submission.job_id}; a RUNNING `steerlab-server "
            "serve` on this store merges the partials when every shard "
            "succeeds\n")
    print(json.dumps(submission.to_dict(), indent=2, sort_keys=True))
    # A submitted job is asynchronous work in flight, not a finished target:
    # `pending` (exit 12 in JSON mode, 0 in human mode as always) is what the
    # state vocabulary is FOR, and what the cluster family already reports for
    # the same situation.
    return CLIResult(
        state="pending",
        message=(f"submitted '{experiment}' --verb {verb} as job "
                 f"{submission.job_id}"
                 + (f", fanned out across {len(submission.shard_job_ids)} "
                    "shard job(s)" if submission.shard_job_ids else "")),
        changed=True,
        payload={"experiment": experiment, "verb": verb,
                 "jobID": submission.job_id,
                 "shardJobIDs": list(submission.shard_job_ids or []),
                 "dryRun": "--dry-run" in args,
                 "submission": submission.to_dict()},
        next_action=next_action("jobs list"))


_FINETUNE_USAGE = (
    "usage: steerlab-server finetune execute <job-dir> [--record rec.json]\n"
    "       steerlab-server finetune plan   <finetune-config.json>\n"
    "       steerlab-server finetune train  <finetune-config.json>\n"
    "       steerlab-server finetune submit <finetune-request.json> "
    "[--plan-only] [--confirm-plan HASH]\n"
    "              [--dry-run] [--force] [--gres G] [--walltime HH:MM:SS] "
    "[--mem M] [--partition P] [--job-name N]\n"
    "(submit takes the WIRE request — camelCase, exactly the JSON POSTed to "
    "/api/finetune/plan\n"
    " and /api/finetune/submit. plan/train take the resolved snake_case "
    "LoRAConfig instead.)\n"
    "(evidence-grade submits need a confirmed plan: --plan-only prints the "
    "planHash, then\n"
    " re-run with --confirm-plan <hash>.)\n")

#: ``finetune submit``'s declared surface. Strict: an undeclared flag is a
#: usage refusal before anything is read, never a silently ignored request
#: (a `--dry-run` typo'd into oblivion would submit the job it was meant to
#: withhold).
_FINETUNE_SUBMIT_BOOLEAN_FLAGS = ("--plan-only", "--dry-run", "--force")
_FINETUNE_SUBMIT_VALUE_FLAGS = {
    "--confirm-plan": "expectedPlanHash",
    "--gres": "gres",
    "--walltime": "walltime",
    "--mem": "memory",
    "--partition": "partition",
    "--job-name": "jobName",
}

#: Keys that identify a snake_case ``LoRAConfig`` handed to the wire verb.
#: The two spellings are the seam open issues §5 names: the routes (and this
#: verb) speak camelCase, ``finetune plan``/``train`` and the submission's own
#: ``finetune-config.json`` speak the resolved dataclass.
_LORA_CONFIG_MARKERS = ("base_model_id", "training_mode", "train_paths",
                        "dataset_root", "expected_hashes", "run_directory")


def _finetune(args: list[str]) -> int:
    """LoRA fine-tuning, headless.

    ``execute`` is the SLURM CHILD entry point for
    ``POST /api/finetune/submit``: it reads the job directory the submission
    materialized (staged dataset bytes + resolved config + confirmed plan),
    installs the checkpoint flag, and trains INTO ``<job-dir>/run``. A
    requeue re-executes the identical command, so it adopts the checkpoint
    that is already there instead of minting a second run — and a job
    directory whose adapter sidecar exists returns success idempotently
    (at-most-once finalization, contract §8).

    ``submit`` is the terminal twin of ``POST /api/finetune/plan`` +
    ``POST /api/finetune/submit`` (open issues §5): evidence-grade training
    submission existed only over HTTP, so all 47 trainings were curl'd by
    hand — and the wire's camelCase against ``LoRAConfig``'s snake_case
    tripped a first-time caller. This verb drives the same two functions the
    routes drive, from ONE wire-shaped request file, with no HTTP hop.

    Exit codes: 0 = adapter written (or already complete), or submitted; 85 =
    checkpointed and resumable (the scheduler record then reads
    ``checkpointed``, and auto-resubmit re-sbatches this same script); 2 = an
    unreadable config or a refused request (plan drift, an unconfirmed
    evidence-grade plan, a failing preflight); 1 = failed; 64 = usage.
    """
    if args and "--help" in args:
        # Asking what the arguments are was never an error (WP0 step 11's
        # rule, applied to this family's own usage page).
        sys.stdout.write(_FINETUNE_USAGE)
        return 0
    if not args or args[0] not in {"execute", "plan", "train", "submit"}:
        sys.stderr.write(_FINETUNE_USAGE)
        return 64
    verb = args[0]
    if len(args) < 2:
        sys.stderr.write(_FINETUNE_USAGE)
        return 64
    target = args[1]

    from .api import finetune_submission as ft
    from .experiment import lora_train

    if verb == "submit":
        return _finetune_submit(args[1:], ft)

    if verb in ("plan", "train"):
        # Local convenience twins of the routes: the same config JSON the
        # submission writes, run without an HTTP hop.
        try:
            with open(target, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"finetune {verb}: cannot read {target}: {exc}\n")
            return 2
        if not isinstance(payload, dict):
            sys.stderr.write(f"finetune {verb}: {target} must hold a JSON object\n")
            return 2
        run_directory = payload.get("run_directory")
        try:
            config = lora_train.config_from_dict(
                {k: v for k, v in payload.items() if k != "run_directory"})
        except lora_train.LoRATrainError as exc:
            sys.stderr.write(f"finetune {verb}: {exc}\n")
            return 2
        if verb == "plan":
            from .api.profile import ServerProfile
            try:
                loaded = _load_finetune_splits(config)
            except Exception as exc:  # noqa: BLE001 - config/dataset problem
                sys.stderr.write(f"finetune plan: {type(exc).__name__}: {exc}\n")
                return 2
            schedule = lora_train.plan_schedule(config, len(loaded.train_rows))
            plan = {
                "resolvedRevision": ft.resolve_revision(config),
                "dtype": ft.resolve_plan_dtype(config, ServerProfile.from_env()),
                "trainingMode": config.training_mode,
                "evidenceGrade": config.evidence_grade,
                "selectionMetric": config.selection_metric,
                "schedule": schedule.to_dict(),
                "evidenceRefusals": lora_train.evidence_refusals(config),
            }
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        try:
            run_dir = lora_train.train(config, run_directory=run_directory)
        except Exception as exc:  # noqa: BLE001 - CLI needs a useful error
            sys.stderr.write(f"finetune train failed: {type(exc).__name__}: {exc}\n")
            return 1
        print(json.dumps({"runDirectory": run_dir}, indent=2, sort_keys=True))
        return 0

    # --- execute ------------------------------------------------------------
    from .experiment import bundles, resume as resume_mod

    record_path = _flag(args, "--record")
    started = time.time()
    try:
        config, run_directory = ft.load_job_config(target)
    except ft.FineTuneRequestError as exc:
        sys.stderr.write(f"finetune execute: {exc}\n")
        return 64
    if ft.is_finished(run_directory, config):
        # A requeue that raced the finalization: idempotent success, never a
        # second run (complete runs are immutable).
        message = f"finetune execute: {run_directory} is already complete"
        sys.stderr.write(message + "\n")
        if record_path:
            bundles.write_child_record(
                record_path, kind="finetune-execute", status="succeeded",
                result={"runDirectory": run_directory, "alreadyComplete": True},
                logs=[message], elapsed_seconds=time.time() - started)
        print(json.dumps({"runDirectory": run_directory,
                          "alreadyComplete": True}, indent=2, sort_keys=True))
        return 0
    resuming = os.path.isfile(
        os.path.join(run_directory, resume_mod.RESUME_STATE_FILENAME))
    os.makedirs(run_directory, exist_ok=True)
    flag = resume_mod.CheckpointFlag().install()
    logs: list[str] = []
    try:
        lora_train.train(config, log=logs.append, run_directory=run_directory,
                         resume=resuming, checkpoint_flag=flag)
    except resume_mod.CheckpointRequested as exc:
        # No child record on purpose (same contract as `bundle execute`): the
        # reconciler reads sacct's exit code 85 as the non-terminal
        # "checkpointed" status, which is what arms auto-resubmit.
        sys.stderr.write(f"finetune execute checkpointed: {exc}\n")
        return resume_mod.CHECKPOINT_EXIT_CODE
    except Exception as exc:  # noqa: BLE001 - CLI needs a useful process error
        sys.stderr.write(
            f"finetune execute failed: {type(exc).__name__}: {exc}\n")
        if record_path:
            bundles.write_child_record(
                record_path, kind="finetune-execute", status="failed",
                error=f"{type(exc).__name__}: {exc}", logs=logs[-200:],
                elapsed_seconds=time.time() - started)
        return 1
    result = {"runDirectory": run_directory,
              "adapterDirectory": os.path.join(
                  run_directory, config.output_name
                  or f"lora-{os.path.basename(config.base_model_id)}"),
              "resumed": resuming}
    if record_path:
        bundles.write_child_record(
            record_path, kind="finetune-execute", status="succeeded",
            result=result, logs=logs[-200:],
            elapsed_seconds=time.time() - started)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_finetune_submit_args(args: list[str]) -> tuple[dict | None, str]:
    """``(options, error)``. Strict: every token after the request file must
    be a declared flag, and a value flag at end of line is a refusal rather
    than a silent None."""
    options: dict = {"path": args[0], "resources": {}}
    index = 1
    while index < len(args):
        token = args[index]
        if token in _FINETUNE_SUBMIT_BOOLEAN_FLAGS:
            options[token.lstrip("-")] = True
            index += 1
            continue
        if token in _FINETUNE_SUBMIT_VALUE_FLAGS:
            if index + 1 >= len(args):
                return None, f"{token} requires a value"
            value = args[index + 1]
            if token == "--confirm-plan":
                options["confirmPlan"] = value
            else:
                options["resources"][_FINETUNE_SUBMIT_VALUE_FLAGS[token]] = value
            index += 2
            continue
        return None, (f"unknown flag {token!r}" if token.startswith("-")
                      else f"unexpected argument {token!r}")
    return options, ""


def _finetune_submit(args: list[str], ft) -> int:
    """``finetune submit`` — the evidence-grade LoRA path, headless.

    The verb performs the route sequence a caller used to curl by hand:
    plan → echo ``planHash`` → submit. It adds no training logic of its own
    — ``plan_response`` and ``submit_finetune`` are the same functions
    ``POST /api/finetune/plan`` and ``POST /api/finetune/submit`` call, so
    the refusals, the preflight, and the plan hash are the routes' own.

    The request file is the WIRE body (camelCase), because that is what the
    routes accept natively: the same JSON can be POSTed verbatim or handed
    to this verb, and nobody hand-translates anything.
    """
    from .api.profile import ServerProfile
    from .api.safe_paths import SafePathResolver
    from .api.jobs import JobManager

    options, error = _parse_finetune_submit_args(args)
    if options is None:
        sys.stderr.write(f"finetune submit: {error}\n" + _FINETUNE_USAGE)
        return 64
    path = options["path"]
    try:
        with open(path, encoding="utf-8") as handle:
            request = json.load(handle)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"finetune submit: cannot read {path}: {exc}\n")
        return 2
    if not isinstance(request, dict):
        sys.stderr.write(
            f"finetune submit: {path} must hold a JSON object\n")
        return 2
    spelled_snake = sorted(k for k in request if k in _LORA_CONFIG_MARKERS)
    if spelled_snake:
        sys.stderr.write(
            f"finetune submit: {path} is a resolved LoRAConfig "
            f"(snake_case key(s): {', '.join(spelled_snake)}), not a "
            "fine-tune REQUEST. This verb takes the wire body — camelCase "
            "baseModelID / trainingMode / dataset.files[].sha256 — exactly "
            "as POSTed to /api/finetune/plan. The snake_case config is what "
            "`finetune plan` and `finetune train` take, and what a "
            "submission writes into its job directory as "
            "finetune-config.json.\n")
        return 2

    # The plan endpoint takes the REQUEST only; the submit-only keys are the
    # client's own additions (the shipped Swift client strips them too).
    plan_body = {k: v for k, v in request.items()
                 if k not in ("resources", "force", "dryRun")}
    resolver = SafePathResolver(ServerProfile.from_env())
    try:
        planned = ft.plan_response(plan_body, resolver=resolver)
    except ft.FineTuneRequestError as exc:
        sys.stderr.write(f"finetune submit: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001 - e.g. a dataset path outside the
        # workspace, which the route surfaces as HTTP 400. Planning has no
        # side effects, so a failure here is always a request problem.
        sys.stderr.write(
            f"finetune submit: cannot plan this request "
            f"({type(exc).__name__}: {getattr(exc, 'detail', exc)})\n")
        return 2
    plan_hash = planned["planHash"]
    # The ECHO, always, on stderr — one document stays on stdout.
    sys.stderr.write(f"finetune submit: planHash {plan_hash}\n")
    for refusal in planned.get("evidenceRefusals") or []:
        sys.stderr.write(f"  evidence refusal: {refusal}\n")
    if options.get("plan-only"):
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 0

    confirmed = options.get("confirmPlan")
    declared = request.get("expectedPlanHash")
    if confirmed and declared and confirmed.lower() != str(declared).lower():
        sys.stderr.write(
            f"finetune submit: --confirm-plan {confirmed} contradicts the "
            f"request's own expectedPlanHash {declared} — pass one, or make "
            "them agree\n")
        return 2

    body = dict(request)
    if confirmed:
        body["expectedPlanHash"] = confirmed
    resources = {**(request.get("resources") or {}), **options["resources"]}
    if resources:
        body["resources"] = resources
    if options.get("force"):
        body["force"] = True
    if options.get("dry-run"):
        body["dryRun"] = True

    try:
        out = ft.submit_finetune(body, jobs=JobManager(), resolver=resolver)
    except ft.PreflightRejection as exc:
        sys.stderr.write(f"finetune submit refused by preflight: {exc}\n")
        print(json.dumps({"preflight": exc.preflight}, indent=2,
                         sort_keys=True))
        return 2
    except ft.FineTuneRequestError as exc:
        sys.stderr.write(
            f"finetune submit: {exc}\n"
            "  repair: steerlab-server finetune submit "
            f"{path} --plan-only   # prints the planHash\n"
            "          steerlab-server finetune submit "
            f"{path} --confirm-plan <hash>\n")
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI needs a useful error
        sys.stderr.write(
            f"finetune submit failed: {type(exc).__name__}: {exc}\n")
        return 1
    sys.stderr.write(
        f"finetune submit: {'prepared' if out.get('dryRun') else 'submitted'} "
        f"job {out['jobId']}"
        + (f" as Slurm job {out['slurmJobID']}" if out.get("slurmJobID") else "")
        + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _load_finetune_splits(config):
    """Load a v2 config's splits (helper for ``finetune plan``)."""
    from .experiment import lora_data
    return lora_data.load_split_rows(config.dataset_spec(),
                                     evidence_grade=config.evidence_grade,
                                     root=config.dataset_root)


def _housekeeping(args: list[str]) -> int:
    """WS3 chores, headless: the same functions the /api/housekeeping routes
    call, so the health card and the terminal always agree."""
    from .api import housekeeping

    if not args or args[0] == "status":
        if "--refresh" in args:
            from .api.jobs import JobManager
            report = housekeeping.refresh(jobs=JobManager())
        else:
            report = housekeeping.status()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args[0] == "maintenance" and len(args) >= 2 and args[1] == "set":
        path = _flag(args, "--file")
        if not path:
            sys.stderr.write("usage: housekeeping maintenance set --file <windows.json>\n")
            return 64
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"housekeeping: cannot read {path}: {exc}\n")
            return 1
        windows = data.get("windows") if isinstance(data, dict) else data
        try:
            stored = housekeeping.write_maintenance(windows)
        except ValueError as exc:
            sys.stderr.write(f"housekeeping: {exc}\n")
            return 1
        print(json.dumps(stored, indent=2, sort_keys=True))
        return 0
    sys.stderr.write(
        "usage: housekeeping status [--refresh] "
        "| housekeeping maintenance set --file <windows.json>\n")
    return 64


def _run_config(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    task = config.get("task", "smoke-test")
    if task == "smoke-test":
        from .experiment.smoke_test import SmokeTestConfig, run
        run(SmokeTestConfig.from_dict(config))
        print("SMOKE TEST PASSED")
        return 0
    if task == "toy-concept":
        from .experiment.toy_concept import ToyConceptConfig, run
        run(ToyConceptConfig.from_dict(config))
        print("TOY CONCEPT PASSED")
        return 0
    sys.stderr.write(f"steerlab-server: unknown task {task!r}\n")
    return 64


def _load_judgments(path: str) -> tuple[list, str | None]:
    """``(rows, instructions_sha256)`` for ``experiment complete-judgment``:
    a JSON list, an object carrying a ``judgments`` list (the HTTP body
    shape — its optional ``instructionsSha256`` is the campaign's claim of
    which judging-instructions.md it judged under), or JSONL — anything
    else refuses loudly before a single pin is checked."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = [json.loads(line) for line in text.splitlines()
                    if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"judgments file {path!r} is neither JSON nor JSONL: {exc}") \
                from None
    instructions_sha = None
    if isinstance(data, dict):
        raw_sha = data.get("instructionsSha256")
        instructions_sha = str(raw_sha) if raw_sha else None
        data = data.get("judgments")
    if not isinstance(data, list) or not data \
            or not all(isinstance(row, dict) for row in data):
        raise ValueError(
            f"judgments file {path!r} must hold a non-empty list of judgment "
            "objects ([{packetID, judge, winner: A|B|tie, model, …}, …] — "
            "as a JSON list, a {\"judgments\": [...]} object, or JSONL)")
    return data, instructions_sha


def _experiment(args: list[str]):
    """The experiment lifecycle verbs.

    Ten of them are on the agent path (audit §2.1) and return a
    ``cli_envelope.CLIResult`` describing what they learned; the rest return an
    exit code as they always did. Authoring — ``create``/``attach``/``freeze``/
    ``duplicate`` and the pin/declare/set verbs — is Mac-authority and lives on
    the Swift CLI and the HTTP API, not here (§3.2).
    """
    from . import cli_envelope as cli_envelope_module, cli_payloads
    from .cli_envelope import CLIResult, advisory, next_action
    from .experiment import adjudication, lifecycle_gates, tasks
    from .experiment.manifest import Manifest
    from .experiment.paths import experiments_directory

    if not args:
        sys.stderr.write(_EXPERIMENT_VERB_LINE)
        return 64
    verb, rest = args[0], args[1:]
    # Answered BEFORE the positional guard below, so a Mac verb typed with no
    # name gets the redirect instead of a usage line for a verb that does not
    # exist here (gate-5 dry run #2, P3).
    mac_authority = cli_envelope_module.MAC_AUTHORITY_VERBS["experiment"]
    if verb in mac_authority:
        return _mac_authority_refusal(
            f"experiment {verb}", mac_authority[verb],
            note="Author and freeze there, then execute here — "
                 "steerlab-server experiment run|sweep|evaluate|analyze read "
                 "the frozen manifest this engine already holds")
    root = None
    if "--root" in rest:
        i = rest.index("--root")
        root = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]

    # Strict argv for the family's pass-through verbs (open-issues §16 repair
    # 3), in the same shape and at the same exit code the envelope layer uses
    # for its declared verbs: a flag the verb does not accept is a malformed
    # invocation, refused before any work, never a silently dropped token.
    unknown = _unknown_experiment_flag(verb, rest)
    if unknown is not None:
        accepted = " ".join(
            sorted(_EXPERIMENT_PASSTHROUGH_FLAGS[verb]))
        sys.stderr.write(
            f"steerlab-server experiment: experiment {verb} does not accept "
            f"{unknown}\n  experiment {verb} accepts: {accepted}\n")
        return 64

    if verb == "list":
        directory = experiments_directory(root)
        names = sorted(
            n for n in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, n))
            or n.endswith(".json")) if os.path.isdir(directory) else []
        if not names:
            print("no experiments")
        listed: list[dict] = []
        for name in names:
            try:
                m = Manifest.load(name.removesuffix(".json"), root)
                print(f"{m.name}  [{m.status}]  model={m.model_id}  "
                      f"concepts={'+'.join(c.name for c in m.concepts)}  "
                      f"conditions={len(m.conditions)}")
                listed.append({
                    "name": m.name, "status": m.status, "modelID": m.model_id,
                    # FULL revision and freeze hash: the human line elides
                    # them, a document must not.
                    "modelRevision": m.model_revision,
                    "concepts": [c.name for c in m.concepts],
                    "conditionCount": len(m.conditions),
                    "freezeHash": m.freeze_hash,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: unreadable ({exc})")
                listed.append({"name": name.removesuffix(".json"),
                               "unreadable": str(exc)})
        return CLIResult(
            message=("no experiments" if not listed
                     else f"{len(listed)} experiment(s)"),
            payload={"count": len(listed), "experiments": listed})

    if not rest:
        # A usage line is only honest for a verb this engine DISPATCHES;
        # printing one for an unrecognised verb asserted that the verb exists
        # (gate-5 dry run #2, P3). An unknown verb gets the roster instead.
        if verb in EXPERIMENT_VERBS:
            sys.stderr.write(f"usage: experiment {verb} <name>\n")
        else:
            sys.stderr.write(_EXPERIMENT_VERB_LINE)
        return 64
    name = rest[0]
    dtype = _flag(rest, "--dtype") or "auto"     # auto = bf16/fp16/fp32 by device
    device = _flag(rest, "--device")             # auto = CUDA → MPS (Apple GPU) → CPU

    if verb == "preflight-endpoints":
        return _preflight_endpoints(name, rest, root)
    if verb == "verify":
        manifest = Manifest.load(name, root)
        violations = manifest.verify(root)
        if not violations:
            print("OK — all pinned inputs verified")
            return CLIResult(
                message="OK — all pinned inputs verified",
                payload={"experiment": manifest.name,
                         "status": manifest.status,
                         "verified": True, "violations": [],
                         "experimentHash": manifest.content_hash()})
        for v in violations:
            print(f"VIOLATION: {v}")
        # Human mode keeps exit 1 — the audit schedules exactly one human-mode
        # migration at this rung (`data check`'s 2 → 65). In JSON mode this is
        # `pinDrift`, 65, with the drifted pins in `result.violations`.
        return CLIResult(
            state="refused", exit_code=1,
            code=lifecycle_gates.PIN_DRIFT, gate=lifecycle_gates.PIN_DRIFT,
            message=(f"{len(violations)} pinned input(s) of '{manifest.name}' "
                     "no longer match their hashes"),
            repair_action=(
                f"steerlab-server experiment verify {manifest.name} "
                "(names every drifted pin) ; then restore the named files, or "
                "author the manifest's replacement on the Mac "
                f"(steerlab-cli experiment duplicate {manifest.name} "
                f"{manifest.name}-v2) and re-pin"),
            payload={"experiment": manifest.name, "status": manifest.status,
                     "verified": False, "violations": list(violations)})
    if verb == "attach-artifact":
        # The ONE authoring verb on this engine, because it is the one whose
        # input lives HERE: a post-hoc derived vector artifact under runs/.
        # Everything it pins is read off the artifact's sidecar — the artifact
        # IS the recipe — so the only choices are which concept name it takes
        # and which concept's held-out data validates it.
        from .experiment import experiment_store
        artifact = _flag(rest, "--artifact")
        if len(rest) < 2 or not artifact:
            sys.stderr.write(
                "usage: experiment attach-artifact <name> <concept> "
                "--artifact <runs/<run>/<artifact>> [--source-concept C] "
                "[--eval-run <optvec-eval run>]\n"
                "  (the artifact path is EXTENSION-LESS: <path>.safetensors + "
                "<path>.json; --eval-run names the OptVec eval run whose "
                "eval.json certifies an optvec vector — surfaced at freeze)\n")
            return 64
        try:
            experiment_store.attach_artifact(
                name, rest[1], artifact,
                source_concept=_flag(rest, "--source-concept"),
                eval_run=_flag(rest, "--eval-run"), root=root)
        except experiment_store.ExperimentStoreError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            return 1
        violations = Manifest.load(name, root).verify(root)
        for v in violations:
            print(f"VIOLATION: {v}")
        print(f"attached '{rest[1]}' from pinned artifact {artifact}")
        return 1 if violations else 0
    if verb == "extract":
        run_directory = tasks.extract(name, root, dtype, device)
        return CLIResult(
            message=f"extracted vectors for '{name}'", changed=True,
            payload={"experiment": name, "runDirectory": run_directory},
            next_action=next_action(f"experiment validate {name}"))
    if verb == "validate":
        run_directory = tasks.validate(name, root, dtype, device)
        # THE QUALITY NUMBERS and THE VACUITY LEDGER (punch list #1, P4 and
        # the 2026-08-17 firewall repair). A chance-level probe — accuracy
        # 0.50 — froze and ran with no machine signal at all, because the
        # envelope reported only that validate had happened. Both now ride the
        # document, in the same keys the Swift twin uses.
        payload = cli_payloads.validation_payload(name, run_directory)
        advisories = []
        for concept in payload.get("vacuousConcepts", []):
            advisories.append(advisory(
                "vacuousValidation",
                f"no held-out probe was scored for '{concept}' — this "
                "evidence will NOT satisfy freeze's validateEvidence gate"))
        scores = payload.get("validation", [])
        for score in scores:
            if score.get("atOrBelowChance"):
                advisories.append(advisory(
                    "probeAtChanceFloor",
                    cli_payloads.probe_advisory_detail(score)))
        vacuous = payload.get("vacuousConcepts") or []
        summary = cli_payloads.validation_summary(scores)
        return CLIResult(
            message=(
                f"validated '{name}' → {os.path.basename(run_directory)}"
                + (f" — {summary}" if summary else "")
                if not vacuous else
                f"VACUOUS validation for '{name}' — no held-out probe for "
                + ", ".join(vacuous)),
            changed=True, payload=payload, advisories=advisories,
            state="okWithAdvisories" if advisories else "ready",
            # `nextAction.verb` is a COMMAND — the contract says "the verb to
            # run next", and an agent executes it verbatim. The Mac-authority
            # gloss and the "author these files first" precondition both moved
            # to `detail`, which exists for exactly that (gate-5 dry run #2,
            # P3: the verb string carried an English parenthetical, so
            # nothing in it was runnable as written).
            next_action=(
                next_action(
                    f"experiment freeze {name}",
                    detail="run it on the Mac (steerlab-cli) — authoring is "
                           "Mac-authority and this engine has no freeze verb")
                if not vacuous else
                next_action(
                    f"experiment validate {name}", requires_human=True,
                    detail="author the named validation.jsonl sets first — "
                           "re-validating without them repeats this result")))
    if verb == "sweep":
        manifest = Manifest.load(name, root)
        # The defaulted-criterion advisory is computed BEFORE the sweep runs,
        # from the manifest alone, so an agent reading `--json` sees it
        # whether or not the grid completed (punch list #1, P3).
        advisories = []
        defaulted = _defaulted_selection_advisory(manifest, root)
        if defaulted:
            advisories.append(advisory("sweepSelectionDefaulted", defaulted))
        run_directory = tasks.sweep(name, root, dtype, device)
        from .experiment import sweep_selection
        criterion = sweep_selection.resolve_selection(
            (manifest.raw.get("sweep") or {}).get("selection")).metric
        payload = cli_payloads.sweep_payload(
            name, run_directory, manifest_status=manifest.status,
            criterion=criterion)
        if payload["recommendationsOnly"]:
            advisories.append(advisory(
                "sweepRecommendationsOnly",
                f"'{name}' is {manifest.status}, so the sweep wrote no "
                "<concept>-recommended condition — the recommendations live "
                f"in {os.path.basename(run_directory)}/recommendations.json, "
                "and promote reads them from there"))
        selected = sum(1 for entry in payload["recommendations"]
                       if entry.get("selected"))
        return CLIResult(
            message=(f"swept '{name}' → {selected} of "
                     f"{len(payload['recommendations'])} concept(s) selected a "
                     f"cell on {criterion} in "
                     f"{os.path.basename(run_directory)}"),
            changed=True, payload=payload, advisories=advisories,
            state="okWithAdvisories" if advisories else "ready",
            next_action=next_action(
                f"experiment promote {name} <concept>"))
    if verb == "run":
        # Headless checkpoint/resume (WS2): SIGUSR1/SIGTERM park the run and
        # exit 85; --resume <run-dir> continues a checkpointed directory.
        #
        # `--shard k/K` (open-issues §16, 2026-08-19): the same multi-GPU
        # partition `bundle execute --verb run --shard k/K` submits, reachable
        # by hand. Without it a checkpointed shard partial had NO CLI path back
        # to life — the resume gate's own repair sentence ("resume it with the
        # same --shard k/K") named a flag this arm did not accept, and four GPU
        # allocations died at that gate after staging and loading 51 GiB.
        from .experiment import resume as resume_mod
        from .experiment import sharding as sharding_mod
        shard_text = _flag(rest, "--shard")
        try:
            shard_spec = (sharding_mod.parse_shard(shard_text)
                          if shard_text is not None else None)
        except sharding_mod.ShardError as exc:
            # A malformed flag VALUE stays the verb's own refusal (the
            # envelope layer owns undeclared FLAGS), and a malformed
            # invocation is 64 in both modes.
            sys.stderr.write(
                f"usage: experiment run <name> [--shard k/K]: {exc}\n")
            return 64
        flag = resume_mod.CheckpointFlag().install()
        try:
            run_directory = tasks.run(
                name, _flag(rest, "--prompts"), root, dtype, device,
                checkpoint=flag, run_directory=_flag(rest, "--resume"),
                shard=shard_spec)
        except resume_mod.CheckpointRequested as exc:
            sys.stderr.write(f"experiment run checkpointed: {exc}\n")
            # 85 stays 85 in human mode (the resume contract every Slurm
            # wrapper reads); in JSON mode it is `pending` — valid
            # asynchronous work, repeating with --resume is correct.
            # The resume command a sharded run needs carries its shard: the
            # gate refuses a partial resumed without it (or under another
            # range), so the printed next action must be executable as
            # written (open-issues §16).
            shard_suffix = ("" if shard_spec is None
                            else f" --shard {shard_spec.label}")
            return CLIResult(
                state="pending", exit_code=resume_mod.CHECKPOINT_EXIT_CODE,
                message=f"run of '{name}' checkpointed: {exc}",
                payload={"experiment": name, "checkpointed": True,
                         **({} if shard_spec is None
                            else {"shard": {"index": shard_spec.index,
                                            "count": shard_spec.count}})},
                next_action=next_action(
                    f"experiment run {name} --resume <run-dir>"
                    + shard_suffix))
        advisories = _implicit_case_family_advisories(name, root)
        return CLIResult(
            message=f"ran '{name}' → {os.path.basename(run_directory)}",
            changed=True,
            payload={"experiment": name, "runDirectory": run_directory,
                     **({} if shard_spec is None
                        else {"shard": {"index": shard_spec.index,
                                        "count": shard_spec.count}})},
            advisories=advisories,
            state="okWithAdvisories" if advisories else "ready",
            # A shard produced a PARTIAL: analyzing it would analyze a
            # fraction of the matrix. The next action is the rest of the
            # fan-out plus the merge, which the submitting server performs.
            next_action=(next_action(f"experiment analyze {name}")
                         if shard_spec is None else
                         next_action(
                             f"experiment run {name} --shard "
                             f"<k>/{shard_spec.count}", requires_human=True,
                             detail=("this directory holds shard "
                                     f"{shard_spec.label} only — run every "
                                     "remaining shard; the submitting server "
                                     "merges the partials once all "
                                     f"{shard_spec.count} succeed"))))
    if verb == "pipeline":
        # The chain runner (stage 3): declared stages with one model load,
        # gate-aborted between stages. Same checkpoint/exit-85 contract as
        # run — the run STAGE parks record-level; --resume <pipeline-dir>
        # reopens the stage ledger and skips completed stages. A gate abort
        # is a recorded determination (pipeline-abort.json), exit 0.
        from .experiment import resume as resume_mod
        # `--shard` is ACCEPTED here and refused with a reason, exactly as
        # `bundle execute --verb pipeline --shard k/K` refuses it
        # (bundles._execute_run_bundle_inner): only `run` has the
        # per-record-independent record set a partition is defined over, and a
        # chain's other stages (extract/validate/sweep/promote/evaluate/
        # analyze) would either duplicate work on every shard or measure a
        # fraction of the matrix. Silently dropping the flag is the trap
        # open-issues §16 is about — a correct-looking command line the CLI
        # half-executes — so it refuses at 64 instead.
        if _flag(rest, "--shard") is not None or "--shard" in rest:
            sys.stderr.write(
                "--shard applies to the 'run' verb only — other stages of a "
                "chain have no independent per-record record set to "
                "partition.\n"
                f"  one shard by hand:  steerlab-server experiment run {name} "
                "--shard k/K\n"
                f"  the whole chain sharded:  steerlab-server study submit "
                f"{name} --verb pipeline --parallel K  (the run stage fans "
                "out; the merge seeds a continuation for the remaining "
                "stages)\n")
            return 64
        flag = resume_mod.CheckpointFlag().install()
        try:
            tasks.pipeline(name, root, dtype, device, checkpoint=flag,
                           pipeline_run_directory=_flag(rest, "--resume"))
        except resume_mod.CheckpointRequested as exc:
            sys.stderr.write(f"experiment pipeline checkpointed: {exc}\n")
            return resume_mod.CHECKPOINT_EXIT_CODE
        return 0
    if verb == "evaluate":
        # --resume-from <partial-run-id>: complete a FAILED evaluation by
        # judging only the cells it never decided, reusing the verdicts it
        # already produced (2026-07-24). Every pin of the partial run is
        # verified first; a differing rubric, epoch, source run, or judge
        # configuration refuses rather than merging two evaluations.
        evaluation = tasks.evaluate(
            name, root, _flag(rest, "--source"),
            allow_unverified_epoch=("--allow-unverified-epoch" in rest),
            resume_from=_flag(rest, "--resume-from"))
        payload = {"experiment": name, "evaluationDirectory": evaluation}
        source = cli_payloads.read_text(
            os.path.join(evaluation, "source-run.txt"))
        if source:
            payload["sourceRun"] = source
        return CLIResult(
            message=f"evaluated '{name}' → {os.path.basename(evaluation)}",
            changed=True, payload=payload)
    if verb == "judge-worker":
        # One judge-model worker of the post-generation judge fan-out
        # (2026-07-23): loads ITS judge model (pinned revision/dtype),
        # judges every packet of the awaiting evaluate run for every pinned
        # local judge resolving to that model, and writes one hash-pinned
        # judgment artifact. The controller merges artifacts through
        # complete_evaluate_judgment. --record writes the standard child
        # record for the Slurm reconciler.
        awaiting = _flag(rest, "--awaiting-run")
        model = _flag(rest, "--model")
        out = _flag(rest, "--out")
        record = _flag(rest, "--record")
        if not (awaiting and model and out):
            sys.stderr.write(
                "usage: experiment judge-worker <name> --awaiting-run <run> "
                "--model <id> --out <artifact.json> [--revision R] "
                "[--dtype D] [--device DEV] [--record <record.json>]\n")
            return 64
        import time as _time
        started = _time.time()
        try:
            result = tasks.judge_worker(
                name, awaiting, model, revision=_flag(rest, "--revision"),
                dtype=_flag(rest, "--dtype") or "auto",
                device=device, out_path=out, root=root)
        except Exception as exc:  # noqa: BLE001 - recorded for reconciliation
            if record:
                from .experiment.bundles import write_child_record
                write_child_record(
                    record, kind="judge-worker", status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=_time.time() - started)
            sys.stderr.write(f"ERROR: {exc}\n")
            return 1
        result.pop("rows", None)  # the artifact file carries the rows
        if record:
            from .experiment.bundles import write_child_record
            write_child_record(
                record, kind="judge-worker", status="succeeded",
                result=result, elapsed_seconds=_time.time() - started,
                record_count=result.get("judgments"))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if verb == "complete-judgment":
        # Phase 2 of a deferred evaluate, headless (2026-08-10): hand a
        # judging client's completed judgments to
        # tasks.complete_evaluate_judgment — the same intake the HTTP route
        # runs, which verifies every emission pin (packets/map/source
        # hashes, rubric + structured prompt, experiment epoch, judge
        # panel, full coverage, winner-consistent verdicts) and aggregates
        # the inline judgments.jsonl + judge-report.json shapes. CPU-only;
        # idempotent (a completed run prints its existing directory).
        awaiting = _flag(rest, "--awaiting-run")
        judgments_path = _flag(rest, "--judgments")
        if not (awaiting and judgments_path):
            sys.stderr.write(
                "usage: experiment complete-judgment <name> "
                "--awaiting-run <run-dir-or-basename> --judgments <file>\n"
                "  (<file>: a JSON list, a {\"judgments\": [...], "
                "\"instructionsSha256\": …} object, or JSONL of\n"
                "   {packetID, judge, winner: A|B|tie, model, "
                "annotatorModel?, …} rows)\n")
            return 64
        evaluate_run = os.path.basename(os.path.normpath(awaiting))
        try:
            judgments, instructions_sha = _load_judgments(judgments_path)
            run_directory = tasks.complete_evaluate_judgment(
                name, evaluate_run, judgments, root=root,
                instructions_sha256=instructions_sha)
        except (OSError, ValueError, RuntimeError) as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            return 1
        print(run_directory)
        return 0
    if verb == "analyze":
        # Statistics + reporting over a prior run (paired effect sizes with
        # bootstrap CIs, FDR/Holm correction, alien residuals, promotion
        # decisions) — the same tasks.analyze the HTTP route runs, pure CPU.
        # [--source <run-dir>] picks the source run (default: newest);
        # --allow-unverified-epoch accepts legacy runs with no experiment-hash
        # stamp (the output is then stamped epochUnverified).
        # --adjudicated-endpoint <file> substitutes an external extraction
        # campaign's verified per-record endpoint values (open-issues §10).
        source = _flag(rest, "--source")
        adjudicated = _flag(rest, "--adjudicated-endpoint")
        if adjudicated and not source:
            # An adjudication is evidence about ONE run; letting it default
            # to the newest one invites joining it against the wrong run,
            # and every verification below would still pass on a run that
            # happens to hash correctly. Usage, not a refusal.
            sys.stderr.write(
                "usage: experiment analyze <name> --source <run-dir> "
                "--adjudicated-endpoint <file>\n"
                "  (--adjudicated-endpoint REQUIRES --source: one file "
                "adjudicates one run, and\n"
                "   defaulting to the newest run would join it against the "
                "wrong generations)\n")
            return 64
        try:
            run_directory = tasks.analyze(
                name, root, source,
                allow_unverified_epoch=("--allow-unverified-epoch" in rest),
                adjudicated_endpoint=adjudicated)
        except adjudication.AdjudicationError as exc:
            # The intake's ladder speaks in repairable sentences; a
            # traceback would bury them. Same shape as complete-judgment's
            # refusals — and JSON mode re-raises so the document carries it.
            sys.stderr.write(f"ERROR: {exc}\n")
            if _JSON_MODE_ACTIVE:
                raise
            return 1
        payload = cli_payloads.analysis_payload(run_directory)
        payload["experiment"] = name
        payload["runDirectory"] = run_directory
        entries = payload.get("effectSizeCount", 0)
        # The endpoint rescue's grammar may have been chosen by the deprecated
        # caseFamily trigger (tasks.analyze). Same advisory, same code, same
        # sentence as the run path — an agent matches on one string.
        advisories = _implicit_case_family_advisories(name, root)
        # An analysis with zero effect-size entries "succeeded" and measured
        # nothing (dry run #0's P0-2). WHY it measured nothing is read off
        # the source run rather than assumed (dry run #2): the fixed sentence
        # claimed "no non-baseline condition" on a run that had two of them
        # and 24 records, which is the one thing the reader must not be told.
        if entries == 0:
            source_run = payload.get("sourceRun")
            if isinstance(source_run, str) and source_run:
                # Sibling of the analyze directory: both live under the
                # workspace's runs/, and `root` is None here unless --root was
                # passed (it is resolved inside tasks).
                count, conditions = cli_payloads.source_run_records(
                    os.path.join(os.path.dirname(run_directory), source_run))
                detail = cli_payloads.empty_analysis_detail(
                    source_run, count, conditions)
            else:
                detail = cli_payloads.EMPTY_ANALYSIS_NO_CONTRAST
            advisories.append(advisory("emptyAnalysis", detail))
        # Entries that are ALL exactly zero are a different fact: the pairing
        # worked and every intervention moved nothing, which is either a real
        # null or an arm declared and never actually injected (P14).
        elif cli_payloads.all_effect_sizes_are_zero(run_directory):
            advisories.append(advisory(
                "allEffectSizesZero",
                f"all {entries} effect size(s) are exactly 0.0 — the pairing "
                "worked and no condition moved any metric. Check that the arms "
                "actually injected: compare generations.jsonl across "
                f"conditions in {os.path.basename(run_directory)}, and confirm "
                "the declared studyType does not inert them (steerlab-server "
                f"experiment verify {name})"))
        return CLIResult(
            message=(f"analyzed '{name}' → {entries} effect-size "
                     f"entr{'y' if entries == 1 else 'ies'} in "
                     f"{os.path.basename(run_directory)}"),
            changed=True, payload=payload, advisories=advisories,
            state="okWithAdvisories" if advisories else "ready")
    if verb == "rescore-style":
        # Post-hoc reasoning-style scoring of a prior run through the pinned
        # taxonomy — NEW files (reasoning-style.csv + reasoning-style.json)
        # in a fresh run directory; the source run is never mutated.
        # Epoch-guarded like analyze. Pure CPU.
        tasks.rescore_style(
            name, root, _flag(rest, "--source"),
            allow_unverified_epoch=("--allow-unverified-epoch" in rest))
        return 0
    if verb == "promote":
        # Headless Promote: mint an agent (variant artifact) from the sweep-
        # selected cell. --cell L:ALPHA is the loud manual override.
        from .experiment import promote as promote_mod
        if len(rest) < 2:
            sys.stderr.write(
                "usage: experiment promote <name> <concept> "
                "[--agent-name <name>] [--cell <layer>:<alpha> --reason <why>]\n"
                "       [--sweep-run <run> [--expect-cell <layer>:<alpha>]\n"
                "        [--expect-artifact <path>] [--expect-epoch <hash>]]\n"
                "       [--qualification <sae-feature-qualification.json>]\n")
            return 64

        def _parse_cell(raw: str | None, flag: str):
            if raw is None:
                return None, 0
            try:
                layer_text, alpha_text = raw.split(":", 1)
                return (int(layer_text), float(alpha_text)), 0
            except ValueError:
                sys.stderr.write(f"{flag} must be <layer>:<alpha>, e.g. 17:0.4\n")
                return None, 64

        cell, code = _parse_cell(_flag(rest, "--cell"), "--cell")
        if code:
            return code
        expect_cell, code = _parse_cell(
            _flag(rest, "--expect-cell"), "--expect-cell")
        if code:
            return code
        # The PINNED contract (B2): --sweep-run alone already removes the
        # ambient "newest run" lookup; the --expect-* flags additionally
        # verify that the plan is still current.
        pins = None
        sweep_run = _flag(rest, "--sweep-run")
        if sweep_run is not None:
            pins = promote_mod.PromotionPins(
                sweep_run=sweep_run,
                experiment_hash=_flag(rest, "--expect-epoch"),
                winning_cell=expect_cell,
                vector_artifact_id=_flag(rest, "--expect-artifact"),
                vector_artifact_hash=_flag(rest, "--expect-artifact-hash"))
        elif expect_cell is not None:
            sys.stderr.write(
                "--expect-cell requires --sweep-run (there is nothing to "
                "check the expectation against otherwise)\n")
            return 64
        try:
            record = promote_mod.promote(
                name, rest[1],
                agent_name=_flag(rest, "--agent-name"),
                cell=cell,
                override_reason=_flag(rest, "--reason"),
                root=root, pins=pins,
                # CITED evidence (SAE proposal r2 §6): the birth certificate
                # gains a qualification block; seating is unchanged.
                qualification=_flag(rest, "--qualification"))
        except (promote_mod.PromoteError, FileNotFoundError) as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            # The typed refusal reaches the envelope through the raise, not
            # through this catch: human mode keeps its `ERROR:` line and its
            # exit 1, and JSON mode needs the gate id, so the exception is
            # re-raised for `_exception_envelope` to classify.
            if _JSON_MODE_ACTIVE:
                raise
            return 1
        variant = record.get("variant") or {}
        return CLIResult(
            message=(f"promoted '{rest[1]}' → agent "
                     f"'{variant.get('name', '?')}'"),
            changed=True,
            payload={
                "experiment": name, "concept": rest[1],
                "agentName": variant.get("name"),
                "artifact": record.get("path"),
                "promotedBy": ((variant.get("promotion") or {})
                               .get("promotedBy")
                               or ("manualOverride" if cell else "criterion")),
                **({"overrideReason": _flag(rest, "--reason")}
                   if cell and _flag(rest, "--reason") else {}),
            },
            next_action=next_action(
                f"experiment confirm {name} --agent {variant.get('name', '?')}"))

    if verb == "confirm":
        # Confirmation stage: declare a perturbation policy around a promoted
        # agent's anchor cell — expands mechanically into ordinary hashed
        # conditions on the DRAFT manifest (confirmation.attach_perturbations).
        from .experiment import confirmation
        agent = _flag(rest, "--agent")
        if not agent:
            sys.stderr.write(
                "usage: experiment confirm <name> --agent <variant-name-or-path> "
                "[--deltas 0.2,0.5] [--no-control]  (default deltas: 0.2)\n")
            return 64
        deltas = (0.2,)
        raw_deltas = _flag(rest, "--deltas")
        if raw_deltas is not None:
            try:
                deltas = tuple(float(x) for x in raw_deltas.split(",") if x.strip())
            except ValueError:
                sys.stderr.write(
                    "--deltas must be comma-separated numbers, e.g. 0.2,0.5\n")
                return 64
        try:
            updated = confirmation.attach_perturbations(
                name, agent, deltas=deltas,
                include_control="--no-control" not in rest, root=root)
        except (confirmation.ConfirmationError, FileNotFoundError) as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            if _JSON_MODE_ACTIVE:
                raise       # the gate id belongs in the document
            return 1
        conditions = [c.get("name") for c in (updated.get("conditions") or [])]
        return CLIResult(
            message=(f"attached a confirmation policy around '{agent}' — "
                     f"{len(conditions)} condition(s)"),
            changed=True,
            payload={"experiment": name, "agent": agent,
                     "deltas": list(deltas),
                     "includeControl": "--no-control" not in rest,
                     "conditionCount": len(conditions),
                     "conditions": conditions},
            # Same rule as `validate`'s: the precondition is `detail`, not
            # part of the command (gate-5 dry run #2, P3).
            next_action=next_action(
                f"experiment run {name}",
                detail="freeze it on the Mac first (steerlab-cli experiment "
                       f"freeze {name}) — authoring is Mac-authority"))

    sys.stderr.write(
        _EXPERIMENT_VERB_LINE
        + "(attach-artifact <name> <concept> --artifact <path> "
        "[--source-concept C] pins an EXISTING\n"
        " vector artifact as a concept — the post-hoc-derived-direction path;\n"
        " evaluate/analyze/rescore-style read a prior run: [--source <run-dir>] "
        "[--allow-unverified-epoch];\n"
        " evaluate also takes [--resume-from <partial-run-id>] to finish a "
        "failed evaluation by\n"
        " judging only its undecided cells, reusing the verdicts it already "
        "produced;\n"
        " complete-judgment <name> --awaiting-run <run> --judgments <file> "
        "finishes a DEFERRED\n"
        " evaluate from a judging client's completed judgment file;\n"
        " create/attach/freeze/duplicate author the firewall manifest and are\n"
        " driven from the Swift app / web UI — see CHANGES-TO-SWIFT-SIDE.md)\n")
    return 64


_VECTORS_USAGE = (
    "usage: steerlab-server vectors compare <a.safetensors> "
    "<b.safetensors> [--json OUT] [--threshold T]\n"
    "       steerlab-server vectors backfill-norms <runDir/name> "
    "[--corpus prompts/neutral/corpus.jsonl] [--output-name N] "
    "[--redenominate] [--model <id>] [--revision R] [--device D] "
    "[--dtype T]\n")


def _parity_could_not_compare_repair(*, incomparable: bool, path_a: str,
                                     path_b: str) -> str:
    """The repair for ``vectors compare``'s third outcome, per class. Both
    forms name BOTH operand paths and where extraction writes an artifact,
    because the commonest instance of this refusal is a caller who does not
    know an artifact is two files. Swift twin:
    ``ExperimentCLIRunner.parityCouldNotCompareRepair``."""
    shape = ("a vector artifact is <runDir>/<name>.safetensors PLUS its "
             "<runDir>/<name>.json sidecar, both written by `steerlab-server "
             "experiment extract <name>`")
    if incomparable:
        return ("both artifacts were read and are not comparable at all — "
                "compare two artifacts extracted from the SAME model "
                f"('{path_a}' vs '{path_b}'); {shape}")
    return (f"check both operand paths and their sidecars — '{path_a}' and "
            f"'{path_b}': {shape}")


def _vectors(args: list[str]):
    """``compare`` — the cross-engine parity harness (WS7.3; Swift twin:
    ``steerlab-cli vectors compare``); ``backfill-norms`` — residual-norm
    backfill for artifacts that predate norm-unit alphas (Swift twin:
    ``steerlab-cli vectors backfill-norms``; API twin:
    ``POST /api/vectors/backfill-norms``)."""
    if args and args[0] == "backfill-norms":
        return _vectors_backfill_norms(args[1:])
    usage = _VECTORS_USAGE
    if len(args) < 3 or args[0] != "compare":
        sys.stderr.write(usage)
        return 64
    from .cli_envelope import CLIResult
    from .experiment import lifecycle_gates
    from .steering import vector_parity
    raw_threshold = _flag(args, "--threshold")
    try:
        threshold = (float(raw_threshold) if raw_threshold is not None
                     else vector_parity.DEFAULT_THRESHOLD)
    except ValueError:
        sys.stderr.write(f"vectors compare: bad --threshold {raw_threshold!r}\n")
        return 64
    try:
        report = vector_parity.compare_paths(args[1], args[2], threshold=threshold)
    except (OSError, ValueError) as exc:
        # THE THIRD OUTCOME: could-not-compare. Human mode has exited 2 here
        # since the verb existed and keeps doing so; what changes (2026-08-18)
        # is that the ENVELOPE stops collapsing it into the same `refused`/65
        # a real parity divergence answers with. A CI script's whole job at
        # this verb is telling those apart, and `state_for_legacy_exit(2)` was
        # making them identical documents.
        #
        # OSError = the artifact or its sidecar is missing/unreadable;
        # ValueError = both read cleanly and are not comparable at all (a
        # hidden-size mismatch, i.e. two different models). Both are
        # `notFound`/66 — the artifacts could not be turned into a comparison —
        # and they differ only in the repair, because a repair that names the
        # wrong thing sends a caller in a circle. Layer-count mismatch is NOT
        # here: that is a comparison that ran (Swift twin: the same split, on
        # `ParityError.Kind`).
        sys.stderr.write(f"vectors compare: {exc}\n")
        return CLIResult(
            state="notFound", exit_code=2, code="notFound",
            message=f"vectors compare could not compare the artifacts: {exc}",
            repair_action=_parity_could_not_compare_repair(
                incomparable=isinstance(exc, ValueError),
                path_a=args[1], path_b=args[2]),
            payload={"operandPaths": [args[1], args[2]],
                     "threshold": threshold})
    text = report.json_text()
    out = _flag(args, "--json")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text)
        sys.stderr.write(f"wrote {out}\n")
    # The parity report is the verb's product: on stdout for a human (where a
    # cross-engine `diff` against the Swift twin's bytes is the whole point of
    # its formatter), and inside `result.report` — unchanged, key for key — for
    # a machine. In `--json` mode this write lands on stderr with every other
    # diagnostic, so stdout carries exactly one document.
    sys.stdout.write(text)
    payload = {
        "threshold": threshold,
        "passed": report.passed,
        "comparedLayerCount": report.compared_layer_count,
        "report": json.loads(text),
    }
    if report.min_cosine is not None:
        payload["minCosine"] = report.min_cosine
    if not report.passed:
        line = (f"vectors compare: FAIL — min cosine "
                f"{report.min_cosine!r} < threshold {threshold!r} over "
                f"{report.compared_layer_count} compared layer(s)")
        sys.stderr.write(line + "\n")
        return CLIResult(
            state="refused", exit_code=1,
            code=lifecycle_gates.PARITY_THRESHOLD,
            gate=lifecycle_gates.PARITY_THRESHOLD,
            message=line,
            repair_action=(
                "steerlab-server experiment extract <name> on the substrate "
                "that will RUN the study (activations do not transfer across "
                "engines), or re-run this compare with an explicitly lowered "
                "--threshold"),
            payload=payload)
    return CLIResult(
        message=(f"parity OK — min cosine {report.min_cosine!r} ≥ threshold "
                 f"{threshold!r} over {report.compared_layer_count} compared "
                 "layer(s)"),
        payload=payload)


def _vectors_backfill_norms(args: list[str]) -> int:
    """``vectors backfill-norms <runDir/name> …`` — measure per-layer residual
    norms for an existing artifact (legacy / SAE import / reader-derived /
    optvec-trained) on the pinned neutral corpus and write a NEW artifact into
    a fresh run directory; the source is never modified.

    This is also the OPT-IN MIGRATION onto the current residual-norm
    denominator convention: the new artifact is stamped
    ``residualNormConvention: "wholeCorpusMean-v1"``. Artifacts without that
    stamp are LEGACY and are never rewritten, recomputed, or warned about —
    running this verb is how a researcher chooses to move one forward.

    Parameter-for-parameter the CLI form of ``POST /api/vectors/backfill-norms``
    (``vectorID``/``neutralCorpusPath``/``outputName``/``redenominate``/
    ``modelID``/``revision``) with the Swift twin's path resolution: the
    reference is a base path with NO extension; absolute stays absolute,
    ``runs/…`` resolves against the workspace root, anything else resolves
    under ``runs/``. ``--model``/``--revision`` are loading conveniences —
    they default to the artifact's own sidecar pins, and the hard
    sidecar-vs-loaded-model guard applies regardless (norms are a per-model
    measurement). Result JSON is key-identical to the API route's. Exit
    codes: 0 = backfilled; 2 = refused (missing artifact/corpus, norms
    already present without ``--redenominate``, wrong model, load or
    measurement failure); 64 = usage."""
    if not args or args[0].startswith("--"):
        sys.stderr.write(_VECTORS_USAGE)
        return 64
    from .experiment import paths, run_config

    reference = args[0]
    # Swift-twin path resolution (docs/CLI-REFERENCE.md §3.7): absolute stays
    # absolute, runs/… joins the workspace root, a bare <run>/<name> lands
    # under runs/.
    if os.path.isabs(reference):
        base = reference
    elif reference.startswith("runs/") or reference.startswith("runs" + os.sep):
        base = os.path.join(paths.project_root(), reference)
    else:
        base = os.path.join(paths.runs_directory(), reference)
    vector_dir, name = os.path.split(base)
    sidecar_path = base + ".json"
    if not name or not (os.path.isfile(sidecar_path)
                        and os.path.isfile(base + ".safetensors")):
        sys.stderr.write(
            f"vectors backfill-norms: no vector artifact at {base!r} — pass "
            "the artifact base path <runDir>/<name> (no extension)\n")
        return 2
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"vectors backfill-norms: unreadable sidecar: {exc}\n")
        return 2
    if "modelID" not in sidecar or "layerCount" not in sidecar:
        sys.stderr.write(
            f"vectors backfill-norms: {reference!r} is not a steering-vector "
            "artifact\n")
        return 2

    # Cheap synchronous refusals BEFORE paying a model load. The module
    # re-checks all of these authoritatively.
    redenominate = "--redenominate" in args
    if sidecar.get("residualNormPerLayer") and not redenominate:
        sys.stderr.write(
            "vectors backfill-norms: artifact already has residual norms "
            f"(source: {sidecar.get('residualNormSource') or 'unrecorded'}) — "
            "backfill never overwrites; pass --redenominate to write a NEW "
            "neutral-corpus-denominated artifact\n")
        return 2

    output_name = _flag(args, "--output-name") or name
    if not output_name or "/" in output_name or os.sep in output_name \
            or output_name in (".", ".."):
        sys.stderr.write(
            f"vectors backfill-norms: --output-name {output_name!r} must be "
            "a plain file-name component\n")
        return 64

    corpus_flag = _flag(args, "--corpus")
    corpus_path = paths.resolve(corpus_flag) if corpus_flag \
        else paths.neutral_corpus_path()
    if not os.path.isfile(corpus_path):
        sys.stderr.write(
            f"vectors backfill-norms: neutral corpus not found: "
            f"{corpus_path}\n")
        return 2

    model_id = _flag(args, "--model") or str(sidecar["modelID"])
    if model_id != str(sidecar["modelID"]):
        sys.stderr.write(
            f"vectors backfill-norms: vectors were extracted on model "
            f"{sidecar['modelID']!r}, not {model_id!r} — a residual-norm "
            "table is a per-model measurement\n")
        return 2
    revision = _flag(args, "--revision")
    if revision is None and sidecar.get("revision"):
        revision = str(sidecar["revision"])

    from .steering import model_loader, norm_backfill
    try:
        device = model_loader.resolve_device(_flag(args, "--device"))
        print(f"loading {model_id}"
              + (f"@{revision}" if revision else "")
              + f" on {device}", file=sys.stderr, flush=True)
        model = model_loader.load(model_id, revision,
                                  dtype=_flag(args, "--dtype"), device=device)
    except (OSError, ValueError, model_loader.ModelLoadError) as exc:
        sys.stderr.write(f"vectors backfill-norms: {exc}\n")
        return 2

    run_dir = paths.make_unique_run_directory(f"backfill-norms-{output_name}")
    run_config.write_run_config(run_dir, "norm-backfill", model_id=model_id,
                                revision=revision,
                                dtype=getattr(model, "dtype", None))
    try:
        result = norm_backfill.backfill_norms(
            model, vector_dir, name, corpus_path, run_dir,
            output_name=output_name, redenominate=redenominate)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"vectors backfill-norms: {exc}\n")
        return 2
    print(json.dumps({"runDirectory": result.run_directory,
                      "artifact": result.artifact_id,
                      "residualNormSource": result.residual_norm_source,
                      "layerCount": result.layer_count},
                     indent=2, sort_keys=True))
    print(f"source artifact untouched: {base}", file=sys.stderr)
    return 0


_SITE_USAGE = ("usage: steerlab-server site qualify [--json OUT] "
               "[--skip-model-fixtures]\n"
               "       steerlab-server site node-scratch-wrapper "
               "[--metadata-root DIR] [--print]\n")

#: Non-gate machine code for a qualification that found a failing check. NOT a
#: :mod:`lifecycle_gates` id: that vocabulary describes a STUDY's state, and
#: "this node does not reproduce a committed contract" describes the NODE. It
#: rides in ``error.code`` beside the other non-gate codes (``notFound``,
#: ``usage``, ``macAuthorityVerb``), with no ``error.gate`` — which is what
#: tells an agent not to switch on it as a gate.
SITE_QUALIFY_FAILED = "siteQualifyFailed"


def _site_node_scratch_wrapper(args: list[str]) -> int:
    """``site node-scratch-wrapper`` — render THE canonical ad-hoc sbatch.

    The one sanctioned starting point for a job this engine does not render
    (ledger 2026-08-23). It carries the site's node-local scratch gres request
    and the cleanup trap already armed, from the SAME definition
    (:mod:`node_scratch`) the study renderer uses — so there is one answer to
    "how does this site clean up", not a rendered one and a copied one that
    drift. The payload command is the only variable: ``sbatch <wrapper> <your
    command>``.

    It lands beside ``controller-job.sbatch`` in the metadata root, for the
    same reason that one does: a per-node artifact an operator submits by
    hand. Re-rendering is idempotent — identical bytes are not rewritten — so
    a standing ritual can call it every time.
    """
    from . import node_scratch

    if _flag(args, "--metadata-root") is None and "--metadata-root" in args:
        sys.stderr.write("--metadata-root requires a directory\n")
        return 64
    metadata_root = _flag(args, "--metadata-root")
    try:
        if "--print" in args:
            sys.stdout.write(node_scratch.render_wrapper())
            return 0
        result = node_scratch.write_wrapper(metadata_root=metadata_root)
    except OSError as exc:
        sys.stderr.write(f"site node-scratch-wrapper: {exc}\n")
        return 2
    sys.stderr.write(
        ("wrote " if result["written"] else "already current: ")
        + result["path"] + "\n"
        + f"  submit an ad-hoc job with:  sbatch {result['path']} "
          "<your command>\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _site(args: list[str]):
    """``site qualify`` — run the committed fixtures on THIS node and print a
    structural-parity report (WP6 R1; release gate 7).

    Report handling follows ``vectors compare``: the report document itself is
    the verb's product, so it goes to stdout for a human (two nodes' reports
    diff directly) and rides unchanged in ``result.report`` for a machine; in
    ``--json`` mode that stdout write lands on stderr with every other
    diagnostic, so exactly one document reaches stdout. ``--json OUT`` keeps
    the historical file spelling.

    The verdict rule: any failing check is ``failed`` (70) and names the
    failing ids; warnings alone are ``okWithAdvisories`` (0) with one advisory
    per warning; skips never change the verdict but are always counted in the
    summary line, so a report that verified almost nothing cannot read as a
    pass.
    """
    from . import site_qualify
    from .cli_envelope import CLIResult, advisory

    if args and args[0] == "node-scratch-wrapper":
        return _site_node_scratch_wrapper(args[1:])
    if not args or args[0] != "qualify":
        sys.stderr.write(_SITE_USAGE)
        return 64

    report = site_qualify.qualify(
        skip_model_fixtures="--skip-model-fixtures" in args)
    text = site_qualify.report_text(report)
    out = _flag(args, "--json")
    if out:
        try:
            with open(out, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            sys.stderr.write(f"site qualify: could not write --json {out}: "
                             f"{exc}\n")
            return 2
        sys.stderr.write(f"wrote {out}\n")
    for line in site_qualify.human_lines(report):
        sys.stderr.write(line + "\n")
    sys.stdout.write(text)

    summary = report["summary"]
    payload = {
        "passed": summary["passed"],
        "warnings": summary["warnings"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "checkCount": summary["total"],
        "failingChecks": site_qualify.failing_ids(report),
        "warningChecks": site_qualify.warning_ids(report),
        "report": report,
    }
    failing = site_qualify.failing_ids(report)
    if failing:
        remedies = "; ".join(
            f"{row['id']}: {row['detail']}" for row in report["checks"]
            if row["status"] == "fail")
        return CLIResult(
            state="failed", exit_code=70, code=SITE_QUALIFY_FAILED,
            message=(f"site qualify FAILED — {', '.join(failing)}. "
                     + summary["line"]),
            repair_action=remedies or (
                "read the failing check's `detail` in the report and repair "
                "the node, then re-run steerlab-server site qualify"),
            payload=payload)
    advisories = [
        advisory("siteQualifyWarning", f"{row['id']}: {row['observed']}")
        for row in report["checks"] if row["status"] == "warn"]
    return CLIResult(
        message=summary["line"], payload=payload, advisories=advisories,
        state="okWithAdvisories" if advisories else "ready")


_DATA_USAGE = ("usage: steerlab-server data check optvec "
               "[--dir prompts/optvec] [--json]\n"
               "       steerlab-server data check lora "
               "[<package-manifest-or-dir>] [--dir adapters] [--json]\n")

#: The server-side ``data check`` templates. Both are DATA-directory driven
#: (not manifest-driven like the Swift twin) because both datasets are
#: authored before any experiment manifest pins them.
_DATA_TEMPLATES = ("optvec", "lora")


def _data(args: list[str]):
    """Study-data readiness — the server side of the ``data check`` layer
    (Swift twin: ``steerlab-cli data check <experiment>``, manifest-driven).
    Two server templates, both authored BEFORE any experiment manifest exists
    and pinned by hash afterward, so both are directory-driven: ``optvec``
    (the OptVec dataset bundle) and ``lora`` (a workspace ``adapters/``
    training-dataset package — manifest, per-arm train/validation files,
    strict row schemas, split integrity, and the package's own QC verdict).
    Prints one line per requirement, blockers first, with per-file SHA-256
    for every VALID file (paste those into the configs verbatim). Exit
    codes: 0 = ready; **65 = at least one blocker** (the audit's one scheduled
    human-mode migration, 2 → 65, landing on both engines together — §2.3);
    2 = no bundle directory at all; 64 = usage."""
    # `data check <experiment>` is the MAC verb's argument domain (it is
    # manifest-driven there). Answering it with this engine's two-template
    # usage line said "you typed a bad subject" when the truth is "that check
    # lives on the other engine" — §7's promise, unkept (gate-5 dry run #2,
    # P3). A subject that is neither template is redirected; a genuinely
    # malformed `data …` keeps its usage line.
    if len(args) >= 2 and args[0] == "check" and args[1] not in _DATA_TEMPLATES:
        refusal = _mac_authority_refusal(
            f"data check {args[1]}",
            f"steerlab-cli data check {args[1]}",
            note="this engine's `data check` takes "
                 + " | ".join(_DATA_TEMPLATES)
                 + " — two directory-driven dataset templates authored before "
                   "any manifest pins them; the manifest-driven readiness "
                   "checklist for an experiment is Mac-authority")
        # The usage line still prints: the redirect says where the check
        # lives, and this says what this engine's own `data check` accepts.
        sys.stderr.write(_DATA_USAGE)
        return refusal
    if len(args) < 2 or args[0] != "check" or args[1] not in _DATA_TEMPLATES:
        sys.stderr.write(_DATA_USAGE)
        return 64
    from . import cli_envelope
    from .cli_envelope import CLIResult
    from .experiment import data_readiness, lifecycle_gates, paths
    positional: list[str] = []
    skip = False
    for value in args[2:]:
        if skip:
            skip = False
        elif value.startswith("-"):
            skip = value == "--dir"     # the only value-taking flag here
        else:
            positional.append(value)
    try:
        if args[1] == "lora":
            target = paths.resolve(
                positional[0] if positional
                else (_flag(args, "--dir")
                      or data_readiness.DEFAULT_LORA_DIRECTORY))
            report = data_readiness.check_lora_package(target)
        else:
            directory = paths.resolve(_flag(args, "--dir")
                                      or data_readiness.DEFAULT_BUNDLE_DIRECTORY)
            report = data_readiness.check_optvec_bundle(directory)
    except OSError as exc:
        sys.stderr.write(f"data check: {exc}\n")
        # No package at all is a data-readiness refusal like any other — same
        # gate, same JSON-mode 65. Its HUMAN code stays 2, unchanged: the
        # audit's scheduled migration is the BLOCKER code, and this path
        # answers "there is nothing here to check".
        return CLIResult(
            state="refused", exit_code=2,
            code=lifecycle_gates.DATA_READINESS,
            gate=lifecycle_gates.DATA_READINESS,
            message=f"data check: {exc}",
            repair_action=(f"author the {args[1]} package at the named path, "
                           f"then steerlab-server data check {args[1]}"),
            payload={"template": args[1], "ready": False})
    for req in report.requirements:
        print(f"{req.status:<8} {req.name} — {req.detail}")
        if req.sha256:
            print(f"         sha256 {req.sha256}")
    counts: dict[str, int] = {}
    for req in report.requirements:
        counts[req.status] = counts.get(req.status, 0) + 1
    breakdown = ", ".join(
        f"{counts[s]} {s}" for s in ("invalid", "missing", "partial",
                                     "present") if s in counts)
    verdict = "ready" if report.ready else \
        f"NOT ready ({len(report.blockers)} blocker(s))"
    summary = (f"{len(report.requirements)} requirement(s): {breakdown} — "
               f"{verdict}")
    print(f"\n{summary}")
    # `--json` is the ENVELOPE flag now, on every agent-path verb, and the
    # report it used to print bare rides inside it (audit §2.2's
    # normalisation). Nothing is lost: `result.report` is the same dict, key
    # for key, plus the classified `blockers` list an agent acts on.
    payload = {
        "template": args[1],
        "ready": report.ready,
        "summary": summary,
        "counts": counts,
        "blockers": [
            {"name": blocker.name, "status": blocker.status,
             "detail": blocker.detail}
            for blocker in report.blockers],
        "report": report.to_dict(),
    }
    if report.ready:
        return CLIResult(message=summary, payload=payload)
    plural = "" if len(report.blockers) == 1 else "s"
    # THE human-mode exit migration the audit schedules by name (§2.3's stated
    # migration debt, §7 row 7: "`data check` blockers (2 → 65)"). It lands
    # once, on both engines, gated by the envelope's schemaVersion — not
    # aliased. `data check` now answers 65 in BOTH modes; every other refusal
    # keeps its human exit code.
    return CLIResult(
        state="refused",
        exit_code=cli_envelope.exit_code_for("refused"),
        code=lifecycle_gates.DATA_READINESS,
        gate=lifecycle_gates.DATA_READINESS,
        message=(f"{len(report.blockers)} blocking data requirement{plural} "
                 f"for the {args[1]} template"),
        repair_action=("author the files named by result.blockers[].name, "
                       f"then steerlab-server data check {args[1]}"),
        payload=payload)


_BATTERY_USAGE = (
    "usage: steerlab-server battery lint <path> [--json]\n"
    "       (path is workspace-relative, e.g. prompts/batteries/x.jsonl)\n"
    "       steerlab-server battery generation-prompt [--count N] "
    "[--avoid <domain text>] [--out <file>]\n")


def _battery(args: list[str]) -> int:
    """Capability-battery verbs. A battery is a CONTROL: it must hold still
    while the intervention moves.

    ``lint`` (2026-08-13) is the preflight — it reads the file's bytes, never
    a model, and names every way this one can fail to hold still: legacy
    arming that lets the study's own system prompt decide the score,
    containment/whole-response text matching that grades response length,
    options that are not discriminative, floor/ceiling and
    effective-item-count risks.

    ``generation-prompt`` is the other end of the same contract: an authoring
    brief, assembled from the format-2 schema and the linter's own thresholds,
    to hand to an LLM so a draft passes ``lint`` first time. Text only — it
    reads no model and writes nothing unless ``--out`` is given.

    Exit codes: 0 = done / no blockers (warnings may still print); 2 = at
    least one lint blocker, so the battery should not be pinned as a
    capability control; 64 = usage."""
    if not args or args[0] not in ("lint", "generation-prompt"):
        sys.stderr.write(_BATTERY_USAGE)
        return 64
    if args[0] == "generation-prompt":
        return _battery_generation_prompt(args[1:])
    if len(args) < 2:
        sys.stderr.write(_BATTERY_USAGE)
        return 64
    from .experiment import battery_lint
    report = battery_lint.lint(args[1])
    if "--json" in args:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        version = report.format_version
        print(f"{report.path}  format {version if version else '?'}  "
              f"{report.item_count} item(s)"
              + (f"  sha256 {report.digest}" if report.digest else ""))
        for finding in report.findings:
            where = f"item {finding.item}" if finding.item else "file"
            print(f"{finding.severity:<8} {where:<9} {finding.code}: "
                  f"{finding.detail}")
        verdict = "no blockers" if report.ok else \
            f"NOT usable as a capability control ({len(report.blockers)} blocker(s))"
        print(f"\n{len(report.blockers)} blocker(s), "
              f"{len(report.warnings)} warning(s) — {verdict}")
    return 0 if report.ok else 2


def _battery_generation_prompt(rest: list[str]) -> int:
    """``battery generation-prompt`` — print the authoring brief.

    Strict flag parsing (the family's discipline): an unknown flag, a missing
    value, or a non-positive ``--count`` is a usage error rather than a
    silently-ignored argument, because the brief it would print looks correct.
    Plain text on stdout by default; ``--out`` writes the same bytes to a file
    and says so on stderr, so stdout stays pipeable either way.
    """
    from .experiment import battery_brief

    count = battery_brief.DEFAULT_ITEM_COUNT
    avoid = ""
    out_path = None
    index = 0
    while index < len(rest):
        option = rest[index]
        if option not in ("--count", "--avoid", "--out"):
            sys.stderr.write(f"battery generation-prompt: unknown argument "
                             f"{option!r}\n" + _BATTERY_USAGE)
            return 64
        if index + 1 >= len(rest):
            sys.stderr.write(f"battery generation-prompt: {option} needs a "
                             "value\n" + _BATTERY_USAGE)
            return 64
        value = rest[index + 1]
        if option == "--count":
            try:
                count = int(value)
            except ValueError:
                sys.stderr.write("battery generation-prompt: --count needs an "
                                 "integer\n")
                return 64
            if count < 1:
                sys.stderr.write("battery generation-prompt: --count must be "
                                 "at least 1\n")
                return 64
        elif option == "--avoid":
            avoid = value
        else:
            out_path = value
        index += 2

    text = battery_brief.generation_prompt(count, avoid=avoid)
    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            sys.stderr.write(f"battery generation-prompt: {exc}\n")
            return 2
        sys.stderr.write(f"wrote {out_path}\n")
        return 0
    sys.stdout.write(text)
    return 0


_OPTVEC_USAGE = (
    "usage: steerlab-server optvec train --config <path.json>\n"
    "       steerlab-server optvec eval --config <path.json>\n"
    "       steerlab-server optvec geometry [--out-name NAME] [--layer N] "
    "<artifact> <artifact> …\n"
    "       steerlab-server optvec geometry --config <path.json>\n"
    "       steerlab-server optvec campaign materialize --config <path.json>\n"
    "       steerlab-server optvec campaign submit <campaign-dir>\n"
    "       steerlab-server optvec campaign status <campaign-dir>\n"
    "       steerlab-server optvec interpret --config <path.json>\n"
    "       steerlab-server optvec family --config <path.json>\n"
    "       steerlab-server optvec jspace --config <path.json>\n"
    "       steerlab-server optvec gradient --config <path.json>\n"
    "       steerlab-server optvec gradient mint <survey-run-dir> <item-id> "
    "[--name N]\n"
    "       steerlab-server optvec fracture --config <path.json>\n")


def _optvec(args: list[str]) -> int:
    """OptVec verbs (docs/OPTVEC-…-PLAN.md): ``train`` optimizes an injection
    vector by backprop through the frozen model (WP2); ``eval`` reads the TEST
    split behaviorally at a dose grid (WP4); ``geometry`` reports the family's
    cosine matrix and participation ratio (WP4); ``campaign`` materializes and
    top-up-submits the Slurm grid (WP6); ``interpret``/``family`` read what a
    solution (and the solution family) contains (WP7); ``jspace`` reads what
    the model computes FROM it through an imported lens (WP8, exploratory
    tier). Each prints its run's JSON. Exit codes: 0 = done; 2 = bad
    config/dataset/artifact; 3 = campaign submit had failed sbatches;
    64 = usage."""
    if not args:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    if args[0] == "train":
        return _optvec_train(args[1:])
    if args[0] == "eval":
        return _optvec_eval(args[1:])
    if args[0] == "geometry":
        return _optvec_geometry(args[1:])
    if args[0] == "campaign":
        return _optvec_campaign(args[1:])
    if args[0] == "interpret":
        return _optvec_interpret(args[1:])
    if args[0] == "family":
        return _optvec_family(args[1:])
    if args[0] == "jspace":
        return _optvec_jspace(args[1:])
    if args[0] == "gradient":
        return _optvec_gradient_verb(args[1:])
    if args[0] == "fracture":
        return _optvec_fracture(args[1:])
    sys.stderr.write(_OPTVEC_USAGE)
    return 64


def _optvec_gradient_verb(args: list[str]) -> int:
    """``gradient --config <json>`` runs the per-item α→0 survey (S4: one
    forward+backward per item plus the dose-ladder linearity check);
    ``gradient mint <survey-run-dir> <item-id> [--name N]`` exports one item's
    direction as a standard lifecycle-compatible artifact."""
    from .experiment import optvec_gradient
    try:
        if args and args[0] == "mint":
            if len(args) < 3:
                sys.stderr.write(_OPTVEC_USAGE)
                return 64
            result = optvec_gradient.mint(args[1], args[2],
                                          name=_flag(args, "--name"))
        else:
            config_path = _flag(args, "--config")
            if not config_path:
                sys.stderr.write(_OPTVEC_USAGE)
                return 64
            config = optvec_gradient.load_config(config_path)
            result = optvec_gradient.survey(
                config, log=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, optvec_gradient.OptVecGradientConfigError,
            optvec_gradient.OptVecGradientDataError) as exc:
        sys.stderr.write(f"optvec gradient: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_fracture(args: list[str]) -> int:
    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    from .experiment import optvec_geometry
    try:
        config = optvec_geometry.load_fracture_config(config_path)
        result = optvec_geometry.fracture(config)
    except (OSError, optvec_geometry.OptVecGeometryError) as exc:
        sys.stderr.write(f"optvec fracture: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_campaign(args: list[str]) -> int:
    """``materialize`` writes the campaign directory (cells + scripts);
    ``submit`` tops the queue up to maxQueued and exits **3** when any cell's
    sbatch failed (the report is on stdout either way — this verb never buries
    a fan-out failure in a zero exit); ``status`` is read-only."""
    from .experiment import optvec_campaign
    try:
        if args and args[0] == "materialize":
            config_path = _flag(args, "--config")
            if not config_path:
                sys.stderr.write(_OPTVEC_USAGE)
                return 64
            config = optvec_campaign.load_config(config_path)
            campaign_dir = optvec_campaign.materialize(config)
            campaign = optvec_campaign.read_campaign(campaign_dir)
            print(json.dumps({"campaignDirectory": campaign_dir,
                              "cellCount": len(campaign["cells"])},
                             indent=2, sort_keys=True))
            return 0
        if args and args[0] == "submit" and len(args) >= 2:
            report = optvec_campaign.submit(args[1])
            print(json.dumps(report, indent=2, sort_keys=True))
            return 3 if report.get("failed") else 0
        if args and args[0] == "status" and len(args) >= 2:
            print(json.dumps(optvec_campaign.status(args[1]),
                             indent=2, sort_keys=True))
            return 0
    except (OSError, optvec_campaign.CampaignConfigError,
            optvec_campaign.CampaignError) as exc:
        sys.stderr.write(f"optvec campaign: {exc}\n")
        return 2
    sys.stderr.write(_OPTVEC_USAGE)
    return 64


def _optvec_interpret(args: list[str]) -> int:
    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    from .experiment import optvec_interpret
    try:
        config = optvec_interpret.load_config(config_path)
        result = optvec_interpret.interpret(
            config, log=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, optvec_interpret.OptVecInterpretConfigError,
            optvec_interpret.OptVecInterpretDataError) as exc:
        sys.stderr.write(f"optvec interpret: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_family(args: list[str]) -> int:
    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    from .experiment import optvec_interpret
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = optvec_interpret.FamilySummaryConfig.from_dict(
                json.load(handle))
        result = optvec_interpret.family_summary(
            config, log=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, optvec_interpret.OptVecInterpretConfigError,
            optvec_interpret.OptVecInterpretDataError) as exc:
        sys.stderr.write(f"optvec family: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_jspace(args: list[str]) -> int:
    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    from .experiment import optvec_jspace
    try:
        config = optvec_jspace.load_config(config_path)
        result = optvec_jspace.analyze(
            config, log=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, optvec_jspace.OptVecJSpaceConfigError,
            optvec_jspace.OptVecJSpaceError) as exc:
        sys.stderr.write(f"optvec jspace: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_train(args: list[str]) -> int:
    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    from .experiment import optvec_train
    try:
        config = optvec_train.load_config(config_path)
        result = optvec_train.train(
            config, log=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, optvec_train.OptVecConfigError,
            optvec_train.OptVecDataError) as exc:
        sys.stderr.write(f"optvec train: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_eval(args: list[str]) -> int:
    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_OPTVEC_USAGE)
        return 64
    from .experiment import optvec_eval
    try:
        config = optvec_eval.load_config(config_path)
        result = optvec_eval.evaluate(
            config, log=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, optvec_eval.OptVecEvalConfigError,
            optvec_eval.OptVecEvalDataError,
            optvec_eval.OptVecArtifactError) as exc:
        sys.stderr.write(f"optvec eval: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optvec_geometry(args: list[str]) -> int:
    from .experiment import optvec_geometry
    config_path = _flag(args, "--config")
    try:
        if config_path:
            config = optvec_geometry.load_config(config_path)
        else:
            flagged = {"--out-name", "--layer", "--config"}
            positional: list[str] = []
            skip = False
            for i, arg in enumerate(args):
                if skip:
                    skip = False
                    continue
                if arg in flagged:
                    skip = True
                    continue
                positional.append(arg)
            raw_layer = _flag(args, "--layer")
            layer = None
            if raw_layer is not None:
                try:
                    layer = int(raw_layer)
                except ValueError:
                    sys.stderr.write(
                        f"optvec geometry: --layer must be an integer "
                        f"(got {raw_layer!r})\n")
                    return 64
            config = optvec_geometry.OptVecGeometryConfig(
                artifacts=positional, name=_flag(args, "--out-name"),
                layer=layer)
        result = optvec_geometry.geometry(config)
    except ValueError as exc:
        # OptVecGeometryError and OptVecArtifactError are both ValueErrors,
        # and both are the same "this cannot be computed as asked" for the
        # caller.
        sys.stderr.write(f"optvec geometry: {exc}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"optvec geometry: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


_SAE_USAGE = (
    "usage: steerlab-server sae candidates check <path> [--json]\n"
    "       steerlab-server sae candidates pin <experiment> <path>\n"
    "       steerlab-server sae family-report --config <path.json> "
    "[--out-name N]\n"
    "       steerlab-server sae qualification record --inputs <inputs.json> "
    "--artifact <runDir/name>\n"
    "       steerlab-server sae qualification show <path> [--json]\n")


def _sae_family_report(args: list[str]) -> int:
    """Cross-family DESCRIPTIVE report (proposal r2 §7 / §8 P1-7): CAA/gm,
    LoRA, OptVec and SAE rows side by side — identities, layers, norms,
    matched-layer cosines, and behavioural deltas copied verbatim from
    analyze artifacts. Carries no inferential statistics by construction."""
    from .experiment import family_report

    config_path = _flag(args, "--config")
    if not config_path:
        sys.stderr.write(_SAE_USAGE)
        return 64
    try:
        config = family_report.load_config(config_path)
        out_name = _flag(args, "--out-name")
        if out_name:
            # Rebuilt, not assigned: the name becomes a run-directory slug and
            # its validation lives in __post_init__.
            config = family_report.FamilyReportConfig(
                artifacts=config.artifacts, name=out_name,
                discover_promotions=config.discover_promotions)
        result = family_report.report(config)
    except ValueError as exc:
        # FamilyReportError and OptVecArtifactError are both ValueErrors, and
        # both mean the same "this report cannot be produced as asked".
        sys.stderr.write(f"sae family-report: {exc}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"sae family-report: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _sae(args: list[str]) -> int:
    """SAE candidate-roster and feature-qualification verbs
    (SAE-VECTOR-INTERVENTION proposal r2 §8 P1-6 and P0-4).

    ``candidates check`` validates a workspace candidate manifest and prints
    its roster summary — schema violations exit 2, so it works as a gate in a
    script; roster/qualification inconsistencies (a 'qualified' entry with no
    artifact pointer) print as WARNINGS and do not change the exit code.
    ``candidates pin`` stamps ``saeCandidates`` = {path, hash} into a DRAFT
    experiment; every edit afterwards is a verify() violation, which is why
    the roster is authored before the study is frozen, not after results
    arrive.

    ``qualification record`` assembles a durable, IMMUTABLE
    ``sae-feature-qualification.json`` from a declared inputs file, bound to
    the imported artifact's exact decoder-row bytes. It is citable evidence
    for ``experiment promote --qualification``, NOT a seating mechanism —
    seating stays sweep → promote like every other vector family (§6).
    ``qualification show`` prints one human-readably.

    All four verbs are offline: nothing here touches Neuronpedia or HF."""
    from .experiment import experiment_store as es
    from .experiment import sae_candidates

    if args and args[0] == "family-report":
        return _sae_family_report(args[1:])
    if len(args) >= 2 and args[0] == "qualification":
        return _sae_qualification(args[1:])
    if len(args) < 2 or args[0] != "candidates":
        sys.stderr.write(_SAE_USAGE)
        return 64
    verb = args[1]
    rest = [a for a in args[2:] if not a.startswith("-")]

    if verb == "check":
        if len(rest) != 1:
            sys.stderr.write(_SAE_USAGE)
            return 64
        try:
            manifest, digest = sae_candidates.load(rest[0])
        except sae_candidates.CandidateManifestError as exc:
            sys.stderr.write(f"sae candidates check: {exc}\n")
            return 2
        summary = manifest.summary()
        # Roster/qualification consistency: an entry whose status records a
        # qualification OUTCOME but names no artifact has a verdict with
        # nothing behind it. A WARNING, never an error — the roster is
        # authored iteratively, and the exit code stays the schema's verdict.
        from .experiment import sae_qualification
        warnings = sae_qualification.roster_warnings(manifest)
        if "--json" in args:
            print(json.dumps({"path": rest[0], "sha256": digest,
                              "warnings": warnings,
                              **summary}, indent=2, sort_keys=True))
            return 0
        print(f"{rest[0]}  sha256 {digest}")
        label = summary["name"] or "(unnamed roster)"
        print(f"{label}: {summary['count']} candidate(s), schemaVersion "
              f"{summary['schemaVersion']}")
        for heading, key in (("by role", "byRole"), ("by status", "byStatus"),
                             ("verification", "byVerification")):
            counts = summary[key]
            body = ", ".join(f"{k} {counts[k]}" for k in sorted(counts)) or "—"
            print(f"  {heading:<13s} {body}")
        print(f"  {'evidence':<13s} "
              f"{summary['withDiscoverySnapshot']} with a discovery snapshot, "
              f"{summary['withQualification']} with a qualification artifact")
        print(f"  {'dictionary':<13s} "
              f"{summary['withGemmaScopeDictionary']} of {summary['count']} "
              "declare an exact gemmaScope release/saeID (the rest are "
              "matched on layer + featureId only)")
        for pending in summary["pendingConstructs"]:
            print(f"  pending slot  {pending['constructLabel']} "
                  f"({pending['role']}) — no feature nominated yet")
        for warning in warnings:
            sys.stderr.write(f"warning: {warning}\n")
        return 0

    if verb == "pin":
        if len(rest) != 2:
            sys.stderr.write(_SAE_USAGE)
            return 64
        try:
            d = es.pin_sae_candidates(rest[0], rest[1])
        except es.ExperimentStoreError as exc:
            sys.stderr.write(f"sae candidates pin: {exc}\n")
            return 2
        block = d["saeCandidates"]
        print(f"pinned {block['path']} into '{rest[0]}' "
              f"(sha256 {block['hash']})")
        return 0

    sys.stderr.write(_SAE_USAGE)
    return 64


def _sae_qualification(args: list[str]) -> int:
    """``sae qualification record|show`` (proposal r2 §8 P0-4).

    Self-contained on purpose (other work is landing in this file
    concurrently): it adds no shared state and touches no other verb.

    Exit codes: 0 = written/printed; 2 = refused (invalid inputs, identity
    mismatch against the artifact's decoder-row hash, uncovered dose grid,
    existing record); 64 = usage.
    """
    from .experiment import sae_qualification as saq

    verb = args[0] if args else ""

    if verb == "record":
        rest = args[1:]
        inputs_path = _flag(rest, "--inputs")
        artifact = _flag(rest, "--artifact")
        if not inputs_path or not artifact:
            sys.stderr.write(
                "sae qualification record: --inputs and --artifact are both "
                f"required (start from {saq.TEMPLATE_PATH})\n" + _SAE_USAGE)
            return 64
        try:
            with open(inputs_path, "rb") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except OSError as exc:
            sys.stderr.write(
                f"sae qualification record: cannot read {inputs_path}: {exc}\n")
            return 2
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            sys.stderr.write(
                f"sae qualification record: {inputs_path} is not valid JSON: "
                f"{exc}\n")
            return 2
        try:
            result = saq.record(inputs=payload, artifact=artifact)
        except saq.QualificationError as exc:
            sys.stderr.write(f"sae qualification record: {exc}\n")
            return 2
        except OSError as exc:
            sys.stderr.write(f"sae qualification record: {exc}\n")
            return 2
        print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
        return 0

    if verb == "show":
        rest = [a for a in args[1:] if not a.startswith("-")]
        if len(rest) != 1:
            sys.stderr.write(_SAE_USAGE)
            return 64
        try:
            record_obj, digest = saq.load(rest[0])
        except saq.QualificationError as exc:
            sys.stderr.write(f"sae qualification show: {exc}\n")
            return 2
        summary = record_obj.summary()
        # Cross-checks are REPORTED here, never fatal: a record whose artifact
        # moved is still readable evidence, and hiding it behind an error
        # would make a stale workspace unreadable rather than explicable.
        problems = saq.consistency_violations(record_obj)
        if "--json" in args:
            print(json.dumps({"path": rest[0], "sha256": digest,
                              "consistencyViolations": problems,
                              **summary}, indent=2, sort_keys=True))
            return 0
        print(f"{rest[0]}  sha256 {digest}")
        print(f"feature {summary['feature']} @ layer {summary['layer']}"
              + (f"  ({summary['constructLabel']})"
                 if summary["constructLabel"] else ""))
        print(f"  artifact      {summary['artifact'] or '—'}")
        print(f"  decoder row   {summary['decoderRowHash'] or '—'}")
        print(f"  decision      {summary['decision'].upper()} "
              f"({summary['decisionDate']})")
        print(f"                {record_obj.decision.rationale}")
        grid = ", ".join(f"{d:g}" for d in summary["doseGrid"])
        print(f"  dose grid     {grid}  signs {', '.join(summary['signs'])}")
        print(f"  construct     {record_obj.construct_probe.metric} "
              f"({summary['constructProbeRows']} row(s))")
        for row in record_obj.construct_probe.results:
            print(f"      {row.sign:<8s} dose {row.dose:<6g} "
                  f"{row.value:<12g} {row.direction or '—'}")
        print(f"  leakage       {record_obj.lexical_leakage.metric} "
              f"({summary['lexicalLeakageRows']} row(s))")
        discriminants = ", ".join(summary["discriminantControls"]) or "none"
        print(f"  discriminant  {discriminants}")
        print(f"  coherence     {record_obj.coherence_gate.metric}: "
              f"{'passed' if summary['coherenceGatePassed'] else 'FAILED'}")
        print(f"  dose response monotone={summary['doseResponseMonotone']}")
        for run in summary["evidenceRuns"]:
            print(f"  evidence      {run['path']}"
                  + (f"  — {run['describes']}" if run.get("describes") else ""))
        for problem in problems:
            sys.stderr.write(f"warning: {problem}\n")
        return 0

    sys.stderr.write(_SAE_USAGE)
    return 64


_GEMMASCOPE_USAGE = (
    "usage: steerlab-server gemmascope import-id --model <id> --release <rel> "
    "--sae-id <sae> --feature <n> --label <construct> "
    "--residual-norm-artifact <runDir/name> [--layer <n>] "
    "[--neuronpedia-url <url>] [--name <artifact-name>]\n"
    "  --residual-norm-artifact is a CALIBRATION DONOR: any artifact of the\n"
    "  same model carrying residualNormPerLayer. It supplies the denominator,\n"
    "  not a semantic pairing.\n")


def _gemmascope(args: list[str]) -> int:
    """``gemmascope import-id`` — mint a steering-vector artifact from a Gemma
    Scope feature named by ID (proposal r2 §8 P0-1/P0-2; API twin:
    ``POST /api/gemmascope/import-id``).

    Self-contained on purpose: the report-based import stays exactly where it
    was, and this verb adds no shared state to the module.

    Exit codes: 0 = imported (result JSON on stdout); 2 = refused (bad donor,
    dimension/layer/model mismatch, unresolvable SAE, existing artifact);
    64 = usage.
    """
    if not args or args[0] != "import-id":
        sys.stderr.write(_GEMMASCOPE_USAGE)
        return 64
    rest = args[1:]
    required = {
        "--model": _flag(rest, "--model"),
        "--release": _flag(rest, "--release"),
        "--sae-id": _flag(rest, "--sae-id"),
        "--feature": _flag(rest, "--feature"),
        "--label": _flag(rest, "--label"),
        "--residual-norm-artifact": _flag(rest, "--residual-norm-artifact"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        sys.stderr.write(f"gemmascope import-id: missing {' '.join(missing)}\n"
                         + _GEMMASCOPE_USAGE)
        return 64
    try:
        feature = int(required["--feature"])
    except ValueError:
        sys.stderr.write(
            f"gemmascope import-id: bad --feature {required['--feature']!r}\n")
        return 64
    layer_flag = _flag(rest, "--layer")
    try:
        layer = int(layer_flag) if layer_flag is not None else None
    except ValueError:
        sys.stderr.write(f"gemmascope import-id: bad --layer {layer_flag!r}\n")
        return 64

    from .experiment import gemma_scope, paths
    from .experiment.run_config import write_run_config

    run_dir = paths.make_unique_run_directory(f"sae-feature-{feature}")
    write_run_config(run_dir, "sae-feature-import",
                     model_id=required["--model"])
    try:
        artifact = gemma_scope.import_feature_by_id(
            model_id=required["--model"], release=required["--release"],
            sae_id=required["--sae-id"], feature=feature,
            label=required["--label"],
            residual_norm_artifact=required["--residual-norm-artifact"],
            run_directory=run_dir, layer=layer,
            neuronpedia_url=_flag(rest, "--neuronpedia-url"),
            name=_flag(rest, "--name"))
    except (ValueError, RuntimeError, OSError) as exc:
        sys.stderr.write(f"gemmascope import-id: {exc}\n")
        return 2
    with open(artifact + ".json", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    print(json.dumps({
        "ok": True, "vectorPath": run_dir,
        "name": os.path.basename(artifact), "artifact": artifact,
        "concept": sidecar.get("concept"),
        "gemmascopeConvention": sidecar.get("gemmascopeConvention"),
        "rawDecoderNorm": sidecar.get("rawDecoderNorm"),
        "gemmascopeTargetNorm": sidecar.get("gemmascopeTargetNorm"),
        "gemmascopeSource": sidecar.get("gemmascopeSource"),
    }, indent=2, sort_keys=True))
    return 0


def _flag(args: list[str], name: str) -> str | None:
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return None


_PREFLIGHT_ENDPOINTS_USAGE = (
    "usage: steerlab-server experiment preflight-endpoints <name> "
    "[--baseline-run DIR] [--out PATH] [--json]\n"
    "       [--band LOW,HIGH] [--min-cell-items N] [--min-items N]\n"
    "  Endpoint-safety preflight BEFORE a screen consumes compute: factorial\n"
    "  aliasing, effective item count per endpoint and stratum cell, and\n"
    "  signed-cancellation exposure from the manifest alone; floor/ceiling\n"
    "  width, missingness and format-compliance sensitivity when\n"
    "  --baseline-run names an existing (read-only) run directory.\n"
    "  Writes a durable JSON report to a new runs/<stamp>-exp-<name>-"
    "endpoint-preflight/\n"
    "  directory, or to --out. Exit 0 = ok/warnings only; 2 = blockers.\n")


def _preflight_endpoints(name: str, rest: list[str], root: str | None) -> int:
    """``experiment preflight-endpoints`` — the P1-8 endpoint-safety preflight.

    Deliberately self-contained (one dispatch line above, everything else
    here): the checks live in
    :mod:`steerlab_server.experiment.endpoint_preflight`, this only parses
    flags, prints, and maps the verdict to an exit code. Exit codes follow the
    ``data check`` convention — 0 ready, 2 blockers or unusable inputs, 64
    usage — because a researcher reads both checks the same way.
    """
    from dataclasses import replace

    from .experiment import endpoint_preflight as preflight_mod

    thresholds = preflight_mod.DEFAULT_THRESHOLDS
    band = _flag(rest, "--band")
    if band:
        try:
            low, high = (float(part) for part in band.split(","))
        except ValueError:
            sys.stderr.write(_PREFLIGHT_ENDPOINTS_USAGE)
            return 64
        if not 0.0 <= low < high <= 1.0:
            sys.stderr.write(
                "preflight-endpoints: --band LOW,HIGH must satisfy "
                "0 <= LOW < HIGH <= 1\n")
            return 64
        thresholds = replace(thresholds, band_low=low, band_high=high)
    for option, attribute in (("--min-cell-items", "min_cell_items"),
                              ("--min-items", "min_effective_items")):
        raw = _flag(rest, option)
        if raw is not None:
            try:
                value = int(raw)
            except ValueError:
                sys.stderr.write(f"preflight-endpoints: {option} needs an "
                                 "integer\n")
                return 64
            if value < 1:
                sys.stderr.write(f"preflight-endpoints: {option} must be "
                                 "at least 1\n")
                return 64
            thresholds = replace(thresholds, **{attribute: value})

    try:
        report = preflight_mod.preflight(
            name, root=root, baseline_run=_flag(rest, "--baseline-run"),
            thresholds=thresholds)
        written = preflight_mod.write_report(
            report, out=_flag(rest, "--out"), root=root)
    except preflight_mod.PreflightError as exc:
        sys.stderr.write(f"preflight-endpoints: {exc}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"preflight-endpoints: {exc}\n")
        return 2

    # The DEPRECATED caseFamily trigger also names this report's numeric
    # endpoint (`meanMonths` rather than `parsedValueMean` —
    # endpoint_preflight._declared_endpoints). Said after the report is built,
    # so a preflight that never ran does not advise about an endpoint it never
    # named. Stderr, because stdout here is the report itself.
    try:
        from .experiment.manifest import (IMPLICIT_CASE_FAMILY_ADVISORY,
                                          Manifest,
                                          implicit_case_family_endpoint)
        if implicit_case_family_endpoint(Manifest.load(name, root)):
            sys.stderr.write(f"ADVISORY: {IMPLICIT_CASE_FAMILY_ADVISORY}\n")
    except Exception:  # noqa: BLE001 — an advisory never sinks a verb
        pass

    if "--json" in rest:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(preflight_mod.format_text(report))
    print(f"\nreport: {written}", file=sys.stderr)
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
