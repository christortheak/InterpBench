"""Thin command adapter for the local study assembly owners."""
from __future__ import annotations

from pathlib import Path

from ..cli_envelope import CLIResult, VerbSpec


VERB_SPECS = (
    VerbSpec("pack", "preview", positional="<file>", purpose="Review a proposed study pack without writing workspace files."),
    VerbSpec("pack", "apply", positional="<file>", purpose="Create a draft from the exact reviewed pack and input files.",
             value_flags=frozenset({"--review-sha256"}), required_flags=frozenset({"--review-sha256"})),
    VerbSpec("pack", "export", positional="<study>", purpose="Export a study and its prompt text inputs; report external dependencies."),
    VerbSpec("experiment", "inspect", positional="<name>", purpose="Read the manifest and its external file digest for a reviewed edit."),
    VerbSpec("experiment", "import-prompts", positional="<name>", purpose="Validate full JSONL records and pin a new immutable prompt version in a reviewed draft.",
             value_flags=frozenset({"--file", "--manifest-sha256"}),
             required_flags=frozenset({"--file", "--manifest-sha256"})),
    VerbSpec("experiment", "inspect-artifact", positional="<path>", purpose="Review both files of a workspace vector artifact without loading a model."),
    VerbSpec("experiment", "attach-artifact", positional="<name> <concept>", purpose="Attach the reviewed vector pair to the reviewed draft through scientific admission.",
             value_flags=frozenset({"--artifact", "--artifact-sha256", "--sidecar-sha256",
                                    "--manifest-sha256", "--source-concept", "--eval-run"}),
             required_flags=frozenset({"--artifact", "--artifact-sha256", "--sidecar-sha256", "--manifest-sha256"})),
    # The Mac verb's shape, plus the reviewed-draft digest every client edit
    # carries. The rubric's SHA-256 is computed from its bytes, never typed.
    VerbSpec("experiment", "pin-rubric", positional="<name> <rubric>",
             purpose="Pin a judge rubric file (a workspace path, usually under prompts/rubrics/) into a reviewed draft, "
                     "computing its SHA-256 from the file; with --judges, declare the judge panel "
                     "(<name>:<kind>[:<model>[:<provider>]][,…]) and with --judge-pin a local judge's "
                     "<judge-name>=<revision>[:<dtype>].",
             value_flags=frozenset({"--judges", "--judge-pin", "--manifest-sha256"}),
             required_flags=frozenset({"--manifest-sha256"})),
)
EXPERIMENT_VERBS = frozenset(s.verb for s in VERB_SPECS if s.family == "experiment")


def _pin_rubric(invocation) -> CLIResult:
    """``experiment pin-rubric``: the one verb here whose flag repeats
    (``--judge-pin``, once per local judge, as on the Mac)."""
    from ..cli_envelope import advisory
    from ..client_cli import ClientRefusal
    from ..experiment import manifest_declaration_policy as policy
    from . import authoring_files as files, judge_rubrics

    args = invocation.positionals
    repeated = sorted(flag for flag, values in invocation.flags.items()
                      if flag != "--judge-pin" and len(values) != 1)
    if len(args) != 2 or invocation.one("--manifest-sha256") is None or repeated:
        raise ClientRefusal(
            code="usage", reason="experiment pin-rubric needs <name> <rubric> and --manifest-sha256, each once.",
            repair_action="steerlab experiment pin-rubric <name> prompts/rubrics/<file>.md "
                          "[--judges <name>:<kind>[,…]] --manifest-sha256 <manifestFileSHA256 from: "
                          "steerlab experiment inspect <name>>")
    name, rubric = args
    try:
        judges = (judge_rubrics.parse_judges(invocation.one("--judges"), name)
                  if invocation.one("--judges") is not None else None)
        pins = [judge_rubrics.parse_judge_pin(raw, name) for raw in invocation.all("--judge-pin")]
        payload = judge_rubrics.pin(name, rubric, judges=judges, judge_pins=pins,
                                    expected=invocation.one("--manifest-sha256"), root=files.root_path())
    except judge_rubrics.JudgeSpecError as exc:
        raise ClientRefusal(code="usage", reason=exc.reason, repair_action=exc.repair_action) from exc
    lines = [f"pinned judge rubric {payload['judgeRubricFile']} @ {payload['judgeRubricHash'][:12]}…"]
    if payload["judges"]:
        lines.append("judges: " + ", ".join(
            f"{j['name']} ({j['kind']})" + (f" @ {j['revision'][:12]}…" if j.get("revision") else "")
            + (f" {j['dtype']}" if j.get("dtype") else "") for j in payload["judges"]))
    notes = payload.get("inheritedFromExistingDeclaration") or []
    if notes:
        lines.append("kept from the panel already declared: " + "; ".join(notes))
    print("\n".join(lines))
    # A one-judge panel is a legal design; the advisory says what it costs.
    advisories = ([advisory("judgePanelTooSmall", policy.SINGLE_JUDGE_PANEL_ADVISORY)]
                  if len(payload["judges"]) == 1 else [])
    return CLIResult(message=lines[0], changed=payload["changed"], payload=payload,
                     state="okWithAdvisories" if advisories else "ready", advisories=advisories)


def run(invocation) -> CLIResult:
    from ..client_cli import ClientRefusal
    from . import authoring_files as files, study_inputs, study_packs

    spec = invocation.spec
    if spec.verb == "pin-rubric":
        return _pin_rubric(invocation)
    args = invocation.positionals
    one = invocation.one
    count = 2 if spec.verb == "attach-artifact" else 1
    if len(args) != count:
        raise ClientRefusal(code="usage", reason=f"{spec.label} needs {spec.positional}.",
                            repair_action=f"steerlab {spec.label} --help")
    missing = sorted(flag for flag in spec.required_flags if one(flag) is None)
    repeated = sorted(flag for flag, values in invocation.flags.items() if len(values) != 1)
    if missing or repeated:
        raise ClientRefusal(code="usage", reason=f"Missing flags: {missing}; repeated flags: {repeated}.",
                            repair_action=f"steerlab {spec.label} --help; supply each declared flag once.")
    root = files.root_path()
    if spec.family == "pack":
        if spec.verb == "export":
            payload = study_packs.export(args[0], root=root)
        else:
            data = Path(args[0]).read_bytes()
            if spec.verb == "preview":
                payload = study_packs.preview(data, root=root)
            else:
                from . import custom_code_commands
                payload = study_packs.apply(data, root=root, expected=one("--review-sha256"))
                # A pack that carries custom code says so on arrival.
                custom_code_commands.attach_notice(payload, payload["study"]["document"], root,
                                                   study=payload["study"]["name"])
    elif spec.verb == "inspect":
        payload = files.snapshot(args[0], root)
    elif spec.verb == "import-prompts":
        payload = study_inputs.import_prompts(args[0], Path(one("--file")).read_text(encoding="utf-8"),
                                             root=root, expected=one("--manifest-sha256"))
    elif spec.verb == "inspect-artifact":
        payload = study_inputs.inspect_artifact(args[0], root=root)
    else:
        payload = {"study": study_inputs.attach_artifact(
            args[0], args[1], one("--artifact"), root=root, expected=one("--manifest-sha256"),
            artifact_sha256=one("--artifact-sha256"), sidecar_sha256=one("--sidecar-sha256"),
            source_concept=one("--source-concept"), eval_run=one("--eval-run")), "changed": True}
    changed = payload.get("changed", False)
    print(f"{spec.label}: " + ("draft saved; review verification before execution" if changed else "review complete"))
    return CLIResult(message="Draft saved; inspect verification before execution." if changed else "Review complete.",
                     changed=changed, payload=payload)
