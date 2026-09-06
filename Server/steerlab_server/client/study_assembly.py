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
)
EXPERIMENT_VERBS = frozenset(s.verb for s in VERB_SPECS if s.family == "experiment")


def run(invocation) -> CLIResult:
    from ..client_cli import ClientRefusal
    from . import authoring_files as files, study_inputs, study_packs

    spec = invocation.spec
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
            payload = (study_packs.preview(data, root=root) if spec.verb == "preview"
                       else study_packs.apply(data, root=root, expected=one("--review-sha256")))
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
