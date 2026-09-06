"""Declarative Python design and study-interview CLI surface."""
from pathlib import Path
from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec("design", "expand", positional="<name>", purpose="Preview distinct panel castings as batch rows without creating studies.", value_flags=frozenset({"--file-sha256", "--casting", "--mode"}), required_flags=frozenset({"--file-sha256", "--casting", "--mode"})),
    VerbSpec("authoring", "study", positional="<intent>", purpose="Emit the shared researcher interview and reviewed study-pack workflow."),
    VerbSpec("design", "list", purpose="List reusable local designs and report unreadable entries."),
    VerbSpec("design", "inspect", positional="<name>", purpose="Inspect a design, its byte review and portable lineage identity."),
    VerbSpec("design", "describe", positional="<name>", purpose="Update only a reviewed design's description.",
             value_flags=frozenset({"--file-sha256", "--description"}), required_flags=frozenset({"--file-sha256", "--description"})),
    VerbSpec("design", "save", positional="<study>", purpose="Save reusable settings from a reviewed source study without modifying it.",
             value_flags=frozenset({"--manifest-sha256", "--name", "--description"}), required_flags=frozenset({"--manifest-sha256"})),
    VerbSpec("design", "update", positional="<name>", purpose="Update a reviewed design from a reviewed study of its lineage.",
             value_flags=frozenset({"--study", "--manifest-sha256", "--file-sha256"}), required_flags=frozenset({"--study", "--manifest-sha256", "--file-sha256"})),
    VerbSpec("design", "instantiate", positional="<name>", purpose="Cast a reviewed design into an ordinary draft.",
             value_flags=frozenset({"--file-sha256", "--casting", "--study-name"}), required_flags=frozenset({"--file-sha256", "--casting"})),
    VerbSpec("design", "batch", positional="<name>", purpose="Mint sibling drafts with explicit per-row outcomes; retry only failed rows.",
             value_flags=frozenset({"--file-sha256", "--rows"}), required_flags=frozenset({"--file-sha256", "--rows"})),
    VerbSpec("agent", "inspect", positional="<path>", purpose="Read an existing agent artifact and its exact byte digest for a casting."),
)


def run(invocation) -> CLIResult:
    from ..client_cli import ClientRefusal
    from . import authoring_files as files, design_files as library, study_designs as designs, design_panels
    args, one, spec = invocation.positionals, invocation.one, invocation.spec
    if (len(args) != (0 if spec.verb == "list" else 1) or any(one(f) is None for f in spec.required_flags)
            or any(len(v) != 1 for v in invocation.flags.values())):
        raise ClientRefusal(code="usage", reason="Supply the declared arguments and each required flag once.", repair_action=f"steerlab {spec.label} --help")
    if spec.family == "authoring":
        from .study_interviews import prompt
        intent = "conceptStudy" if args[0] == "confirmAgent" else args[0]
        if intent not in ("conceptStudy", "agentComparison", "multiAgent"):
            raise ClientRefusal(code="usage", reason="Choose conceptStudy, agentComparison or multiAgent.", repair_action="steerlab authoring study <intent> --json")
        text = prompt(intent)
        print(text)
        return CLIResult(message="Study coauthoring instructions emitted.", payload={"intent": intent, "prompt": text})
    root = files.root_path()
    if spec.family == "agent":
        path = library.ordinary(root, args[0])
        from ..experiment.manifest_files import digest_bytes
        sha = digest_bytes(path.read_bytes())
        arm = design_panels.review_agent({"artifactPath": args[0], "artifactFileSHA256": sha}, root)
        result = {"path": args[0], "artifactFileSHA256": sha, "artifact": arm["artifact"], "document": arm["artifact"]}
    elif spec.verb == "list":
        result = library.catalog(root)
    elif spec.verb == "inspect":
        result = library.inspect(args[0], root)
    elif spec.verb == "describe":
        result = designs.describe(args[0], one("--description"), root=root, expected=one("--file-sha256"))
    elif spec.verb == "save":
        result = designs.create(args[0], root=root, expected=one("--manifest-sha256"), name=one("--name"), description=one("--description"))
    elif spec.verb == "update":
        result = designs.update(args[0], one("--study"), root=root, expected=one("--file-sha256"), source_expected=one("--manifest-sha256"))
    elif spec.verb == "expand":
        from .design_expansion import expand
        result = expand(args[0], library.decode(Path(one("--casting")).read_bytes()), one("--mode"), root=root, expected=one("--file-sha256"))
    elif spec.verb == "instantiate":
        result = designs.instantiate(args[0], library.decode(Path(one("--casting")).read_bytes()), root=root,
                                     expected=one("--file-sha256"), study_name=one("--study-name"))
    else:
        result = designs.batch(args[0], library.decode(Path(one("--rows")).read_bytes()), root=root, expected=one("--file-sha256"))
        if not result["ok"]:
            # A partial batch must retain both its successful writes and repair.
            return CLIResult(message="Some drafts were created; inspect every row and retry only failures.", changed=result["changed"], payload=result, state="failed" if any(r.get("issue", {}).get("state") == "failed" for r in result["results"]) else "refused",
                             code="designBatchIncomplete", repair_action=result["repairAction"], exit_code=65)
    print(f"{spec.label}: complete; review the returned document before execution")
    return CLIResult(message="Review complete.", changed=result.get("changed", False), payload=result,
                     advisories=[{"code": "designDerivationWarning", "detail": warning} for warning in result.get("warnings", [])])
