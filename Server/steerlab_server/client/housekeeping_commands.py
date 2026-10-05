"""``experiment rename|delete``, ``design rename|delete``, and ``agent delete``.

Every verb previews by default: it reads the current state, applies every
rule, and prints what would change together with the reviewed file digest,
writing nothing. The change happens only when the same command is repeated
with that digest and ``--yes``, and it is refused if the file changed in
between. Same verbs, flags, and result keys as on the Mac (``steerlab-cli``).
"""
from __future__ import annotations

from ..cli_envelope import CLIResult, VerbSpec

#: The reviewed-digest flag each family already uses for its other reviewed
#: edits, so the digest an inspection printed is the one typed here.
DIGEST_FLAGS = {"experiment": "--manifest-sha256", "design": "--file-sha256",
                "agent": "--artifact-sha256"}

VERB_SPECS = (
    VerbSpec("experiment", "rename", positional="<name> <new-name>",
             purpose="Preview renaming a draft study; repeat with its reviewed "
                     "--manifest-sha256 and --yes to rename it. Runs keep the "
                     "name they recorded.",
             boolean_flags=frozenset({"--yes"}),
             value_flags=frozenset({"--manifest-sha256"})),
    VerbSpec("experiment", "delete", positional="<name>",
             purpose="Preview deleting a draft study; repeat with its reviewed "
                     "--manifest-sha256 and --yes to move it into "
                     "experiments/.trash-<time>/. Nothing in runs/ changes.",
             boolean_flags=frozenset({"--yes"}),
             value_flags=frozenset({"--manifest-sha256"})),
    VerbSpec("design", "rename", positional="<name> <new-name>",
             purpose="Preview renaming a template; repeat with its reviewed "
                     "--file-sha256 and --yes to rename it. Studies created "
                     "from it are unchanged.",
             boolean_flags=frozenset({"--yes"}),
             value_flags=frozenset({"--file-sha256"})),
    VerbSpec("design", "delete", positional="<name>",
             purpose="Preview deleting a template; repeat with its reviewed "
                     "--file-sha256 and --yes to move it into "
                     "templates/.trash-<time>/. Studies created from it are "
                     "unchanged.",
             boolean_flags=frozenset({"--yes"}),
             value_flags=frozenset({"--file-sha256"})),
    VerbSpec("agent", "delete", positional="<path>",
             purpose="Preview deleting an agent no study uses; repeat with its "
                     "reviewed --artifact-sha256 and --yes to move it into a "
                     ".trash folder. Refused while any study uses it.",
             boolean_flags=frozenset({"--yes"}),
             value_flags=frozenset({"--artifact-sha256"})),
)
LABELS = frozenset(spec.label for spec in VERB_SPECS)
EXPERIMENT_VERBS = frozenset(s.verb for s in VERB_SPECS if s.family == "experiment")


def handles(invocation) -> bool:
    return invocation.spec is not None and invocation.spec.label in LABELS


def run(invocation) -> CLIResult:
    from ..client_cli import ClientRefusal
    from ..cli_envelope import next_action
    from . import authoring_files as files, housekeeping

    spec, args = invocation.spec, invocation.positionals
    family, verb = spec.family, spec.verb
    digest_flag = DIGEST_FLAGS[family]
    wanted = 2 if verb == "rename" else 1
    if len(args) != wanted or any(len(v) != 1 for k, v in invocation.flags.items() if k != "--yes"):
        raise ClientRefusal(code="usage", reason=f"usage: steerlab {spec.label} {spec.positional} "
                            f"[{digest_flag} <sha256> --yes]", repair_action=f"steerlab {spec.label} --help")
    digest, confirmed = invocation.one(digest_flag), invocation.has("--yes")
    if confirmed and digest is None:
        raise ClientRefusal(code="usage", reason=f"--yes needs the reviewed {digest_flag} from the preview.",
                            repair_action=f"steerlab {spec.label} {' '.join(args)}  (preview it, then "
                                          f"repeat with {digest_flag} <digest> --yes)")
    root = files.root_path()
    key = {"experiment": "manifestFileSHA256", "design": "designFileSHA256", "agent": "artifactFileSHA256"}[family]
    try:
        if family == "experiment" and verb == "rename":
            review = (housekeeping.rename_study(args[0], args[1], root=root, expected=digest) if confirmed
                      else housekeeping.review_study_rename(args[0], args[1], root=root))
        elif family == "experiment":
            review = (housekeeping.delete_study(args[0], root=root, expected=digest) if confirmed
                      else housekeeping.review_study_delete(args[0], root=root))
        elif verb == "rename":
            review = (housekeeping.rename_template(args[0], args[1], root=root, expected=digest) if confirmed
                      else housekeeping.review_template_rename(args[0], args[1], root=root))
        elif family == "design":
            review = (housekeeping.delete_template(args[0], root=root, expected=digest) if confirmed
                      else housekeeping.review_template_delete(args[0], root=root))
        else:
            review = (housekeeping.delete_agent(args[0], root=root, expected=digest) if confirmed
                      else housekeeping.review_agent_delete(args[0], root=root))
    except housekeeping.Usage as exc:
        raise ClientRefusal(code="usage", reason=exc.reason, repair_action=exc.repair_action) from exc
    except housekeeping.TemplateRefusal as exc:
        raise ClientRefusal(code=exc.code, reason=exc.reason, repair_action=exc.repair_action,
                            state="blocked" if exc.malformed else "refused") from exc
    except housekeeping.Refusal as exc:
        payload = {"path": exc.path}
        if exc.code == housekeeping.AGENT_IN_USE_CODE:
            payload["usedBy"] = exc.users
        raise ClientRefusal(code=exc.code, reason=exc.reason, repair_action=exc.repair_action,
                            state="refused", payload=payload) from exc
    except FileNotFoundError as exc:
        if family == "design":
            raise ClientRefusal(code="designNotFound", state="notFound",
                                reason=f"There is no template named '{args[0]}' in this workspace.",
                                repair_action="steerlab design list  (the templates this workspace holds)") from exc
        raise
    if digest is not None and not confirmed and digest != review[key]:
        raise ClientRefusal(code="staleManifest", gate="staleManifest", state="refused",
                            reason="The file changed after the digest you gave was read.",
                            repair_action="Read the new preview, show the researcher what changed, and use its digest.")
    subject = {"experiment": "study", "design": "template", "agent": "agent"}[family]
    if review["applied"]:
        line = (f"Renamed {subject} '{review['name']}' to '{review['newName']}'." if verb == "rename"
                else f"Moved {subject} '{review['name']}' to {review['destination']}/.")
        print(line)
        for effect in review["effects"]:
            if not effect.startswith("Moves "):
                print(effect)
        return CLIResult(message=line, changed=True, payload=review)
    confirm = f"{family} {verb} {' '.join(args)} {digest_flag} {review[key]} --yes"
    review["confirmCommand"] = f"steerlab {confirm}"
    print("Preview — nothing has changed yet.")
    for effect in review["effects"]:
        print("  " + effect)
    for advisory in review["advisories"]:
        print("  note: " + advisory)
    print(f"To apply, after the researcher agrees: steerlab {confirm}")
    return CLIResult(
        message=f"Preview of {verb} for {subject} '{review['name']}'; nothing changed.",
        payload=review,
        next_action=next_action(confirm, missing_permission_flags=[digest_flag, "--yes"],
                                detail="Show the researcher what will change and apply only after they agree."))
