"""The rendered controller script's provenance, and serverd's boot signal.

Open-issues §1 field report (2026-08-20): the walltime self-chain is shell code
in ``Server/scripts/controller-job.sbatch.template``, a deploy refreshes the
template without re-rendering ``$STEERLAB_METADATA_ROOT/controller-job.sbatch``,
and the operator's standing ritual submits the RENDERED copy. serverd 47564632
therefore ran the chain fix's code for 24 h under an Aug-17 launching script
and left no successor.

These tests pin the three things that make that visible: the stamp the render
writes, the comparison every reader recomputes, and the boot warning that fires
exactly when the topology says controller + Slurm and the launching script
exported no chain marker.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess

import pytest

from steerlab_server import controller_render, site_qualify

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
CONTROLLER = os.path.join(SCRIPTS, "controller-job.sbatch.template")


def _template_sha() -> str:
    with open(CONTROLLER, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _rendered(text: str, tmp_path, name="controller-job.sbatch") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def _stamped(sha, *, rendered_at="2026-08-20T09:00:00Z", source="52a7176",
             chain=True) -> str:
    marker = controller_render.STAMP_MARKER
    body = (f"{marker} template=controller-job.sbatch.template sha256={sha} "
            f"renderedAt={rendered_at} source={source}\n"
            "# a comment\n")
    if chain:
        body += f'export {controller_render.CHAIN_ENV_VAR}="template-{sha}"\n'
    return "#!/usr/bin/env bash\n" + body


class _Profile:
    """The two ServerProfile fields the topology rule reads."""

    def __init__(self, launch_topology="batch", executor="slurm", profile="cluster"):
        self.launch_topology = launch_topology
        self.executor = executor
        self.profile = profile


# =============================================================================
# The template's half of the contract
# =============================================================================


def test_the_chain_is_template_side_and_the_template_stamps_itself():
    """The ledger's open question, asserted rather than remembered.

    Every mechanism of the self-chain is shell code in this file — which is
    exactly why a rendered copy from before the fix carries none of it, no
    matter how current the SERVER code it launches is.
    """
    text = open(CONTROLLER, encoding="utf-8").read()
    assert "chain_successor() {" in text
    assert "\ntrap chain_successor USR1\n" in text
    assert 'sbatch --export=NONE "$STEERLAB_CONTROLLER_SCRIPT"' in text
    # No Python-side half: nothing in the server traps USR1 or chains.
    assert controller_render.STAMP_MARKER in text
    for placeholder in controller_render.PLACEHOLDERS:
        assert placeholder in text
    export = (f'export {controller_render.CHAIN_ENV_VAR}='
              f'"{controller_render.CHAIN_VALUE_PREFIX}@TEMPLATE_SHA256@"')
    assert export in text
    # Beside the trap, not somewhere else: the marker is a claim about the
    # file that carries the chain.
    assert text.index(export) < text.index("\ntrap chain_successor USR1\n")


def test_the_rerender_command_matches_the_swift_literal():
    """One string, four surfaces. The Swift twin
    (``ClusterProvisioner.renderControllerCommand``) asserts this same
    literal, so the ledger, the qualify repair, the boot warning, and the
    runbook cannot drift into four near-identical commands."""
    assert controller_render.RERENDER_COMMAND == (
        "steerlab-cli cluster controller start --site <site> --render-only")


# =============================================================================
# Parsing
# =============================================================================


def test_parse_stamp_reads_every_field_and_refuses_placeholders():
    sha = "a" * 64
    fields = controller_render.parse_stamp(_stamped(sha))
    assert fields["sha256"] == sha
    assert fields["renderedAt"] == "2026-08-20T09:00:00Z"
    assert fields["source"] == "52a7176"
    # An UNRENDERED template: placeholders are the absence of a fact, never a
    # fact. The whole point is that a hand render cannot fake an identity.
    unrendered = open(CONTROLLER, encoding="utf-8").read()
    unresolved = controller_render.parse_stamp(unrendered)
    # The template's static `template=` label survives (it names the file, and
    # is true either way); every placeholder-valued field is DROPPED.
    assert set(unresolved) == {"template"}
    assert controller_render.parse_chain_marker(unrendered) is None
    assert controller_render.parse_stamp("#!/bin/sh\necho hi\n") == {}


def test_parse_chain_marker_reads_the_exported_value():
    sha = "b" * 64
    assert controller_render.parse_chain_marker(_stamped(sha)) == f"template-{sha}"
    assert controller_render.parse_chain_marker(_stamped(sha, chain=False)) is None


# =============================================================================
# The comparison
# =============================================================================


def test_a_script_rendered_from_this_template_is_current(tmp_path):
    sha = _template_sha()
    root = _rendered(_stamped(sha), tmp_path)
    report = controller_render.inspect_rendered_script(metadata_root=root)
    assert report["status"] == "current"
    assert report["templateSha256"] == sha
    assert report["stampedSha256"] == sha
    assert report["sourceRevision"] == "52a7176"


def test_a_script_from_a_different_template_is_stale_and_names_both(tmp_path):
    root = _rendered(_stamped("c" * 64, rendered_at="2026-08-17T04:00:00Z"),
                     tmp_path)
    report = controller_render.inspect_rendered_script(metadata_root=root)
    assert report["status"] == "stale"
    assert "cccccccccccc" in report["detail"]
    assert _template_sha()[:12] in report["detail"]
    assert "2026-08-17T04:00:00Z" in report["detail"]


def test_an_unstamped_script_is_stale_as_the_era_it_comes_from(tmp_path):
    """The 47564632 case exactly. It must not read as ``unknown``: its era is
    precisely the one whose rendered copies drop the chain."""
    root = _rendered("#!/usr/bin/env bash\n# SteerLab controller job\n", tmp_path)
    report = controller_render.inspect_rendered_script(metadata_root=root)
    assert report["status"] == "stale"
    assert "no render stamp" in report["detail"]


def test_a_matching_script_with_no_chain_export_is_stale(tmp_path):
    root = _rendered(_stamped(_template_sha(), chain=False), tmp_path)
    report = controller_render.inspect_rendered_script(metadata_root=root)
    assert report["status"] == "stale"
    assert controller_render.CHAIN_ENV_VAR in report["detail"]


def test_a_missing_artifact_is_absent_not_stale(tmp_path):
    report = controller_render.inspect_rendered_script(metadata_root=str(tmp_path))
    assert report["status"] == "absent"


def test_an_unreadable_template_is_unknown_never_stale(tmp_path):
    """Doctrine: an unproven fact never licenses an action. A wrong ``stale``
    sends an operator re-rendering over a working script, and the third time
    that happens they stop reading the finding."""
    root = _rendered(_stamped("d" * 64), tmp_path)
    empty = tmp_path / "no-code-here"
    empty.mkdir()
    report = controller_render.inspect_rendered_script(
        metadata_root=root, root=str(empty))
    assert report["status"] == "unknown"
    assert report["templatePath"] is None


def test_the_template_is_found_in_a_flattened_payload(tmp_path):
    flat = tmp_path / "payload" / "scripts"
    flat.mkdir(parents=True)
    (flat / "controller-job.sbatch.template").write_text("x", encoding="utf-8")
    assert controller_render.template_path(str(tmp_path / "payload")) == str(
        flat / "controller-job.sbatch.template")


# =============================================================================
# The boot signal
# =============================================================================


def test_the_boot_warning_fires_only_for_a_slurm_controller_with_no_marker():
    controller = _Profile()
    status = controller_render.boot_chain_status(controller, environ={})
    assert status["state"] == "unchained"
    lines = controller_render.boot_warning_lines(controller, environ={})
    assert lines and "controller self-chain" in lines[1]
    assert any(controller_render.RERENDER_COMMAND in line for line in lines)
    # …and nowhere else. A workstation, a local executor, and — the one that
    # would otherwise be a FALSE POSITIVE — a login-node cluster daemon
    # (topology `tunnel`), which `server_role` also calls "controller" but
    # which has no controller-job script and so no chain to miss. A false
    # positive is how an operator learns to skip the warning.
    for profile in (_Profile(launch_topology="local", executor="local",
                             profile="local"),
                    _Profile(launch_topology="batch", executor="local"),
                    _Profile(launch_topology="tunnel", executor="slurm",
                             profile="cluster")):
        assert not controller_render.is_controller_topology(profile)
        assert controller_render.boot_warning_lines(profile, environ={}) == []
        assert controller_render.boot_chain_status(
            profile, environ={})["state"] == "notApplicable"


def test_a_gpu_session_worker_never_warns(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    assert controller_render.boot_warning_lines(_Profile(), environ={}) == []


def test_a_marker_from_this_template_is_silent_and_a_foreign_one_warns():
    controller = _Profile()
    sha = _template_sha()
    good = {controller_render.CHAIN_ENV_VAR: f"template-{sha}"}
    assert controller_render.boot_chain_status(controller, good)["state"] == "chained"
    assert controller_render.boot_warning_lines(controller, good) == []

    foreign = {controller_render.CHAIN_ENV_VAR: "template-" + "e" * 64}
    status = controller_render.boot_chain_status(controller, foreign)
    assert status["state"] == "mismatched"
    assert any("eeeeeeeeeeee" in line for line in status["warnings"])
    assert any(controller_render.RERENDER_COMMAND in line
               for line in status["warnings"])


def test_an_unresolved_placeholder_marker_counts_as_absent():
    """A hand-rendered script leaves ``@TEMPLATE_SHA256@`` standing. That is
    not an identity, and must read as unchained rather than as a mismatch."""
    controller = _Profile()
    environ = {controller_render.CHAIN_ENV_VAR: "template-@TEMPLATE_SHA256@"}
    assert controller_render.boot_chain_status(
        controller, environ)["state"] == "unchained"


def test_the_warning_is_never_a_refusal():
    """A chain-less controller SERVES correctly — it just has to be cycled by
    hand. Refusing to start would cost the researcher a working daemon over a
    resubmission convenience, so the block says so in as many words."""
    lines = controller_render.boot_warning_lines(_Profile(), environ={})
    assert any("safe to keep serving" in line for line in lines)


# =============================================================================
# The qualify row
# =============================================================================


def test_qualify_declares_the_controller_script_row():
    assert "controllerScript" in site_qualify.CHECK_IDS
    spec = next(c for c in site_qualify.CHECKS if c.id == "controllerScript")
    # Contract text, written for a stranger qualifying their own site.
    assert "template" in spec.what
    assert "walltime" in spec.why or "self-chain" in spec.why


def _run_row(monkeypatch, metadata_root):
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(metadata_root))
    spec = next(c for c in site_qualify.CHECKS if c.id == "controllerScript")
    return spec.run(site_qualify.Context())


def test_the_qualify_row_passes_on_a_current_script(monkeypatch, tmp_path):
    root = _rendered(_stamped(_template_sha()), tmp_path)
    outcome = _run_row(monkeypatch, root)
    assert outcome.status == "pass"
    assert _template_sha()[:12] in outcome.observed


def test_the_qualify_row_warns_with_the_repair_on_a_stale_script(
        monkeypatch, tmp_path):
    root = _rendered(_stamped("f" * 64), tmp_path)
    outcome = _run_row(monkeypatch, root)
    assert outcome.status == "warn"
    assert controller_render.RERENDER_COMMAND in outcome.detail
    # Advisory by design — a stale script still serves.
    assert "still SERVES" in outcome.detail


def test_the_qualify_row_skips_where_nothing_is_rendered(monkeypatch, tmp_path):
    """A skip is a hole, and holes are counted — never a verdict. A login node
    or a workstation has nothing rendered and has failed nothing."""
    outcome = _run_row(monkeypatch, tmp_path)
    assert outcome.status == "skip"
    assert controller_render.RERENDER_COMMAND in outcome.detail


def test_a_stale_script_never_changes_the_qualify_verdict_to_failed(
        monkeypatch, tmp_path):
    root = _rendered(_stamped("f" * 64), tmp_path)
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(root))
    report = site_qualify.qualify(skip_model_fixtures=True)
    assert "controllerScript" in site_qualify.warning_ids(report)
    assert "controllerScript" not in site_qualify.failing_ids(report)


# =============================================================================
# End to end: the render the Swift command performs, mirrored in a real shell
# =============================================================================


@pytest.mark.parametrize("hasher", ["sha256sum", "shasum"])
def test_the_rendered_artifact_verifies_against_the_template_it_came_from(
        tmp_path, hasher):
    """The cross-engine anchor: render with the same shell the Swift command
    composes, then read the result back with THIS module. If either side's idea
    of the digest changes, this fails."""
    bash = "/bin/bash"
    meta = tmp_path / "meta"
    meta.mkdir()
    rendered = meta / "controller-job.sbatch"
    hash_clause = (f'sha256sum "{CONTROLLER}" 2>/dev/null'
                   if hasher == "sha256sum"
                   else f'shasum -a 256 "{CONTROLLER}" 2>/dev/null')
    command = (
        f'STEERLAB_TPL_SHA="$( {{ {hash_clause} || echo unknown; }} '
        "| awk '{print $1}' )\" && "
        'STEERLAB_RENDERED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ || echo unknown)" && '
        'STEERLAB_SOURCE_REV="$( { cat /nonexistent 2>/dev/null '
        '|| echo unknown; } | head -n 1 )" && '
        "{ printf '%s\\n' '#!/usr/bin/env bash' '#SBATCH --ntasks=1'; "
        f'tail -n +2 "{CONTROLLER}" '
        "| sed -e 's|@PYTHON@|/usr/bin/python3|g' -e 's|@PORT@|8080|g'"
        ' -e "s|@TEMPLATE_SHA256@|$STEERLAB_TPL_SHA|g"'
        ' -e "s|@RENDERED_AT@|$STEERLAB_RENDERED_AT|g"'
        ' -e "s|@SOURCE_REVISION@|$STEERLAB_SOURCE_REV|g"; } '
        f'> "{rendered}"')
    proc = subprocess.run([bash, "-c", command], text=True, capture_output=True,
                          check=False)
    assert proc.returncode == 0, proc.stderr

    report = controller_render.inspect_rendered_script(metadata_root=str(meta))
    assert report["status"] == "current", report["detail"]
    assert report["stampedSha256"] == _template_sha()
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                    report["renderedAt"] or "")
    # …and the exported marker the boot check reads is the same digest.
    text = rendered.read_text(encoding="utf-8")
    assert controller_render.parse_chain_marker(text) == (
        f"template-{_template_sha()}")
    # No unresolved placeholder survives onto an ACTIVE line.
    active = [ln for ln in text.splitlines()
              if ln.startswith("#SBATCH ") or not ln.lstrip().startswith("#")]
    assert not [ln for ln in active if re.search(r"@[A-Z0-9_]+@", ln)]
