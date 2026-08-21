"""The drift gate for the server's generated regions of
``docs/CLI-REFERENCE.md``, and the ``--help`` surface they share a source with
(WP0-AGENT-SURFACE-AUDIT §5, §7 step 11).

Swift twin: ``Tests/ExperimentKitTests/CLIReferenceGenerationTests.swift``.

The audit's finding this suite closes: the reference document claims to be
"read out of the dispatch code", nine drift instances said otherwise, and the
one section with ZERO verified drift was the one backed by a declarative table.
Generating the synopsis from :data:`cli_envelope.VERB_SPECS` and comparing the
committed bytes makes the class of drift unreproducible.

Deliberately NOT a build step: the text is COMMITTED so the document reads on
GitHub with no toolchain, and the repair for a red test is
``steerlab-server docs cli-reference --write`` followed by a commit.
"""

import os
import re

import pytest

from steerlab_server import cli, cli_envelope, cli_help, cli_reference

DOC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "CLI-REFERENCE.md")

DENYLIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts", "export-denylist.txt")


def _document() -> str:
    with open(DOC, encoding="utf-8") as handle:
        return handle.read()


# =============================================================================
# 1. The gate
# =============================================================================


def test_cli_reference_regions_match():
    drifted = cli_reference.drift(_document())
    assert not drifted, (
        f"docs/CLI-REFERENCE.md is out of date in {len(drifted)} region(s): "
        f"{', '.join(drifted)}. Repair: steerlab-server docs cli-reference "
        "--write, then commit.")


def test_rewriting_an_in_sync_document_changes_nothing():
    """``--check`` and ``--write`` must agree, or the gate is unrepairable."""
    document = _document()
    assert cli_reference.rewrite(document) == document


def test_every_declared_verb_is_in_exactly_one_region():
    """A verb added to the table and to no region would be documented nowhere
    while every test still passed."""
    counted: dict = {}
    for labels in cli_reference.REGIONS.values():
        for label in labels:
            counted[label] = counted.get(label, 0) + 1
    for spec in cli_envelope.VERB_SPECS:
        assert counted.get(spec.label) == 1, (
            f"{spec.label} appears in {counted.get(spec.label, 0)} region(s)")
    assert len(counted) == len(cli_envelope.VERB_SPECS)


def test_the_server_owns_only_server_regions():
    """Both engines write into one document; each rewrites only its own ids."""
    for region_id in cli_reference.REGIONS:
        assert region_id.startswith("server-")
    document = _document()
    for region_id in ("swift-workspace", "swift-cluster", "swift-remote"):
        assert cli_reference.begin_marker(region_id) in document
    rewritten = cli_reference.rewrite(document)
    for region_id in ("swift-workspace", "swift-cluster"):
        assert (cli_reference.extract(rewritten, region_id)
                == cli_reference.extract(document, region_id))


def test_a_missing_marker_refuses_rather_than_skipping():
    document = _document().replace(
        cli_reference.begin_marker("server-jobs"), "")
    with pytest.raises(ValueError):
        cli_reference.drift(document)


def test_the_printed_experiment_verb_list_is_complete():
    """Audit §1.3 D9: the two hand-written verb lists disagreed with each other
    and with the dispatch — the first omitted `pipeline` and `judge-worker`,
    the second omitted those plus `preflight-endpoints`. One list now, and it
    is asserted against the dispatch's own `verb == "…"` sites."""
    source = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(cli.__file__))), "steerlab_server", "cli.py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    body = text[text.index("def _experiment("):text.index("def _vectors(")]
    dispatched = set(re.findall(r'if verb == "([a-z-]+)"', body))
    assert dispatched <= set(cli.EXPERIMENT_VERBS), (
        f"dispatched but unlisted: {sorted(dispatched - set(cli.EXPERIMENT_VERBS))}")


# =============================================================================
# 2. `--help` renders from the same table
# =============================================================================


def test_every_verb_has_a_purpose_and_a_synopsis():
    for spec in cli_envelope.VERB_SPECS:
        assert spec.purpose, f"{spec.label} declares no purpose"
        page = cli_help.verb_text(spec)
        assert page.startswith(f"usage: steerlab-server {spec.label}")
        assert spec.purpose in page
        for flag in spec.declared_flags:
            assert flag in page, f"{spec.label} --help omits {flag}"
        assert page.endswith("\n")


def test_help_text_is_neutral():
    """Contract text, not prose: no institution, study, or person names. Same
    rule the release scanner applies — one case-insensitive regex per line,
    word-boundary anchored unless the line opens with ``raw:``.

    Research-tree-only BY DESIGN: the name-bearing denylist must never ship
    (WP-P split), so in the release tree this gate has no input and skips —
    there, neutrality was already enforced at export time by the scanner,
    and the shipped public-tier scan carries the rest."""
    if not os.path.isfile(DENYLIST):
        pytest.skip("no export denylist here (release tree) — neutrality is "
                    "the export-time scanner's job")
    patterns = []
    with open(DENYLIST, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line[4:] if line.startswith("raw:")
                            else rf"\b{line}\b")
    pages = [cli_help.verb_text(spec) for spec in cli_envelope.VERB_SPECS]
    pages += [cli_help.family_text(family)
              for family in sorted({s.family for s in cli_envelope.VERB_SPECS})]
    pages += [cli_reference.body(region) for region in cli_reference.REGIONS]
    for page in pages:
        for pattern in patterns:
            assert not re.search(pattern, page, re.IGNORECASE), (
                f"a generated page matches the denylist pattern {pattern!r}")


def test_help_runs_nothing_and_exits_zero(tmp_path, monkeypatch, capsys):
    """`--help` is answered before the verb's own positional requirements: a
    caller asking what the arguments are must not have to supply them first."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "run", "--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: steerlab-server experiment run <name>")
    assert "--dtype <dtype>" in out
    # Nothing was created: no experiment named `--help`, no run directory.
    assert not os.path.isdir(os.path.join(str(tmp_path), "runs"))


def test_family_help_lists_the_agent_path_verbs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "--help"]) == 0
    out = capsys.readouterr().out
    assert "promote" in out and "analyze" in out
    # And says out loud that the family is bigger than the agent path.
    assert "run unchanged" in out


def test_help_in_json_mode_is_one_document(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "promote", "--help", "--json"]) == 0
    import json
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready"
    assert envelope["verb"] == "experiment promote"
    assert envelope["result"]["positional"] == "<name> <concept>"
    assert any(entry["flag"] == "--cell" for entry in envelope["result"]["flags"])


def test_help_is_declared_and_neighbouring_typos_are_still_64(
        tmp_path, monkeypatch, capsys):
    """The one parsing change of step 11, stated as a test."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "analyze", "--help"]) == 0
    capsys.readouterr()
    assert cli.main(["experiment", "analyze", "demo", "--hepl"]) == 64
    assert "--help" in capsys.readouterr().err


# =============================================================================
# 3. The generator verb
# =============================================================================


def test_docs_verb_checks_the_committed_document(capsys):
    assert cli.main(["docs", "cli-reference", "--check"]) == 0
    assert "match the verb table" in capsys.readouterr().out


def test_docs_verb_refuses_on_drift(tmp_path, capsys):
    drifted = _document().replace(
        "steerlab-server jobs list", "steerlab-server jobs list [--nonsense]")
    path = tmp_path / "drifted.md"
    path.write_text(drifted, encoding="utf-8")
    assert cli.main(["docs", "cli-reference", "--check", "--path",
                     str(path)]) == 65
    err = capsys.readouterr().err
    assert "DRIFT: region server-jobs" in err
    assert "--write" in err


def test_docs_verb_writes_only_its_own_regions(tmp_path):
    document = _document()
    drifted = document.replace(
        "steerlab-server jobs list", "steerlab-server jobs list [--nonsense]")
    path = tmp_path / "doc.md"
    path.write_text(drifted, encoding="utf-8")
    assert cli.main(["docs", "cli-reference", "--write", "--path",
                     str(path)]) == 0
    repaired = path.read_text(encoding="utf-8")
    assert cli_reference.drift(repaired) == []
    for region_id in ("swift-workspace", "swift-cluster"):
        assert (cli_reference.extract(repaired, region_id)
                == cli_reference.extract(document, region_id))


def test_docs_usage_is_64(capsys):
    assert cli.main(["docs"]) == 64
    assert "cli-reference" in capsys.readouterr().err
