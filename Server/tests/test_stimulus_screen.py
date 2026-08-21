"""Independence-from-task screen: workspace-defined forbidden vocabulary.

Contract under test (Finding A, 2026-07-19): the screen is domain-agnostic —
the vocabulary is data, resolved --vocabulary → workspace file → shipped
judicial default — the output always names which vocabulary governed, and the
exit codes (0 clean / 1 findings / 2 usage) stay stable for the app's button.
"""

import json

import pytest

from steerlab_server.experiment import stimulus_screen


# The judicial default list is study DATA: it is excluded from the released
# tree (`scripts/export-allowlist.txt`), so the tests that exercise the
# fallback branch skip there rather than fail. The screen MECHANICS
# (loading, shape checks, regex escaping, resolution order, exit codes) are
# all covered against fixtures written here, which run everywhere.
needs_default = pytest.mark.skipif(
    not stimulus_screen.default_vocabulary_available(),
    reason="no judicial default list here (release tree) — it is study data; "
           "the screen resolves its vocabulary from --vocabulary or the "
           "workspace file, both covered by fixture-based tests")


def judicial_pattern():
    return stimulus_screen.load_vocabulary(
        stimulus_screen.default_vocabulary_path()).pattern


def write_concept(root, name, lines):
    concept = root / name
    concept.mkdir(parents=True)
    (concept / "positive.jsonl").write_text(
        "".join(json.dumps({"text": line}) + "\n" for line in lines))
    return concept


def write_vocabulary(path, lists):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lists": lists}))
    return path


# ── screening mechanics (judicial default pattern) ─────────────────────────

@needs_default
def test_screen_text_flags_and_passes():
    pattern = judicial_pattern()
    assert stimulus_screen.screen_text(
        "The judge handed down a harsh sentence.", pattern) == [
            "judge", "sentence"]
    assert stimulus_screen.screen_text(
        "The hiker felt a rising dread as fog swallowed the trail.",
        pattern) == []
    # Word-start prefix matching: 'courtyard' trips 'court' — over-broad by
    # design (a flagged line is a prompt for human review).
    assert stimulus_screen.screen_text(
        "They met in the courtyard.", pattern) == ["court"]


@needs_default
def test_screen_directory(tmp_path):
    concept = write_concept(tmp_path, "fear", [
        "Her heart pounded as the floor creaked.",
        "The defendant trembled before the jury.",
    ])
    (concept / "validation.jsonl").write_text("not json\n")
    findings = stimulus_screen.screen_directory(
        str(concept), judicial_pattern())
    flagged = [f for f in findings if f.get("terms")]
    assert len(flagged) == 1
    assert flagged[0]["file"] == "positive.jsonl"
    assert flagged[0]["line"] == 2
    assert "defendant" in flagged[0]["terms"] and "jury" in flagged[0]["terms"]
    errors = [f for f in findings if f.get("error")]
    assert errors and errors[0]["file"] == "validation.jsonl"


@needs_default
def test_clean_directory(tmp_path):
    concept = write_concept(tmp_path, "calm", [
        "Waves lapped gently at the shore."])
    assert stimulus_screen.screen_directory(
        str(concept), judicial_pattern()) == []


# ── vocabulary loading ─────────────────────────────────────────────────────

def test_load_vocabulary_flattens_and_dedupes(tmp_path):
    path = write_vocabulary(tmp_path / "vocab.json", {
        "b-second": {"note": "n", "terms": ["harbor", "anchor"]},
        "a-first": {"note": "n", "terms": ["anchor", "mast"]},
    })
    vocabulary = stimulus_screen.load_vocabulary(str(path))
    assert vocabulary.list_names == ("a-first", "b-second")
    assert sorted(vocabulary.terms) == ["anchor", "harbor", "mast"]


@pytest.mark.parametrize("payload", [
    {},                                            # no lists
    {"lists": {}},                                 # empty lists
    {"lists": {"x": {"terms": []}}},               # empty terms
    {"lists": {"x": {"note": "no terms key"}}},    # missing terms
    {"lists": {"x": {"terms": ["ok", 3]}}},        # non-string term
    ["not", "an", "object"],                       # wrong top-level type
])
def test_load_vocabulary_rejects_bad_shapes(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        stimulus_screen.load_vocabulary(str(path))


def test_terms_are_regex_escaped(tmp_path):
    path = write_vocabulary(tmp_path / "vocab.json", {
        "punct": {"note": "n", "terms": ["a.b"]}})
    pattern = stimulus_screen.load_vocabulary(str(path)).pattern
    assert stimulus_screen.screen_text("we saw a.b here", pattern) == ["a.b"]
    assert stimulus_screen.screen_text("we saw aXb here", pattern) == []


# ── resolution order + output labeling (through main) ──────────────────────

def test_custom_vocabulary_flags_custom_terms(tmp_path, capsys):
    concept = write_concept(tmp_path, "salt", [
        "The harbor lights burned all night.",
        "The judge said nothing.",  # judicial term — must NOT trip here
    ])
    vocab = write_vocabulary(tmp_path / "maritime.json", {
        "maritime": {"note": "sea-domain leakage", "terms": ["harbor"]}})
    code = stimulus_screen.main(
        [str(concept), "--vocabulary", str(vocab)])
    out = capsys.readouterr().out
    assert code == 1
    assert f"vocabulary file {vocab}" in out
    assert "harbor" in out
    assert "1 flagged line(s)" in out  # the judge line stays clean


def test_workspace_file_auto_discovered(tmp_path, capsys):
    workspace = tmp_path / "ws"
    concept = write_concept(workspace / "prompts" / "concepts", "salt", [
        "The anchor dragged along the seabed."])
    vocab = write_vocabulary(
        workspace / "prompts" / "screens" / "forbidden-vocabulary.json",
        {"maritime": {"note": "n", "terms": ["anchor"]}})
    code = stimulus_screen.main([str(concept)])
    out = capsys.readouterr().out
    assert code == 1
    assert f"workspace file {vocab}" in out
    assert "anchor" in out


@needs_default
def test_fallback_labels_itself_judicial(tmp_path, capsys):
    concept = write_concept(tmp_path, "fear", [
        "The defendant trembled before the jury."])
    code = stimulus_screen.main([str(concept)])
    out = capsys.readouterr().out
    assert code == 1
    assert "shipped judicial-study default" in out
    assert "defendant" in out


@needs_default
def test_clean_run_still_names_the_vocabulary(tmp_path, capsys):
    concept = write_concept(tmp_path, "calm", [
        "Waves lapped gently at the shore."])
    code = stimulus_screen.main([str(concept)])
    out = capsys.readouterr().out
    assert code == 0
    assert "shipped judicial-study default" in out
    assert "clean" in out


# ── exit-code contract (the Swift button parses these) ─────────────────────

def test_usage_errors_exit_2(tmp_path, capsys):
    assert stimulus_screen.main([]) == 2
    assert stimulus_screen.main(["--nonsense", str(tmp_path)]) == 2
    assert stimulus_screen.main([str(tmp_path), "--vocabulary"]) == 2
    assert stimulus_screen.main(["a", "b"]) == 2
    assert stimulus_screen.main([str(tmp_path / "missing-dir")]) == 2
    capsys.readouterr()


def test_unloadable_vocabulary_exits_2(tmp_path, capsys):
    concept = write_concept(tmp_path, "calm", ["Quiet morning."])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert stimulus_screen.main(
        [str(concept), "--vocabulary", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "cannot load vocabulary" in err


@pytest.mark.skipif(
    stimulus_screen.default_vocabulary_available(),
    reason="this checkout carries the default list — nothing to fall back from")
def test_absent_default_refuses_instead_of_screening_against_nothing(
        tmp_path, capsys):
    """Release-tree behaviour: with no --vocabulary, no workspace file, and no
    shipped default, the screen REFUSES (exit 2) and names the repair. A
    clean verdict against an empty word list would be worse than no screen."""
    concept = write_concept(tmp_path, "calm", ["Waves lapped at the shore."])
    code = stimulus_screen.main([str(concept)])
    err = capsys.readouterr().err
    assert code == 2
    assert "no forbidden-vocabulary list resolved" in err
    assert stimulus_screen.WORKSPACE_VOCABULARY_RELPATH in err
    assert "--vocabulary" in err
