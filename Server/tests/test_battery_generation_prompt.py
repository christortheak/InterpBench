"""``battery generation-prompt`` — the capability-battery authoring brief.

The brief's value is that it is ASSEMBLED from the contracts rather than
restated from memory: the format-2 schema comes from
:mod:`steerlab_server.experiment.battery`, and the rules it hands the drafter
are :mod:`steerlab_server.experiment.battery_lint`'s own thresholds and
finding codes. These tests pin that property — every threshold and every mode
name is asserted THROUGH the module constant, so moving a threshold without
the brief following is a red test rather than a brief that quietly lies.

The other three things asserted here: the worked example is present and is the
shipped template (not a paraphrase of it), the closing loop tells the drafter
to lint before pinning, and ``--avoid`` text reaches the output — an authoring
helper that silently drops the one study-specific argument it takes is worse
than no helper.
"""

import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import battery, battery_brief, battery_lint


def _squashed(text: str) -> str:
    """The brief is line-wrapped for reading, so a supplied phrase survives as
    words, not as one contiguous substring. Compare on collapsed whitespace."""
    return " ".join(text.split())


SEED_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "WorkspaceSeed", "prompts", "templates", "battery",
    "capability-battery-v2.template.jsonl")


# =============================================================================
# 1. The brief carries the real contract
# =============================================================================


def test_the_brief_states_the_format_two_header_contract(tmp_path):
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert f"batteryFormat {battery.FORMAT_CURRENT}" in text
    for key in ("batteryFormat", "scoring", "promptMode", "systemPrompt",
                "qwenThinkingEnabled", "maxTokens"):
        assert f'"{key}"' in text, f"the header key {key} is undocumented"
    assert str(battery.BATTERY_MAX_TOKENS) in text
    for mode in battery.SCORING_MODES:
        assert mode in text


def test_the_brief_states_the_item_schema_for_both_grading_modes(tmp_path):
    text = battery_brief.generation_prompt(root=str(tmp_path))
    # choiceProbability: options, at least 2, answer must be among them.
    assert '"options"' in text and '"answer"' in text and '"prompt"' in text
    assert "MUST appear verbatim" in text
    # generatedText: grading is REQUIRED and inference is refused.
    for mode in battery.GRADING_MODES:
        assert f'"{mode}"' in text, f"grading mode {mode} is undocumented"
    assert "MUST NOT be present" in text


def test_the_brief_carries_the_linters_own_thresholds(tmp_path):
    """Through the constants, so a moved threshold moves the brief."""
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert f"fewer than {battery_lint.MIN_ITEMS} items" in text
    assert (f"fewer than {battery_lint.MIN_DISCRIMINATIVE_OPTIONS} options"
            in text)
    assert f"{battery_lint.MAX_OPTION_LENGTH_RATIO:g}x apart" in text


def test_the_brief_names_the_ways_a_draft_can_be_blocked(tmp_path):
    text = battery_brief.generation_prompt(root=str(tmp_path)).lower()
    assert "blockers" in text and "warnings" in text
    # The two schema blockers a drafter is most likely to trip.
    assert "not among" in text                       # answer ∉ options
    assert "no header line" in text                  # the legacy format
    # And the coupling the whole repair was about.
    assert "response-format instruction" in text


def test_the_brief_says_a_battery_is_a_control(tmp_path):
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert "CONTROL" in text and "HOLD STILL" in text


# =============================================================================
# 2. The worked example
# =============================================================================


def test_the_brief_embeds_the_shipped_template(tmp_path):
    """Not a paraphrase: the example a drafter copies is the file the
    workspace ships, and the brief says where it came from."""
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert battery_brief.TEMPLATE_PATH in text
    with open(SEED_TEMPLATE, encoding="utf-8") as handle:
        template = handle.read()
    assert template.strip() in text


def test_the_inlined_fallback_has_not_drifted_from_the_shipped_template():
    """The fallback exists for a tree with no templates directory. It is a
    COPY, so it needs a gate — otherwise it becomes the stale example nobody
    notices is stale."""
    with open(SEED_TEMPLATE, encoding="utf-8") as handle:
        assert handle.read() == battery_brief._TEMPLATE_FALLBACK


def test_a_workspace_template_wins_over_the_inlined_copy(tmp_path):
    """A researcher who edited their own template must see their edit."""
    path = os.path.join(str(tmp_path), *battery_brief.TEMPLATE_PATH.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"batteryFormat": 2, "description": "MY OWN TEMPLATE"}\n'
                     '{"id": "x", "prompt": "p", "answer": "a", '
                     '"options": ["a", "b", "c"]}\n')
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert "MY OWN TEMPLATE" in text
    assert "arith-sum" not in text                    # the shipped one is gone


def test_the_example_the_brief_ships_actually_loads_and_lints(tmp_path):
    """The worked example must satisfy the contract it is an example OF."""
    rel = "prompts/batteries/example.jsonl"
    path = os.path.join(str(tmp_path), *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(battery_brief._TEMPLATE_FALLBACK)
    spec = battery.load_spec(rel, str(tmp_path))
    assert spec.format_version == battery.FORMAT_CURRENT and spec.isolated
    report = battery_lint.lint(rel, str(tmp_path))
    assert report.ok, [f.code for f in report.blockers]


# =============================================================================
# 3. Count and --avoid
# =============================================================================


def test_the_requested_count_is_what_the_brief_asks_for(tmp_path):
    text = battery_brief.generation_prompt(37, root=str(tmp_path))
    assert "Produce 37 items" in text
    assert "then 37 item lines" in text


def test_the_default_count_clears_the_linters_floor():
    assert battery_brief.DEFAULT_ITEM_COUNT >= battery_lint.MIN_ITEMS


def test_a_count_below_the_floor_says_so_instead_of_pretending(tmp_path):
    text = battery_brief.generation_prompt(4, root=str(tmp_path))
    assert "fewItems" in text and str(battery_lint.MIN_ITEMS) in text
    # And the floor-clearing default carries no such note.
    assert "fewItems warning" not in battery_brief.generation_prompt(
        root=str(tmp_path))


def test_avoid_text_becomes_a_rule_in_the_brief(tmp_path):
    avoid = "tide tables, harbour depths, and anything a pilot would consult"
    text = battery_brief.generation_prompt(avoid=avoid, root=str(tmp_path))
    assert avoid in _squashed(text)
    assert "STAY OUT OF THE STUDY'S DOMAIN" in text


def test_no_avoid_text_still_states_the_rule_and_says_it_was_not_supplied(
        tmp_path):
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert "STAY OUT OF THE STUDY'S DOMAIN" in text
    assert "was not supplied" in text


def test_the_brief_is_concept_agnostic(tmp_path):
    """The ENGINE assumes no domain: with no --avoid, nothing in the output
    may name a study family. (Item content in the worked example is marked as
    an example and is deliberately generic.)"""
    text = battery_brief.generation_prompt(root=str(tmp_path)).lower()
    for word in ("judicial", "judge", "court", "sentencing", "legal",
                 "defendant"):
        assert word not in text, f"the brief presumes a domain: {word!r}"


# =============================================================================
# 4. The closing loop
# =============================================================================


def test_the_brief_closes_on_lint_then_pin(tmp_path):
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert "prompts/batteries/" in text
    assert "steerlab-server battery lint" in text
    lint_at = text.index("steerlab-server battery lint")
    pin_at = text.lower().index("pin it in an experiment manifest")
    assert lint_at < pin_at, "the brief must say lint BEFORE pin"


# =============================================================================
# 5. The CLI verb
# =============================================================================


def test_cli_prints_the_brief_on_stdout(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "battery",
                     "generation-prompt"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("CAPABILITY BATTERY — AUTHORING BRIEF")
    assert out == battery_brief.generation_prompt(root=str(tmp_path))


def test_cli_honours_count_and_avoid(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "battery", "generation-prompt",
                     "--count", "12", "--avoid", "tide tables"]) == 0
    out = capsys.readouterr().out
    assert "Produce 12 items" in out and "tide tables" in _squashed(out)


def test_cli_out_writes_the_file_and_keeps_stdout_clean(tmp_path, capsys):
    target = tmp_path / "brief.txt"
    assert cli.main(["--root", str(tmp_path), "battery", "generation-prompt",
                     "--out", str(target)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(target) in captured.err
    assert target.read_text(encoding="utf-8") == \
        battery_brief.generation_prompt(root=str(tmp_path))


def test_cli_out_reports_an_unwritable_path_as_a_failure(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "battery", "generation-prompt",
                     "--out", str(tmp_path / "nope" / "brief.txt")]) == 2
    assert "generation-prompt" in capsys.readouterr().err


@pytest.mark.parametrize("rest,fragment", [
    (["--count"], "needs a value"),
    (["--avoid"], "needs a value"),
    (["--out"], "needs a value"),
    (["--count", "many"], "integer"),
    (["--count", "0"], "at least 1"),
    (["--count", "-3"], "at least 1"),
    (["--nonsense", "x"], "unknown argument"),
    (["20"], "unknown argument"),
])
def test_cli_refuses_bad_flags_rather_than_printing_a_plausible_brief(
        tmp_path, capsys, rest, fragment):
    """A silently-ignored argument is the dangerous failure here: the brief it
    prints looks exactly right and asks for the wrong thing."""
    assert cli.main(["--root", str(tmp_path), "battery",
                     "generation-prompt", *rest]) == 64
    captured = capsys.readouterr()
    assert fragment in captured.err
    assert captured.out == ""


def test_an_unknown_battery_subverb_is_still_usage(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "battery", "draft"]) == 64
    err = capsys.readouterr().err
    assert "generation-prompt" in err and "lint" in err


def test_the_family_usage_lists_both_verbs(capsys):
    assert cli.main([]) == 64
    err = capsys.readouterr().err
    assert "battery lint" in err and "battery generation-prompt" in err


# =============================================================================
# 6. The route
# =============================================================================


def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app
    return TestClient(app)


def test_route_returns_the_same_text_the_cli_prints():
    body = _client().get("/api/battery/generation-prompt").json()
    assert body["count"] == battery_brief.DEFAULT_ITEM_COUNT
    assert body["prompt"] == battery_brief.generation_prompt(
        battery_brief.DEFAULT_ITEM_COUNT)


def test_route_honours_count_and_avoid():
    resp = _client().get("/api/battery/generation-prompt",
                         params={"count": 15, "avoid": "tide tables"})
    body = resp.json()
    assert body["count"] == 15
    assert "Produce 15 items" in body["prompt"]
    assert "tide tables" in _squashed(body["prompt"])


def test_route_is_a_read_and_therefore_not_swept_up_by_the_mutating_rule():
    """The WP-S classification reason, stated as a test: it is a GET, so it
    needs no entry in the mutating-route allowlist."""
    from steerlab_server.api.app import (_OPEN_MUTATING_PATHS,
                                         request_is_privileged)
    path = "/api/battery/generation-prompt"
    assert not request_is_privileged("GET", path)
    assert path not in _OPEN_MUTATING_PATHS
