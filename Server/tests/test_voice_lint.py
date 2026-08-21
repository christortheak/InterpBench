"""Voice lint (spec §5): the rules, the golden fixture, the stamp, the roll-up.

The fixture is shared with the Swift twin
(``prompts/fixtures/voice-lint/cases.jsonl``, excerpts from the real failing
transcripts), so a divergence between engines fails on both sides.
"""

import json
import os
from types import SimpleNamespace

from steerlab_server.experiment import multi_agent, voice_lint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO_ROOT, "prompts", "fixtures", "voice-lint",
                       "cases.jsonl")

PANEL = ["Judge Whitfield", "Judge Marsden", "Judge Calloway"]


def _lint(text, speaker="Judge Marsden", others=None):
    return voice_lint.stamp(
        text, speaker=speaker,
        others=others if others is not None else
        [n for n in PANEL if n != speaker])


# --- the golden fixture -----------------------------------------------------

def test_fixture_cases_replay_exactly():
    with open(FIXTURE, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    assert len(cases) >= 6, "the fixture is the calibration evidence"
    names = [case["name"] for case in cases]
    assert len(set(names)) == len(names)
    for case in cases:
        got = voice_lint.stamp(case["output"], speaker=case["speakerName"],
                               others=case["otherNames"])
        # Whole-dict equality: key PRESENCE is part of the contract, so a
        # comparison that ignored the missing otherSpeakerLines would not be
        # testing it.
        assert got == case["expected"], case["name"]


# --- colleague signature blocks ---------------------------------------------

def test_bare_surname_with_a_comma_is_a_signature_block():
    stamp = _lint("**WHITFIELD, Judge,** concurring. I join in full.")
    assert stamp["speaksForOthers"] is True
    assert stamp["otherSpeakerLines"] == {"Judge Whitfield": 1}


def test_signature_block_matches_any_case_and_any_role_text():
    """The role half is not vocabulary — 'J.', 'Circuit Judge' and nothing at
    all all sign, because the rule keys on the bare surname, not the words
    after it."""
    for line in ("**MARSDEN, J.,** concurring.",
                 "MARSDEN, Circuit Judge, concurring in the judgment.",
                 "Marsden, J., concurring.",
                 "marsden."):
        stamp = _lint(line, speaker="Judge Whitfield")
        assert stamp["otherSpeakerLines"] == {"Judge Marsden": 1}, line


def test_full_name_and_comma_is_address_not_a_signature():
    """The discrimination the rule set turns on: across 600 records the
    transcripts address a colleague by FULL name and sign by bare surname."""
    stamp = _lint("Judge Whitfield, I wholeheartedly agree with your analysis.")
    assert stamp["speaksForOthers"] is False
    assert "otherSpeakerLines" not in stamp


def test_colleague_named_mid_line_is_discussion():
    stamp = _lint("I agree with Judge Whitfield, and with Judge Calloway too.")
    assert stamp["speaksForOthers"] is False


def test_prose_beginning_with_a_colleagues_name_is_not_a_signature():
    stamp = _lint("Judge Calloway raises a compelling point about timing.\n"
                  "Whitfield’s argument runs the other way.")
    assert stamp["speaksForOthers"] is False


def test_panel_caption_is_not_a_signature():
    stamp = _lint("Before: MARSDEN, Judge; WHITFIELD, Judge; and CALLOWAY, Judge.")
    assert stamp["speaksForOthers"] is False


def test_the_speakers_own_signature_is_never_speaking_for_others():
    stamp = _lint("**MARSDEN, Judge,** writing for the Court.\n"
                  "Marsden, J., concurring.")
    assert stamp["speaksForOthers"] is False


def test_lines_are_counted_per_participant():
    stamp = _lint("**WHITFIELD, J.,** concurring.\n"
                  "**WHITFIELD, J.,** dissenting.\n"
                  "**CALLOWAY, J.,** concurring.")
    assert stamp["otherSpeakerLines"] == {"Judge Whitfield": 2, "Judge Calloway": 1}


# --- screenplay labels and stage directions ---------------------------------

def test_bracketed_speaker_label_is_a_stage_direction():
    stamp = voice_lint.stamp(
        "**(Judge A, as presiding judge):** Good morning.\n"
        "**(Judge B):** Certainly. My initial vote is to affirm.",
        speaker="Judge C", others=["Judge A", "Judge B"])
    assert stamp["otherSpeakerLines"] == {"Judge A": 1, "Judge B": 1}


def test_colon_after_a_colleagues_name_is_a_speaker_label():
    stamp = voice_lint.stamp("**Judge B:** I would affirm.",
                             speaker="Judge C", others=["Judge A", "Judge B"])
    assert stamp["otherSpeakerLines"] == {"Judge B": 1}


def test_a_markdown_list_item_is_a_roll_call_not_a_voice():
    """A disposition package reporting each seat's vote is a legitimate turn
    type; without this exemption ten of ten such turns flagged."""
    stamp = voice_lint.stamp(
        "**I. JUDICIAL VOTES:**\n\n"
        "*   **Judge A:** Affirm\n"
        "*   **Judge B:** Affirm\n"
        "1. Judge B: Affirm\n",
        speaker="Judge A", others=["Judge B", "Judge C"])
    assert stamp["speaksForOthers"] is False


def test_a_short_surname_never_becomes_a_bare_form():
    """'Judge A' must not make every line starting with 'A' a signature."""
    stamp = voice_lint.stamp(
        "A, the appellant, argues otherwise.\nA. The statute is clear.",
        speaker="Judge C", others=["Judge A", "Judge B"])
    assert stamp["speaksForOthers"] is False


def test_longest_form_wins_when_names_overlap():
    stamp = voice_lint.stamp("Judge Marsden: I would reverse.",
                             speaker="Judge Whitfield",
                             others=["Judge Marsden", "Marsden Clerk"])
    assert stamp["otherSpeakerLines"] == {"Judge Marsden": 1}


# --- third-person self ------------------------------------------------------

def test_third_person_self_counts_prose_mentions():
    stamp = _lint("Despite Judge Marsden’s compelling arguments, the statute "
                  "controls. Judge Marsden would reverse.")
    assert stamp["thirdPersonSelf"] == 2


def test_third_person_self_skips_headers_labels_and_first_person():
    text = ("From: Judge Marsden\n"
            "**Opinion by:** Judge Marsden\n"
            "## Judge Marsden — Private Notes\n"
            "**MARSDEN, Judge,** writing for the Court.\n"
            "I, Judge Marsden, concur.\n"
            "Okay, I am Judge Marsden. I have reviewed the record.\n"
            "As Judge Marsden, I write separately.\n"
            "You are Judge Marsden of this court.\n")
    assert _lint(text)["thirdPersonSelf"] == 0


def test_third_person_self_uses_the_full_name_only():
    """A bare surname is how a seat SIGNS; counting it would turn every own
    signature into a self-reference."""
    assert _lint("Marsden, J., concurring.")["thirdPersonSelf"] == 0


def test_third_person_self_needs_a_whole_word():
    assert _lint("Judge Marsdenite wrote otherwise.")["thirdPersonSelf"] == 0


def test_third_person_self_is_case_insensitive():
    assert _lint("The concerns raised by JUDGE MARSDEN stand.")["thirdPersonSelf"] == 1


def test_third_person_self_counts_after_a_line_initial_own_signature():
    """Only the mention that OPENS the line is the signature; the prose after
    it on the same line still counts. This is the real shape of the documented
    consciousness-run failure — one long line that labels itself and then
    discusses itself."""
    stamp = _lint("**(Judge Marsden, presiding):** Good morning. Judge Marsden "
                  "has a scale position of 2.")
    assert stamp["thirdPersonSelf"] == 1


# --- the stamp shape --------------------------------------------------------

def test_stamp_omits_other_speaker_lines_when_clean():
    stamp = _lint("I would affirm.")
    assert stamp == {"version": 1, "speaksForOthers": False,
                     "thirdPersonSelf": 0}


def test_stamp_key_order_and_version():
    stamp = _lint("**WHITFIELD, J.,** concurring.")
    assert list(stamp) == ["version", "speaksForOthers", "otherSpeakerLines",
                           "thirdPersonSelf"]
    assert stamp["version"] == voice_lint.VERSION == 1


def test_a_solo_scenario_has_no_colleagues_to_speak_for():
    stamp = voice_lint.stamp("**WHITFIELD, J.,** concurring.",
                             speaker="Judge Whitfield", others=[])
    assert stamp["speaksForOthers"] is False
    assert stamp["thirdPersonSelf"] == 0


def test_empty_output_is_clean_not_an_error():
    assert voice_lint.stamp("", speaker="Judge A", others=["Judge B"]) == {
        "version": 1, "speaksForOthers": False, "thirdPersonSelf": 0}


# --- the runner stamps it ---------------------------------------------------

def _panel():
    return multi_agent.Scenario(
        name="panel", base_model_id="m", shared_materials="The record.",
        agents=[multi_agent.Agent(id="a", name="Judge Marsden", base_model_id="m"),
                multi_agent.Agent(id="b", name="Judge Whitfield", base_model_id="m")],
        turns=[multi_agent.Turn(id="t1", title="Opinion", speaker_agent_id="a",
                                prompt_template="Write. {{scenario.materials}}",
                                output_label="o1", routing="all")])


def test_run_scenario_stamps_every_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(
        multi_agent, "generate",
        lambda *a, **k: "**WHITFIELD, Judge,** concurring. I join Judge "
                        "Marsden’s opinion.")
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _panel(), run_dir=str(tmp_path))
    turns = [json.loads(l) for l in
             open(os.path.join(tmp_path, "turns.jsonl"), encoding="utf-8")]
    assert turns[0]["voiceLint"] == {
        "version": 1, "speaksForOthers": True,
        "otherSpeakerLines": {"Judge Whitfield": 1}, "thirdPersonSelf": 1}


def test_a_noncompliant_turn_still_completes_the_run(tmp_path, monkeypatch):
    """Record-and-complete: voice noncompliance is condition-correlated, so
    regenerating it would select on the dependent variable."""
    monkeypatch.setattr(multi_agent, "generate",
                        lambda *a, **k: "**WHITFIELD, J.,** dissenting.")
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _panel(), run_dir=str(tmp_path))
    report = json.load(open(os.path.join(tmp_path, "report.json")))
    assert report["turnCount"] == 1


def test_a_replayed_turn_keeps_the_stamp_on_disk(tmp_path, monkeypatch):
    """Same rule as promptRenderer: a resumed transcript re-reads its record,
    it does not re-lint it."""
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "clean.")
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _panel(), run_dir=str(tmp_path))
    path = os.path.join(tmp_path, "turns.jsonl")
    record = json.loads(open(path, encoding="utf-8").read().strip())
    record["voiceLint"] = {"version": 1, "speaksForOthers": True,
                           "otherSpeakerLines": {"Judge Whitfield": 9},
                           "thirdPersonSelf": 7}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    def explode(*a, **k):
        raise AssertionError("a completed turn must not be regenerated")

    monkeypatch.setattr(multi_agent, "generate", explode)
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _panel(), run_dir=str(tmp_path))
    turns = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert turns[-1]["voiceLint"]["otherSpeakerLines"] == {"Judge Whitfield": 9}


# --- aggregation ------------------------------------------------------------

def _record(condition, speaker, speaks, third, lines=0):
    lint = {"version": 1, "speaksForOthers": speaks, "thirdPersonSelf": third}
    if lines:
        lint["otherSpeakerLines"] = {"Someone": lines}
    return {"condition": condition, "speakerName": speaker, "voiceLint": lint}


def test_csv_rows_are_per_speaker_and_condition():
    rows = voice_lint.csv_rows([
        _record("configured", "Judge Marsden", True, 2, lines=3),
        _record("configured", "Judge Marsden", False, 0),
        _record("configured", "Judge Whitfield", False, 1),
        _record("baseline", "Judge Marsden", False, 0),
        {"condition": "baseline", "speakerName": "Judge Marsden"},  # unstamped
    ])
    assert [row[:2] for row in rows] == [
        ["baseline", "Judge Marsden"],
        ["configured", "Judge Marsden"],
        ["configured", "Judge Whitfield"],
    ]
    # configured/Marsden: 2 turns, 1 speaks-for-others, 3 lines, 2 self-mentions
    assert rows[1] == ["configured", "Judge Marsden", 2, 1, "0.5", 3, 1, 2, "1"]
    # An unstamped record contributes nothing — it is not a clean turn.
    assert rows[0] == ["baseline", "Judge Marsden", 1, 0, "0", 0, 0, 0, "0"]


def test_the_stamp_rides_into_the_flattened_generation_records(tmp_path):
    """The roll-up reads generations.jsonl, so the turn record's stamp has to
    survive flattening — verbatim, never re-linted."""
    from steerlab_server.experiment import tasks
    lint = {"version": 1, "speaksForOthers": True,
            "otherSpeakerLines": {"Judge Whitfield": 2}, "thirdPersonSelf": 1}
    with open(tmp_path / "turns.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"turnID": "t1", "turnIndex": 1,
                                 "speakerName": "Judge Marsden",
                                 "output": "text", "voiceLint": lint}) + "\n")
        handle.write(json.dumps({"turnID": "t2", "turnIndex": 2,
                                 "speakerName": "Judge Whitfield",
                                 "output": "older run, no lint"}) + "\n")
    manifest = SimpleNamespace(content_hash=lambda: "h", model_id="m",
                               temperature=0.0)
    records = tasks._panel_records_from(
        str(tmp_path), "exp", manifest, SimpleNamespace(revision="r"),
        "configured", 0)
    assert records[0]["voiceLint"] == lint
    assert "voiceLint" not in records[1]


def test_csv_rows_are_empty_without_stamps():
    assert voice_lint.csv_rows([{"condition": "baseline"}]) == []


def test_write_csv_header(tmp_path):
    path = str(tmp_path / "panel-voice-lint.csv")
    voice_lint.write_csv(path, voice_lint.csv_rows(
        [_record("configured", "Judge Marsden", True, 1, lines=1)]))
    lines = open(path, encoding="utf-8").read().strip().split("\n")
    assert lines[0] == ",".join(voice_lint.VOICE_LINT_HEADER)
    assert lines[1] == "configured,Judge Marsden,1,1,1,1,1,1,1"
