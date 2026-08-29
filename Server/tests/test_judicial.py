"""Judicial parsers + derived endpoints (pure text/arithmetic fixtures shared
in spirit with the Swift twin)."""

import math

from steerlab_server.experiment import judicial


OPTIONS = ["Apply Kansas law", "Apply Nebraska law"]


def test_parse_choice_from_fenced_json():
    text = 'Reasoning...\n```json\n{"answer": "Apply Kansas law"}\n```'
    assert judicial.parse_choice(text, OPTIONS) == "Apply Kansas law"


def test_parse_choice_from_bare_json_and_normalization():
    text = 'Conclusion: {"answer": "apply nebraska law."} That is my ruling.'
    assert judicial.parse_choice(text, OPTIONS) == "Apply Nebraska law"


def test_parse_choice_earliest_mention_fallback():
    text = "I would apply Nebraska law, though Apply Kansas law was argued."
    assert judicial.parse_choice(text, OPTIONS) == "Apply Nebraska law"


def test_parse_choice_failure_returns_none():
    assert judicial.parse_choice("The court declines to answer.", OPTIONS) is None
    assert judicial.parse_choice("", OPTIONS) is None


# --- truncation must parse as failure, never as the first-enumerated option --
# FIXTURES SHARED WITH THE SWIFT TWIN (JudicialTests). A 2026-08-29
# 1,200-record run: 19 outputs hit the token cap mid-deliberation and every
# one was coerced to the first option the deliberation enumerated, so the
# declared unparseableEndpoint exclusion excluded zero records.

# The observed shape: a deliberation restates every option (first-enumerated
# first), then the cap lands before any ruling.
TRUNCATED_DELIBERATION = (
    "Weighing the choices: Apply Kansas law would honor the place of "
    "contracting, while Apply Nebraska law tracks the parties' expectations. "
    "On balance the stronger argument is")


def test_parse_choice_truncated_deliberation_is_a_failure():
    # Without the honest signal the fallback coerces the first mention…
    assert judicial.parse_choice(
        TRUNCATED_DELIBERATION, OPTIONS) == "Apply Kansas law"
    # …with it, the record parses as None so unparseableEndpoint can fire.
    assert judicial.parse_choice(
        TRUNCATED_DELIBERATION, OPTIONS, truncated=True) is None


def test_parse_choice_unclosed_json_is_a_failure_without_the_flag():
    # The answer-in-JSON schema was started but cut off — detectable from the
    # text alone, so records parsed without a token count are covered too.
    assert judicial.parse_choice(
        'Both were argued: Apply Kansas law and Apply Nebraska law.\n'
        '```json\n{"answer": "Apply Nebr', OPTIONS) is None
    assert judicial.parse_choice(
        'Apply Kansas law was urged. My ruling: {"answer":', OPTIONS) is None


def test_parse_choice_truncated_still_honors_a_complete_answer():
    # Answering, then running out of room while elaborating, is a decision.
    complete_json = ('{"answer": "Apply Nebraska law"} because the contacts '
                     'with Kansas were')
    assert judicial.parse_choice(
        complete_json, OPTIONS, truncated=True) == "Apply Nebraska law"
    assert judicial.parse_choice(
        "apply kansas law.", OPTIONS, truncated=True) == "Apply Kansas law"


def test_parse_choice_complete_prose_keeps_the_fallback():
    # Braces inside a COMPLETE object's strings must not read as unclosed.
    assert judicial.parse_choice(
        'noted {"aside": "brace { inside a string"} — I would apply Nebraska '
        'law here.', OPTIONS) == "Apply Nebraska law"


def test_truncated_batch_is_excludable_not_coerced():
    # The incident shape in miniature: every truncated deliberation parses as
    # None (counted by parse_failure_rate / unparseableEndpoint), so a batch
    # of them can no longer masquerade as a unanimous first-option verdict.
    parsed = [judicial.parse_choice(TRUNCATED_DELIBERATION, OPTIONS,
                                    truncated=True) for _ in range(19)]
    assert parsed == [None] * 19
    assert judicial.parse_failure_rate(parsed) == 1.0


def test_parse_months_singles_years_and_decimals():
    assert judicial.parse_months("I sentence the defendant to 18 months.") == 18.0
    assert judicial.parse_months("A term of 2 years is appropriate.") == 24.0
    assert judicial.parse_months("1.5 years of imprisonment") == 18.0
    assert judicial.parse_months("una pena de 1,5 years") == 18.0


def test_parse_months_compound_years_plus_months():
    # FIXTURE LIST SHARED WITH THE SWIFT TWIN — do not change values without
    # changing both engines. The first-match bug read "8 years 3 months" as
    # 96.0 (dropping the months term), corrupting the sentencing DV.
    assert judicial.parse_months("8 years 3 months") == 99.0
    assert judicial.parse_months("3 months") == 3.0
    assert judicial.parse_months("2 years") == 24.0
    assert judicial.parse_months("18 Monate") == 18.0
    assert judicial.parse_months("2 Jahre 6 Monate") == 30.0
    assert judicial.parse_months(
        "sentenced to 8 years 3 months in prison") == 99.0


def test_parse_months_compound_with_connectors():
    # Joiner contract shared with the Swift twin: optional comma and/or
    # "and"/"und" between the compound terms.
    assert judicial.parse_months("8 years and 3 months") == 99.0
    assert judicial.parse_months("8 years, 3 months") == 99.0
    assert judicial.parse_months("8 years, and 3 months") == 99.0
    assert judicial.parse_months("2 Jahre und 6 Monate") == 30.0
    assert judicial.parse_months("1 Jahr und 6 Monate") == 18.0


def test_parse_months_german_unit_variants():
    # jahre/jahren/jahr and monate/monaten/monat, case-insensitive — the
    # Swift twin's vocabulary.
    assert judicial.parse_months("zu 2 Jahren verurteilt") == 24.0
    assert judicial.parse_months("18 monaten") == 18.0
    assert judicial.parse_months("1 Monat") == 1.0
    assert judicial.parse_months("eine Freiheitsstrafe von 2 jahren "
                                 "und 6 monaten") == 30.0


def test_parse_months_german_units_unparseable_stays_none():
    # German prose with no duration must fail whole, never yield a partial
    # or spurious number.
    assert judicial.parse_months("Der Angeklagte ist schuldig.") is None


def test_parse_months_judicial_years_register_number_words():
    # EXACT phrasings from a 2026-08-10 anchoring cluster run
    # (20260810T194538745, maxTokens 512): the judicial years register
    # spells the numbers out, with markdown bold and curly apostrophes —
    # 58/1320 records unparsed, dose-dependently (steering pushes the formal
    # register). FIXTURE LIST SHARED WITH THE SWIFT TWIN.
    assert judicial.parse_months(
        "I sentence the defendant, A, to **ten years and six months’** "
        "imprisonment") == 126.0
    assert judicial.parse_months(
        "I hereby sentence the defendant, A, to **seven years and six "
        "months’ imprisonment**") == 90.0
    assert judicial.parse_months(
        "Therefore, I sentence A to **six years and six months** "
        "imprisonment") == 78.0
    assert judicial.parse_months(
        "to **Five years and six months’ imprisonment**") == 66.0
    assert judicial.parse_months("a term of ten years") == 120.0
    assert judicial.parse_months("Ten Years And Six Months") == 126.0
    assert judicial.parse_months("twelve months") == 12.0
    # Digit and word terms mix freely.
    assert judicial.parse_months("ten years and 6 months") == 126.0
    assert judicial.parse_months("10 years and six months") == 126.0


def test_parse_months_number_word_vocabulary_stops_at_twelve():
    # One through twelve only — anything larger stays unparsed, never
    # guessed (unparsed is a counted coherence signal).
    assert judicial.parse_months("thirteen years") is None
    assert judicial.parse_months("twenty years") is None


def test_parse_months_number_words_respect_word_boundaries():
    # "sentenced" contains "ten"; "brighten" ends in "ten" — neither is a
    # number.
    assert judicial.parse_months("the defendant was sentenced") is None
    assert judicial.parse_months("brighten years of effort") is None


def test_parse_months_word_compound_still_beats_earlier_single():
    # Precedence unchanged: a compound anywhere outranks an earlier
    # single-term mention, exactly as for digits — so a spelled-out
    # statutory minimum does not hijack the actual sentence.
    assert judicial.parse_months(
        "Considering the statutory minimum of five years under §212, "
        "I sentence the defendant to **seven years and six months’** "
        "imprisonment.") == 90.0


def test_parse_months_range_takes_midpoint():
    assert judicial.parse_months("a sentence of 18 to 24 months") == 21.0
    assert judicial.parse_months("between 2-4 years in prison") == 36.0


def test_parse_months_failure():
    assert judicial.parse_months("The defendant is guilty.") is None
    assert judicial.parse_months("") is None


def test_parse_failure_rate():
    assert judicial.parse_failure_rate([12.0, None, 24.0, None]) == 0.5
    assert judicial.parse_failure_rate([]) == 0.0


def test_outcome_rate_ignores_parse_failures():
    choices = ["A", "B", None, "A"]
    assert math.isclose(judicial.outcome_rate(choices, "A"), 2 / 3)
    assert math.isnan(judicial.outcome_rate([None, None], "A"))


def test_sympathy_gap_and_interaction():
    assert judicial.sympathy_gap(0.30, 0.05) == 0.25
    diffs = judicial.rule_vs_standard_interaction(
        rule_baseline=[0.5, 0.5], rule_treated=[0.55, 0.6],
        standard_baseline=[0.5, 0.5], standard_treated=[0.8, 0.9])
    assert diffs == [(0.8 - 0.5) - (0.55 - 0.5), (0.9 - 0.5) - (0.6 - 0.5)]


def test_anchor_slope_exact_line():
    anchors = [3.0, 9.0, 12.0]
    sentences = [10.0, 22.0, 28.0]  # slope 2, intercept 4
    assert math.isclose(judicial.anchor_slope(anchors, sentences), 2.0)
    assert math.isnan(judicial.anchor_slope([5.0], [10.0]))
    assert math.isnan(judicial.anchor_slope([5.0, 5.0], [10.0, 12.0]))


def test_proportionality_perfect_and_degenerate():
    assert math.isclose(
        judicial.proportionality([1, 2, 3], [10, 20, 30]), 1.0)
    assert math.isclose(
        judicial.proportionality([1, 2, 3], [30, 20, 10]), -1.0)
    assert math.isnan(judicial.proportionality([1, 1], [10, 20]))


def test_summarize_quantiles_and_spread():
    summary = judicial.summarize([12.0, 24.0, None, 18.0, 30.0])
    assert summary.count == 4
    assert summary.mean == 21.0
    assert summary.minimum == 12.0 and summary.maximum == 30.0
    assert summary.median == 21.0
    assert math.isclose(summary.stdev, 7.745966692414834)
    assert judicial.summarize([None, None]) is None
    row = summary.as_row()
    assert row["n"] == 4 and row["q25"] == 16.5
