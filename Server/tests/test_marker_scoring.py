"""Concept-marker scoring: one tokenization on both engines.

Marker words are matched against the text's words, which are runs of letters
and marks in any script, read in NFC, case-folded, NFC spelling; marker
characters are matched one code point at a time in the NFC text, with case.
Density is markers per word.

The expectations below are worked out by hand from those rules, not read
back from either engine. The shared fixture
``Tests/Fixtures/cross-engine/marker-scoring.json`` holds the same cases as
this engine scores them, and the Mac suite (``ScoringTests``) reads the same
file.
"""

from __future__ import annotations

import json
import os

import pytest

from steerlab_server.experiment import scoring
from steerlab_server.experiment.scoring import MarkerRubric

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "marker-scoring.json")

REGENERATE = ("stale fixture — re-run "
              "`Server/.venv.nosync/bin/python "
              "scripts/regenerate-cross-engine-fixtures.py` and commit")


def test_an_accented_word_is_one_word():
    # The defect: splitting on anything but a-z made "naïve" two words
    # ("na", "ve"), so the density was 1/3 here and 1/2 on the Mac.
    rubric = MarkerRubric.from_markers(["warm"], "")
    assert scoring.marker_tokens("naïve warm") == ["naïve", "warm"]
    assert rubric.count("naïve warm") == 1
    assert rubric.density("naïve warm") == 0.5


def test_accented_marker_words_match():
    # l / été / est / très / chaud: five words, three markers.
    rubric = MarkerRubric.from_markers(["été", "très", "chaud"], "")
    text = "L'été est très CHAUD."
    assert scoring.marker_tokens(text) == ["l", "été", "est", "très", "chaud"]
    assert rubric.count(text) == 3
    assert rubric.density(text) == 3 / 5


def test_normalization_and_case_never_split_a_match():
    # Decomposed text, and a decomposed upper-case marker word, both read as
    # the one precomposed word.
    assert MarkerRubric.from_markers(["été"], "").density("été") == 1.0
    rubric = MarkerRubric.from_markers(["ÉTÉ"], "")
    assert rubric.words == {"été"}
    assert rubric.density("Un été.") == 0.5


def test_full_case_folding_not_lower_casing():
    # straße folds to strasse, so all three spellings are the marker; lower
    # casing would match only the second.
    rubric = MarkerRubric.from_markers(["straße"], "")
    assert rubric.count("STRASSE straße Strasse") == 3
    assert rubric.density("STRASSE straße Strasse") == 1.0
    # A final sigma and a capital sigma fold to the same letter.
    assert MarkerRubric.from_markers(["λόγος"], "").count("ΛΌΓΟΣ λόγος") == 2


def test_separators():
    # Digits, punctuation, hyphens and apostrophes separate words:
    # warm / warm / warm / hearted / don / t — six words, three markers.
    rubric = MarkerRubric.from_markers(["warm"], "")
    text = "warm2warm, warm-hearted; don't"
    assert len(scoring.marker_tokens(text)) == 6
    assert rubric.density(text) == 3 / 6


def test_vowel_signs_stay_inside_a_word():
    # Devanagari vowel signs and the virama are marks, not letters.
    rubric = MarkerRubric.from_markers(["नमस्ते"], "")
    assert scoring.marker_tokens("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]
    assert rubric.density("नमस्ते दुनिया") == 0.5


def test_characters_are_code_points_with_case():
    # One word (the unspaced run), one character marker.
    assert MarkerRubric.from_markers(["勇気"], "勇").density("勇気がある。") == 1.0
    # A decomposed marker character in the file, decomposed text: two hits
    # over two words.
    assert MarkerRubric.from_markers([], "é").density("café café") == 1.0
    # É is not é: two hits over two words.
    assert MarkerRubric.from_markers([], "é").count("ÉTÉ été") == 2
    # A thumbs-up with a skin-tone modifier still contains the thumbs-up.
    assert MarkerRubric.from_markers([], "\U0001F44D").count("great \U0001F44D\U0001F3FD") == 1


def test_a_text_without_words_has_zero_density():
    rubric = MarkerRubric.from_markers([], "!")
    assert rubric.count("...!!!") == 3
    assert rubric.density("...!!!") == 0.0
    assert MarkerRubric.from_markers(["warm"], "").density("") == 0.0


def test_folding_is_locale_independent():
    # The dotted capital I folds to "i" plus a combining dot, so it does not
    # match the plain word (it would under a Turkish locale's rules).
    rubric = MarkerRubric.from_markers(["ince"], "")
    assert rubric.count("İnce ince") == 1


def test_a_rubric_built_in_code_is_normalized_too():
    rubric = MarkerRubric(words={"TRÈS"}, characters={"é"})
    assert rubric.words == {"très"}
    assert rubric.characters == {"é"}


def test_directory_loader(tmp_path):
    (tmp_path / "markers.json").write_text(
        json.dumps({"words": ["Été"], "characters": "é"}), encoding="utf-8")
    rubric = MarkerRubric.from_directory(str(tmp_path))
    assert rubric is not None
    assert rubric.words == {"été"} and rubric.characters == {"é"}
    (tmp_path / "markers.json").write_text('{"words": [], "characters": ""}',
                                           encoding="utf-8")
    assert MarkerRubric.from_directory(str(tmp_path)) is None


def _cases():
    with open(FIXTURE, encoding="utf-8") as handle:
        return [pytest.param(case, id=case["label"])
                for case in json.load(handle)["cases"]]


@pytest.mark.parametrize("case", _cases())
def test_the_shared_fixture_is_this_engines_reading(case):
    rubric = MarkerRubric.from_markers(case["words"], case["characters"])
    assert scoring.marker_tokens(case["text"]) == case["tokens"], REGENERATE
    assert rubric.count(case["text"]) == case["count"], REGENERATE
    assert rubric.density(case["text"]) == case["density"], REGENERATE
