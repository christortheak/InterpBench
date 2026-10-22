"""The chat-format concept importer (``concept import-chat``).

``Tests/Fixtures/cross-engine/chat-concept-import.json`` is written by this
engine (``scripts/regenerate-cross-engine-fixtures.py``) and read by the Swift
suite (``ChatConceptImportTests``), which must write the same bytes for every
success case and give the same refusal for every other. The first test here
holds the fixture to this engine's current behaviour, so a change on this side
cannot leave Swift passing against stale bytes. The rest drive the client verb
against a temporary workspace. Every input is synthetic.
"""

import base64
import hashlib
import json
import os

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import chat_concept_import as chat
from steerlab_server.steering.stimulus_set import StimulusSet
from steerlab_server.steering.token_bank_downsampling import selected_indices

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                       "chat-concept-import.json")
REGENERATE = ("stale fixture — regenerate chat-concept-import.json with "
              "scripts/regenerate-cross-engine-fixtures.py and commit")


def test_the_cross_engine_fixture_is_current():
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)
    assert len(fixture["cases"]) >= 20, "the fixture's coverage shrank"
    codes = set()
    for case in fixture["cases"]:
        positive = (base64.b64decode(case["positiveBase64"])
                    if "positiveBase64" in case else case["positive"].encode())
        try:
            prepared = chat.prepare(
                case["name"], positive, case["negative"].encode(),
                positive_file=case["positiveFile"],
                negative_file=case["negativeFile"],
                validation_pairs=case["validationPairs"], seed=case["seed"],
                created_at=fixture["createdAt"])
        except chat.ChatImportRefusal as refusal:
            assert case.get("refusal") == {
                "code": refusal.code, "reason": refusal.reason,
                "repairAction": refusal.repair_action,
                "state": refusal.state}, f"{case['label']}: {REGENERATE}"
            codes.add(refusal.code)
            continue
        assert "files" in case, f"{case['label']}: {REGENERATE}"
        assert [{"name": name, "text": data.decode()}
                for name, data in prepared["files"]] == case["files"], \
            f"{case['label']}: {REGENERATE}"
        assert prepared["stimulusSetHash"] == case["stimulusSetHash"]
        assert prepared["validationHash"] == case["validationHash"]
        assert prepared["repeatedTexts"] == case["repeatedTexts"], \
            f"{case['label']}: {REGENERATE}"
        assert [{"code": code, "detail": detail}
                for code, detail in chat.advisories(prepared)] \
            == case["advisories"], f"{case['label']}: {REGENERATE}"
        assert chat.summary_lines(dict(
            prepared, directory=f"prompts/concepts/{prepared['name']}")) \
            == case["summaryLines"], f"{case['label']}: {REGENERATE}"
    # Every refusal the importer can give from bytes is in the fixture.
    assert codes >= {"chatRowUnreadable", "chatRowRoles", "rowCountMismatch",
                     "noUsablePairs", "validationTooLarge", "usage"}


def test_the_fixture_exercises_every_drop_reason_and_the_hold_out():
    with open(FIXTURE, encoding="utf-8") as handle:
        first = json.load(handle)["cases"][0]
    provenance = json.loads(next(f["text"] for f in first["files"]
                                 if f["name"] == "provenance.json"))
    assert all(provenance["dropped"][reason] for reason in
               ("promptMismatch", "identicalReplies", "emptyReply"))
    assert provenance["validation"]["pairs"] == 2
    assert provenance["sources"]["positive"]["systemMessages"] == 1
    # The NFC/NFD prompt pair is a MISMATCH: text is compared by code point.
    assert provenance["dropped"]["promptMismatch"] == [4]


def _row(prompt, reply):
    return json.dumps({"messages": [{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": reply}]})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    root = tmp_path / "workspace"
    (root / "prompts" / "concepts").mkdir(parents=True)
    positive = tmp_path / "claims.jsonl"
    negative = tmp_path / "denials.jsonl"
    positive.write_text("".join(_row(f"Prompt {i}", f"I feel {i}.") + "\n"
                                for i in range(1, 7)))
    negative.write_text("".join(_row(f"Prompt {i}", f"I compute {i}.") + "\n"
                                for i in range(1, 7)))
    return root, positive, negative


def _run(capsys, root, *args):
    code = client_cli.main(["--root", str(root), "concept", "import-chat",
                            *map(str, args), "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_the_verb_writes_a_concept_the_stimulus_loaders_read(workspace, capsys,
                                                             monkeypatch):
    root, positive, negative = workspace
    monkeypatch.setattr(chat.time, "time", lambda: 1_760_832_000.5)
    code, document, err = _run(capsys, root, "machine-experience",
                               "--positive", positive, "--negative", negative,
                               "--validation-pairs", 2, "--seed", 7)
    assert code == 0, err
    assert document["state"] == "ready" and document["changed"] is True
    assert document["verb"] == "concept import-chat"
    assert document["nextAction"]["verb"] == \
        "experiment attach <name> machine-experience"
    # Six different pairs: nothing repeats, so nothing is advised.
    assert "advisories" not in document
    assert document["result"]["repeatedTexts"] == {
        "negative": {"keptRepeats": 0, "validationInExtraction": 0},
        "positive": {"keptRepeats": 0, "validationInExtraction": 0}}
    directory = root / "prompts" / "concepts" / "machine-experience"
    assert sorted(os.listdir(directory)) == [
        "concept.json", "negative.jsonl", "positive.jsonl", "provenance.json",
        "validation.jsonl"]
    assert os.listdir(root / "prompts" / "concepts") == ["machine-experience"]

    # The draw is the token bank's, over the kept pairs, recorded by row.
    assert selected_indices(6, 2, 7) == [3, 5]
    stimuli = StimulusSet.from_directory(str(directory))
    assert list(stimuli.positive) == ["I feel 1.", "I feel 2.", "I feel 3.", "I feel 5."]
    assert list(stimuli.negative) == ["I compute 1.", "I compute 2.",
                                      "I compute 3.", "I compute 5."]
    assert document["result"]["stimulusSetHash"] == stimuli.hash
    rows = [json.loads(line) for line in
            (directory / "validation.jsonl").read_text().splitlines()]
    assert rows == [{"text": "I feel 4.", "expresses": True},
                    {"text": "I compute 4.", "expresses": False},
                    {"text": "I feel 6.", "expresses": True},
                    {"text": "I compute 6.", "expresses": False}]
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance == document["result"]["provenance"]
    assert provenance["validation"] == {"draw": chat.DRAW, "pairs": 2,
                                        "rows": [4, 6], "seed": 7}
    assert provenance["sources"]["positive"]["file"] == "claims.jsonl"
    assert provenance["sources"]["positive"]["sha256"] == \
        hashlib.sha256(positive.read_bytes()).hexdigest()
    assert '"createdAt" : "2025-10-19T00:00:00Z"' in \
        (directory / "concept.json").read_text()


def test_refusals_are_typed_and_write_nothing(workspace, capsys):
    root, positive, negative = workspace
    concepts = root / "prompts" / "concepts"
    (concepts / "taken").mkdir()

    code, document, err = _run(capsys, root, "taken", "--positive", positive,
                               "--negative", negative)
    assert code == 65 and document["state"] == "refused"
    assert document["error"]["code"] == "conceptExists"
    assert "already exists" in err
    assert os.listdir(concepts / "taken") == []

    code, document, _ = _run(capsys, root, "fresh", "--positive",
                             f"{positive}x", "--negative", negative)
    assert code == 66 and document["error"]["code"] == "notFound"
    assert document["error"]["repairAction"] == \
        "check the path to the positive chat file"

    for args in (("Fresh Name", "--positive", positive, "--negative", negative),
                 ("fresh", "--positive", positive),
                 ("fresh", "--positive", positive, "--negative", negative,
                  "--validation-pairs", 1),
                 ("fresh", "--positive", positive, "--negative", negative,
                  "--validation-pairs", 1, "--seed", chat.MAX_SEED + 1)):
        code, document, _ = _run(capsys, root, *args)
        assert code == 64 and document["state"] == "blocked", args
        assert document["error"]["repairAction"], args

    positive.write_text(_row("Prompt 1", "I feel 1.") + "\n")
    code, document, _ = _run(capsys, root, "fresh", "--positive", positive,
                             "--negative", negative)
    assert code == 65 and document["error"]["code"] == "rowCountMismatch"
    assert os.listdir(concepts) == ["taken"]


def test_the_output_rows_are_the_stimulus_shape():
    """One ``{"text": …}`` per line, with the space every concept file in the
    Demo Workspaces has — the shape both engines' stimulus loaders read."""
    prepared = chat.prepare(
        "probe", (_row("Q", "yes") + "\n").encode(),
        (_row("Q", "no") + "\n").encode(), positive_file="p.jsonl",
        negative_file="n.jsonl", created_at="2026-10-19T00:00:00Z")
    files = dict(prepared["files"])
    assert files["positive.jsonl"] == b'{"text": "yes"}\n'
    assert files["negative.jsonl"] == b'{"text": "no"}\n'


def test_line_separators_in_a_reply_keep_each_row_one_line(tmp_path):
    """U+0085, U+2028, and U+2029 are valid raw inside a JSON string, but this
    engine's stimulus reader splits lines with ``str.splitlines``, which breaks
    on them. The importer escapes them, so the concept loads on this engine as
    well as on the Mac, with the reply's text intact."""
    reply = "I notice\u2028a pause\u2029 and\u0085 then go on."
    positive = (_row("Prompt 1", reply) + "\n" + _row("Prompt 2", "I feel 2.") + "\n").encode()
    negative = (_row("Prompt 1", "I compute 1.") + "\n"
                + _row("Prompt 2", "I compute 2.") + "\n").encode()
    prepared = chat.prepare("separators", positive, negative,
                            positive_file="claims.jsonl", negative_file="denials.jsonl",
                            validation_pairs=0, seed=None, created_at="2026-10-19T00:00:00Z")
    directory = tmp_path / "separators"
    directory.mkdir()
    for name, data in prepared["files"]:
        assert not any(c in data.decode() for c in "\u0085\u2028\u2029"), name
        (directory / name).write_bytes(data)
    stimuli = StimulusSet.from_directory(str(directory))
    assert list(stimuli.positive) == [reply, "I feel 2."]


# --- repeated texts: a row hold-out is not a text hold-out -------------------
#
# Every count below is worked out by hand from the rows; the cross-engine
# fixture then holds the Swift twin to the same numbers and sentences.

#: The advisory, written out here rather than imported, so a change to the
#: sentence has to be made on purpose.
OVERLAP_ADVICE = (
    "{total} of the {texts} texts in validation.jsonl ({positive} positive, "
    "{negative} negative) also appear, exactly, in the same side's extraction "
    "file, so validation can read better than it would on unseen text. Pairs "
    "are held out by row, not by text. To hold out texts with no exact repeat "
    "in extraction, remove the repeated rows from the source files and import "
    "again under a new name, or import with no hold-out and write "
    "validation.jsonl from texts the extraction files do not contain.")

#: What provenance.json held before repeated texts were counted: the counts
#: are reported, never written, so the importer's bytes stay chat-pairs-v1.
PROVENANCE_KEYS = {"artifactType", "concept", "dropped", "importedAt",
                   "importer", "pairs", "rule", "schemaVersion", "sources",
                   "validation"}


def _prepare_rows(rows, *, validation_pairs=0, seed=None):
    """``chat.prepare`` over ``(prompt, positive reply, negative reply)``."""
    return chat.prepare(
        "repeats", "".join(_row(p, r) + "\n" for p, r, _ in rows).encode(),
        "".join(_row(p, r) + "\n" for p, _, r in rows).encode(),
        positive_file="claims.jsonl", negative_file="denials.jsonl",
        validation_pairs=validation_pairs, seed=seed,
        created_at="2026-10-19T00:00:00Z")


def _counts(prepared):
    """(positive keptRepeats, negative keptRepeats, positive
    validationInExtraction, negative validationInExtraction)."""
    sides = prepared["repeatedTexts"]
    return (sides["positive"]["keptRepeats"], sides["negative"]["keptRepeats"],
            sides["positive"]["validationInExtraction"],
            sides["negative"]["validationInExtraction"])


def _lines(prepared):
    return chat.summary_lines(dict(prepared, directory="prompts/concepts/repeats"))


def _written(prepared, name):
    return dict(prepared["files"])[name].decode()


def test_no_repeated_texts_counts_zero_and_advises_nothing():
    prepared = _prepare_rows(
        [(f"Prompt {i}", f"I feel {i}.", f"I compute {i}.") for i in range(1, 7)],
        validation_pairs=2, seed=7)
    assert _counts(prepared) == (0, 0, 0, 0)
    assert prepared["overlapAdvisory"] is None
    assert chat.advisories(prepared) == []
    # Only the two lines every import prints: what was imported, and where.
    assert [line.split(" ")[0] for line in _lines(prepared)] == ["imported", "wrote"]


def test_repeats_inside_extraction_are_counted_but_not_advised():
    # Positive "Warm." twice (rows 2, 3); negative "Dim." twice (rows 4, 5).
    # Seed 0 holds out row 1, whose texts appear nowhere else.
    prepared = _prepare_rows([
        ("Prompt 1", "Even.", "Odd."),
        ("Prompt 2", "Warm.", "Cold."),
        ("Prompt 3", "Warm.", "Chilly."),
        ("Prompt 4", "Bright.", "Dim."),
        ("Prompt 5", "Vivid.", "Dim."),
    ], validation_pairs=1, seed=0)
    assert prepared["provenance"]["validation"]["rows"] == [1]
    assert _counts(prepared) == (1, 1, 0, 0)
    assert prepared["overlapAdvisory"] is None
    assert chat.advisories(prepared) == []
    assert ("1 of the 5 kept positive replies and 1 of the 5 kept negative "
            "replies repeat an earlier reply on the same side, and every copy "
            "is kept") in _lines(prepared)


def test_one_pair_four_times_with_one_held_out_is_advised_not_refused():
    """The review's case: one pair four times over, one pair held out. Both
    validation texts are also extraction texts, and the import still
    succeeds, writes every copy, and keeps its provenance shape."""
    prepared = _prepare_rows([("Prompt", "I feel.", "I compute.")] * 4,
                             validation_pairs=1, seed=1)
    assert prepared["provenance"]["validation"]["rows"] == [2]
    assert _counts(prepared) == (3, 3, 1, 1)
    advice = OVERLAP_ADVICE.format(total=2, texts=2, positive=1, negative=1)
    assert prepared["overlapAdvisory"] == advice
    assert chat.advisories(prepared) == [("validationOverlapsExtraction", advice)]
    assert _lines(prepared)[-1] == advice
    # The claim itself, read off the written files.
    assert _written(prepared, "positive.jsonl") == '{"text": "I feel."}\n' * 3
    assert _written(prepared, "negative.jsonl") == '{"text": "I compute."}\n' * 3
    assert _written(prepared, "validation.jsonl") == (
        '{"text": "I feel.", "expresses": true}\n'
        '{"text": "I compute.", "expresses": false}\n')
    provenance = json.loads(_written(prepared, "provenance.json"))
    assert set(provenance) == PROVENANCE_KEYS
    assert provenance["importer"] == "chat-pairs-v1"


def test_repeats_on_both_sides_of_the_split_are_counted_by_code_point():
    # Seed 36 holds out rows 2, 4, and 8. Extraction keeps rows 1, 3, 5, 6, 7.
    #   positive: "Warm." (row 2) is row 1's and "Bright." (row 4) is row 3's,
    #     so 2 held-out texts are extraction texts; NFD "caf\u00e9" (row 8) is not
    #     NFC "caf\u00e9" (row 7). Repeats among the 8 kept: Warm., Bright. -> 2.
    #   negative: "Cold." (row 2) is row 1's; "Grey." and "Juice." are not, so
    #     1. Repeats among the 8 kept: Cold., and "Quiet." (rows 5, 6, both
    #     extraction) -> 2.
    prepared = _prepare_rows([
        ("Prompt 1", "Warm.", "Cold."),
        ("Prompt 2", "Warm.", "Cold."),
        ("Prompt 3", "Bright.", "Dim."),
        ("Prompt 4", "Bright.", "Grey."),
        ("Prompt 5", "Loud.", "Quiet."),
        ("Prompt 6", "Soft.", "Quiet."),
        ("Prompt 7", "caf\u00e9", "Tea."),
        ("Prompt 8", "cafe\u0301", "Juice."),
    ], validation_pairs=3, seed=36)
    assert prepared["provenance"]["validation"]["rows"] == [2, 4, 8]
    assert _counts(prepared) == (2, 2, 2, 1)
    advice = OVERLAP_ADVICE.format(total=3, texts=6, positive=2, negative=1)
    assert chat.advisories(prepared) == [("validationOverlapsExtraction", advice)]
    assert _lines(prepared)[-2:] == [
        "2 of the 8 kept positive replies and 2 of the 8 kept negative replies "
        "repeat an earlier reply on the same side, and every copy is kept",
        advice]


def test_the_verb_reports_an_overlapping_hold_out_as_an_advisory(tmp_path,
                                                                 capsys,
                                                                 monkeypatch):
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    root = tmp_path / "workspace"
    (root / "prompts" / "concepts").mkdir(parents=True)
    positive = tmp_path / "claims.jsonl"
    negative = tmp_path / "denials.jsonl"
    positive.write_text((_row("Prompt", "I feel.") + "\n") * 4)
    negative.write_text((_row("Prompt", "I compute.") + "\n") * 4)
    code, document, err = _run(capsys, root, "one-pair", "--positive", positive,
                               "--negative", negative,
                               "--validation-pairs", 1, "--seed", 1)
    advice = OVERLAP_ADVICE.format(total=2, texts=2, positive=1, negative=1)
    assert code == 0, err
    assert document["state"] == "okWithAdvisories" and document["changed"]
    assert document["advisories"] == [
        {"code": "validationOverlapsExtraction", "detail": advice}]
    assert document["result"]["repeatedTexts"] == {
        "negative": {"keptRepeats": 3, "validationInExtraction": 1},
        "positive": {"keptRepeats": 3, "validationInExtraction": 1}}
    assert (root / "prompts" / "concepts" / "one-pair" / "validation.jsonl").exists()


def test_the_fixture_covers_repeats_with_and_without_the_advisory():
    with open(FIXTURE, encoding="utf-8") as handle:
        cases = [case for case in json.load(handle)["cases"] if "files" in case]
    advised = [case for case in cases if case["advisories"]]
    repeats_only = [case for case in cases if not case["advisories"]
                    and case["repeatedTexts"]["positive"]["keptRepeats"]]
    assert advised and repeats_only
    assert {a["code"] for case in advised for a in case["advisories"]} == \
        {"validationOverlapsExtraction"}
