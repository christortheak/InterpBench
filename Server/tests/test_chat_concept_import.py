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
