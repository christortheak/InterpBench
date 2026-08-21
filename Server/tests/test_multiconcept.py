"""Grand-mean multi-concept corpus loader/saver + math (no model)."""

import json
import os

import pytest

from steerlab_server.experiment import multiconcept
from steerlab_server.steering import vector_math as vm


def _story(tmp, concept, texts, split="build"):
    d = os.path.join(str(tmp), "prompts", "emotions", concept)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "stories.jsonl"), "w", encoding="utf-8") as h:
        for i, t in enumerate(texts):
            h.write(json.dumps({"concept": concept, "topic": "t", "text": t,
                                "split": split, "id": f"{concept}-{i}"}) + "\n")


def test_list_and_load_corpus(tmp_path):
    _story(tmp_path, "fear", ["a", "b"])
    _story(tmp_path, "joy", ["c"])
    concepts = multiconcept.list_story_concepts(str(tmp_path))
    assert {c["concept"]: c["stories"] for c in concepts} == {"fear": 2, "joy": 1}
    rows, hashes = multiconcept.load_corpus(root=str(tmp_path))
    assert len(rows) == 3 and set(hashes) == {"fear", "joy"}
    # subset
    rows2, _ = multiconcept.load_corpus(["fear"], root=str(tmp_path))
    assert [c for c, _ in rows2] == ["fear", "fear"]


def test_save_stories_roundtrip(tmp_path):
    info = multiconcept.save_stories(
        "anger", [{"text": "grr", "topic": "x", "split": "build"},
                  {"text": "  ", "split": "build"}], root=str(tmp_path))
    assert info["stories"] == 1 and info["hash"]
    rows = multiconcept.read_stories("anger", root=str(tmp_path))
    assert rows[0]["text"] == "grr" and rows[0]["concept"] == "anger"


def test_grand_mean_math_matches_definition():
    # vector(fear) = mean(fear rows) − mean(all rows)
    fear = [[2.0, 0.0], [4.0, 0.0]]   # mean [3,0]
    joy = [[0.0, 2.0]]                 # all mean [(2+4+0)/3, (0+0+2)/3] = [2, 0.667]
    direction = vm.grand_mean_difference(concept=fear, population=fear + joy)
    assert direction[0] == pytest.approx(1.0, abs=1e-4)       # 3 - 2
    assert direction[1] == pytest.approx(-0.6667, abs=1e-3)   # 0 - 0.667
