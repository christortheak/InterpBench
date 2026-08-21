"""Claude proposal parsing (no network) + concept deletion."""

import os

from steerlab_server.experiment import authoring, proposals


def test_parse_pairs_tolerant_of_prose_and_fences():
    text = (
        "Here are your pairs:\n```\n"
        '{"positive": "I am thrilled", "negative": "I am calm"}\n'
        '{"positive": "what joy", "negative": "how dull"}\n'
        "```\nHope that helps!")
    pairs = proposals.parse_pairs(text)
    assert len(pairs) == 2
    assert pairs[0] == {"positive": "I am thrilled", "negative": "I am calm"}


def test_generate_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert proposals.available() is False
    import pytest
    with pytest.raises(RuntimeError):
        proposals.generate_pairs("fear")


def test_delete_removes_datasets_only(tmp_path):
    authoring.create_concept("temp", root=str(tmp_path))
    authoring.save_concept("temp", ["a"], ["b"], root=str(tmp_path))
    assert os.path.isdir(os.path.join(tmp_path, "prompts", "concepts", "temp"))
    result = authoring.delete_concept("temp", root=str(tmp_path))
    assert "concepts" in result["removed"]
    assert not os.path.exists(os.path.join(tmp_path, "prompts", "concepts", "temp"))
