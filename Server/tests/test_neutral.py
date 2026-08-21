"""Neutral corpus parsing + storage (no model)."""

from steerlab_server.experiment import neutral


def test_parse_jsonl():
    out = neutral.parse_texts('{"text":"one"}\n{"text":"two"}')
    assert out == ["one", "two"]


def test_parse_prose_paragraphs():
    out = neutral.parse_texts("para one\nstill one\n\npara two")
    assert out == ["para one\nstill one", "para two"]


def test_parse_json_array():
    assert neutral.parse_texts('["a","b"]') == ["a", "b"]


def test_save_and_list_norm_corpus(tmp_path):
    info = neutral.save_corpus(["x", "y", "  "], name=None, root=str(tmp_path))
    assert info["name"] == "norm" and info["count"] == 2 and info["hash"]
    texts, h = neutral.read_corpus(None, root=str(tmp_path))
    assert texts == ["x", "y"] and h == info["hash"]
    corpora = neutral.list_corpora(root=str(tmp_path))
    assert corpora[0]["name"] == "norm" and corpora[0]["count"] == 2


def test_save_projection_corpus(tmp_path):
    neutral.save_corpus(["a", "b"], name="assistant-dialogue", root=str(tmp_path))
    names = {c["name"] for c in neutral.list_corpora(root=str(tmp_path))}
    assert {"norm", "assistant-dialogue"} <= names


# --- token-bank memory cap (deterministic downsample) ------------------------

def test_deterministic_row_selection_caps_and_reproduces():
    from steerlab_server.steering.extractor import deterministic_row_selection

    first = deterministic_row_selection(1000, 64, seed=42)
    again = deterministic_row_selection(1000, 64, seed=42)
    assert first == again
    assert len(first) == 64
    assert all(0 <= i < 1000 for i in first)
    # A different seed (i.e. a different corpus hash) picks a different bank.
    other = deterministic_row_selection(1000, 64, seed=43)
    assert other != first
    # Under the cap (or uncapped): keep everything.
    assert deterministic_row_selection(50, 64, seed=42) is None
    assert deterministic_row_selection(1000, None, seed=42) is None


def test_save_basis_stamps_downsample_provenance(tmp_path):
    path = neutral.save_basis(
        model_id="org/m", revision="abc", corpus_name="norm",
        corpus_hash="ab" * 32, components_by_layer={0: [[1.0, 0.0]]},
        residual_norm_per_layer=[1.0], token_rows=64,
        run_directory=str(tmp_path), token_positions_total=1000,
        token_positions_kept=64, downsample_seed=7)
    import json
    artifact = json.load(open(path))
    assert artifact["tokenRows"] == 64
    assert artifact["tokenPositionsTotal"] == 1000
    assert artifact["tokenPositionsKept"] == 64
    assert artifact["downsampleSeed"] == 7
