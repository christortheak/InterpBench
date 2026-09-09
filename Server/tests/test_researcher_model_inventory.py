import json
from steerlab_server.steering.model_loader import local_model_ids


def test_inventory_excludes_explicit_non_generation_architectures(tmp_path):
    for name, architecture in [("chat", "ExampleForCausalLM"), ("vision", "ExampleForConditionalGeneration"),
                               ("classifier", "ExampleForSequenceClassification"), ("features", "SparseAutoencoder")]:
        snapshot = tmp_path / "hub" / f"models--org--{name}" / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    assert local_model_ids(str(tmp_path)) == ["org/chat", "org/vision"]
