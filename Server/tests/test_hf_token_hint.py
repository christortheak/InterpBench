"""Startup HF-token detection: env vars or a cached huggingface-cli login."""

from steerlab_server import cli


def test_hf_token_detection(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert not cli._hf_token_configured()
    (tmp_path / "token").write_text("hf_stored", encoding="utf-8")
    assert cli._hf_token_configured()
    (tmp_path / "token").unlink()
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_env2")
    assert cli._hf_token_configured()
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN")
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert cli._hf_token_configured()
