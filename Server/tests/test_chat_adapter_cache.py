"""Chat-path LoRA adapter cache: load once per (dir, content-hash), reuse via
set_adapter across turns, park disabled between requests so plain generation on
the same model slot stays bit-identical to a never-adapted model. Fakes only —
no peft weights, no GPU. The measured experiment paths (tasks.py/multi_agent.py)
keep the strict apply_adapter → remove_adapter lifecycle, untested here except
for the enable_adapters re-arm guard."""

import pytest

from steerlab_server.api import variant_chat
from steerlab_server.experiment import model_variant


class _FakeLM:
    """Records the transformers PeftAdapterMixin adapter API calls."""

    def __init__(self):
        self.calls = []
        self.loaded = {}
        self.active = None
        self.enabled = True  # PEFT layers start enabled after a load

    def load_adapter(self, path, adapter_name):
        assert adapter_name not in self.loaded, "load over an existing name"
        self.calls.append(("load", str(path), adapter_name))
        self.loaded[adapter_name] = str(path)
        self.active = adapter_name
        self.enabled = True

    def set_adapter(self, name):
        assert name in self.loaded, f"set_adapter on unloaded {name!r}"
        self.calls.append(("set", name))
        self.active = name

    def enable_adapters(self):
        self.calls.append(("enable",))
        self.enabled = True

    def disable_adapters(self):
        assert self.loaded, "disable_adapters with nothing loaded"
        self.calls.append(("disable",))
        self.enabled = False

    def delete_adapter(self, name):
        assert name in self.loaded, f"delete_adapter on unloaded {name!r}"
        self.calls.append(("delete", name))
        del self.loaded[name]


class _FakeModel:
    model_id = "org/model"
    revision = "abc"

    def __init__(self):
        self.model = _FakeLM()

    def adapter_active(self) -> bool:
        return self.model.enabled and bool(self.model.loaded) \
            and self.model.active in self.model.loaded


def _adapter_dir(tmp_path, name="adapter", weights=b"weights-v1"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(weights)
    return d


def _variant(adapter_dir=None):
    adapters = [{"adapterDirectory": str(adapter_dir)}] if adapter_dir else []
    return model_variant.ModelVariant(name="v", base_model_id="org/model",
                                      adapters=adapters)


def _calls(model, kind):
    return [c for c in model.model.calls if c[0] == kind]


def test_second_turn_same_variant_reuses_loaded_adapter(tmp_path):
    model = _FakeModel()
    variant = _variant(_adapter_dir(tmp_path))
    for _ in range(2):
        with variant_chat.prepared_variant(model, variant) as injections:
            assert injections == []
            assert model.adapter_active()
        assert not model.model.enabled  # parked dormant between requests
    assert len(_calls(model, "load")) == 1  # the whole point: no per-turn reload
    assert _calls(model, "delete") == []
    name = model_variant.ChatAdapterCache.ADAPTER_NAME
    assert list(model.model.loaded) == [name]


def test_changed_adapter_bytes_delete_and_reload(tmp_path):
    model = _FakeModel()
    d = _adapter_dir(tmp_path)
    variant = _variant(d)
    with variant_chat.prepared_variant(model, variant):
        pass
    # Same path, new content — mtime is irrelevant, the content hash must miss.
    (d / "adapter_model.safetensors").write_bytes(b"weights-v2-retrained")
    with variant_chat.prepared_variant(model, variant):
        assert model.adapter_active()
    assert len(_calls(model, "load")) == 2
    assert len(_calls(model, "delete")) == 1


def test_plain_generate_after_variant_leaves_adapters_inactive(tmp_path):
    model = _FakeModel()
    with variant_chat.prepared_variant(model, _variant(_adapter_dir(tmp_path))):
        assert model.adapter_active()
    # stripInterventions chat turn = the plain-generate control: the dormant
    # adapter must be disabled for the whole generation.
    with variant_chat.prepared_variant(model, _variant(), strip_interventions=True):
        assert not model.model.enabled
    assert not model.model.enabled


def test_no_adapter_variant_keeps_adapters_inactive(tmp_path):
    model = _FakeModel()
    # Nothing cached yet: an adapter-less variant must not touch the adapter API.
    with variant_chat.prepared_variant(model, _variant()):
        pass
    assert model.model.calls == []
    # After a cached adapter, an adapter-less variant generation runs disabled.
    with variant_chat.prepared_variant(model, _variant(_adapter_dir(tmp_path))):
        pass
    with variant_chat.prepared_variant(model, _variant()):
        assert not model.model.enabled
    assert len(_calls(model, "load")) == 1


def test_cache_lives_on_the_model_slot(tmp_path):
    """Cache is stored on the model wrapper, so a registry eviction (new model
    object) naturally starts cold — and two slots never share state."""
    variant = _variant(_adapter_dir(tmp_path))
    model_a, model_b = _FakeModel(), _FakeModel()
    for model in (model_a, model_b, model_a):
        with variant_chat.prepared_variant(model, variant):
            pass
    assert len(_calls(model_a, "load")) == 1
    assert len(_calls(model_b, "load")) == 1
    assert variant_chat._adapter_cache(model_a) is not variant_chat._adapter_cache(model_b)


def test_missing_adapter_directory_raises(tmp_path):
    model = _FakeModel()
    variant = _variant(tmp_path / "nope")
    with pytest.raises(ValueError):
        with variant_chat.prepared_variant(model, variant):
            pass


def test_adapter_content_hash_tracks_config_and_weights(tmp_path):
    d = _adapter_dir(tmp_path)
    h1 = model_variant.adapter_content_hash(str(d))
    assert h1 == model_variant.adapter_content_hash(str(d))  # deterministic
    (d / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 16}',
                                           encoding="utf-8")
    h2 = model_variant.adapter_content_hash(str(d))
    assert h2 != h1
    (d / "adapter_model.safetensors").write_bytes(b"other")
    assert model_variant.adapter_content_hash(str(d)) not in (h1, h2)
    # Irrelevant files (logs, READMEs) do not invalidate the cache.
    (d / "README.md").write_text("notes", encoding="utf-8")
    h3 = model_variant.adapter_content_hash(str(d))
    (d / "README.md").unlink()
    assert h3 == model_variant.adapter_content_hash(str(d))


def test_experiment_apply_adapter_rearms_disabled_layers(tmp_path):
    """The strict experiment path re-enables PEFT layers a chat turn may have
    parked disabled on the same registry slot (the flag is per-layer, not
    per-adapter-name)."""
    model = _FakeModel()
    with variant_chat.prepared_variant(model, _variant(_adapter_dir(tmp_path))):
        pass
    assert not model.model.enabled
    exp_dir = _adapter_dir(tmp_path, name="exp-adapter", weights=b"exp")
    handle = model_variant.apply_adapter(model, _variant(exp_dir))
    assert handle == "v" and model.model.enabled and model.model.active == "v"
    model_variant.remove_adapter(model, handle)
    assert "v" not in model.model.loaded
