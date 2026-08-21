"""The study-level dtype pin (2026-07-24).

Round 4 finding 2 closed the JUDGE side: a judge's dtype is a pin, refused
if unloadable and stamped as actually-loaded. That left an asymmetry — the
model which PRODUCED the text being judged had no pin at all, and greedy
decoding is not precision-proof (at a near-tie between two tokens, bf16 and
fp16 round differently, the argmax flips, and the continuation diverges).

`manifest.dtype` closes it. Server-honored, Swift-validated-but-unconsumed,
the same shape as `judges[].dtype`. Absent still means "let the device
decide", which is what every study did before the key existed.
"""

import contextlib
import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _study(root, name="s", *, dtype=None):
    concept = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    d = es.load_raw(name, root)
    if dtype is not None:
        d["dtype"] = dtype
    es.save_raw(d, root)
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"v-exp-{name}-validate")
    os.makedirs(rundir, exist_ok=True)
    json.dump({"schemaVersion": 1, "task": "validate",
               "substrate": "python-hf-transformers",
               "validationScopeHash": scope},
              open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"fear": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, "validation-report.json"), "w"))
    return name


class _FakeModel:
    def __init__(self, dtype):
        self.dtype = dtype
        self.model_id = "org/m"
        self.revision = "abc"


# --- the manifest key ----------------------------------------------------------


def test_the_pin_round_trips_and_is_omitted_when_absent(tmp_path):
    root = str(tmp_path)
    name = _study(root, dtype="bfloat16")
    assert Manifest.load(name, root).dtype == "bfloat16"
    plain = _study(root, "plain")
    assert Manifest.load(plain, root).dtype is None
    # Legacy manifests carry no key at all — absence must not become a value.
    assert "dtype" not in es.load_raw(plain, root)


# --- freeze validates it (the Mac is the authoring surface) --------------------


def test_freeze_refuses_an_unloadable_study_dtype(tmp_path):
    root = str(tmp_path)
    name = _study(root, dtype="banana")
    with pytest.raises(es.ExperimentStoreError, match="can load") as excinfo:
        es.freeze(name, root=root)
    assert "study dtype 'banana'" in str(excinfo.value)


def test_force_freeze_stamps_it_under_measurement_pins(tmp_path):
    """`measurementPins` was RESERVED in the cross-engine gate vocabulary.
    The study dtype is the first gate to use it."""
    root = str(tmp_path)
    name = _study(root, dtype="banana")
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["freezeForced"] is True
    assert "measurementPins" in frozen["forcedGatesSkipped"]


@pytest.mark.parametrize("spelling", ["bfloat16", "bf16", "fp16", "float32"])
def test_vocabulary_spellings_freeze(tmp_path, spelling):
    root = str(tmp_path)
    name = _study(root, f"ok{spelling}", dtype=spelling)
    assert es.freeze(name, root=root)["status"] == "frozen"


def test_an_unpinned_study_is_untouched(tmp_path):
    root = str(tmp_path)
    name = _study(root)
    assert es.unloadable_study_dtype_problem(es.load_raw(name, root)) is None
    assert es.freeze(name, root=root)["status"] == "frozen"


# --- the pin is honored at load ------------------------------------------------


def test_the_pin_wins_over_the_callers_auto_default(tmp_path):
    manifest = Manifest(name="s", model_id="org/m", dtype="float16")
    assert tasks._effective_dtype(manifest, "auto") == "float16"
    assert tasks._effective_dtype(manifest, "") == "float16"


def test_an_explicitly_conflicting_flag_refuses_rather_than_being_overridden(
        tmp_path):
    """Silently overriding the caller would hide the disagreement; silently
    obeying the caller would produce a run whose precision contradicts its
    own manifest. Refuse and name both."""
    manifest = Manifest(name="s", model_id="org/m", dtype="float16")
    with pytest.raises(RuntimeError, match="pins dtype 'float16'") as excinfo:
        tasks._effective_dtype(manifest, "bfloat16")
    assert "frozen recipe" in str(excinfo.value)


def test_an_alias_of_the_same_dtype_is_not_a_conflict():
    manifest = Manifest(name="s", model_id="org/m", dtype="bfloat16")
    assert tasks._effective_dtype(manifest, "bf16") == "bfloat16"


def test_an_unpinned_study_passes_the_callers_choice_through():
    manifest = Manifest(name="s", model_id="org/m")
    assert tasks._effective_dtype(manifest, "auto") == "auto"
    assert tasks._effective_dtype(manifest, "float32") == "float32"


# --- the resident-model conflict -----------------------------------------------


def test_a_resident_model_at_another_precision_refuses():
    """A loaded model's precision cannot be changed, so running the study at
    the resident precision while the manifest claims another would be
    exactly the false pin this key exists to prevent."""
    manifest = Manifest(name="s", model_id="org/m", dtype="float16")
    with pytest.raises(RuntimeError, match="resident as") as excinfo:
        tasks._assert_resident_dtype_matches(manifest, _FakeModel("bfloat16"))
    message = str(excinfo.value)
    assert "cannot be changed" in message
    assert "Unload it" in message


def test_a_matching_resident_model_is_fine():
    manifest = Manifest(name="s", model_id="org/m", dtype="bfloat16")
    tasks._assert_resident_dtype_matches(manifest, _FakeModel("bfloat16"))
    # Alias spellings compare canonically.
    manifest.dtype = "bf16"
    tasks._assert_resident_dtype_matches(manifest, _FakeModel("bfloat16"))


def test_an_unpinned_study_takes_whatever_is_resident():
    """Historical behaviour, deliberately preserved: no pin, no opinion."""
    manifest = Manifest(name="s", model_id="org/m")
    tasks._assert_resident_dtype_matches(manifest, _FakeModel("float32"))


# --- the run stamp (schema 3) --------------------------------------------------


def test_the_run_stamp_records_the_actual_dtype(tmp_path):
    from steerlab_server.experiment.run_config import write_run_config

    path = write_run_config(str(tmp_path), "run", dtype="bfloat16")
    assert json.load(open(path))["dtype"] == "bfloat16"


def test_actual_dtype_strips_the_torch_prefix():
    assert tasks._actual_dtype(_FakeModel("torch.bfloat16")) == "bfloat16"
    assert tasks._actual_dtype(_FakeModel("bfloat16")) == "bfloat16"
    assert tasks._actual_dtype(None) is None


# --- the study-model judge check against the HELD model (round 5, F1) ---------


class _Ref:
    def __init__(self, name="judge-1", revision=None, dtype=None):
        self.name = name
        self.revision = revision
        self.dtype = dtype


def test_a_study_model_judge_is_checked_against_the_weights_in_hand():
    """Freeze refuses divergence and the sweep preflight refuses it against
    the MANIFEST; this is the check against reality — the only one an
    unpinned 'let the device decide' dtype can be measured against."""
    manifest = Manifest(name="s", model_id="org/m", model_revision="abc")
    held = _FakeModel("bfloat16")
    held.revision = "abc"

    # Agreement (and absence) pass.
    tasks._assert_study_model_judge_matches_held(_Ref(), manifest, held)
    tasks._assert_study_model_judge_matches_held(
        _Ref(revision="abc", dtype="bf16"), manifest, held)

    with pytest.raises(RuntimeError, match="cannot load a second revision"):
        tasks._assert_study_model_judge_matches_held(
            _Ref(revision="other"), manifest, held)
    with pytest.raises(RuntimeError, match="cannot load a second"):
        tasks._assert_study_model_judge_matches_held(
            _Ref(dtype="float32"), manifest, held)


# --- co-residency capacity (round 5, finding 3) --------------------------------


def test_gpu_capacity_guard_measures_free_memory_not_total(monkeypatch):
    """A second model loading BESIDE a resident one was measured against a
    device that looked empty — nothing anywhere checked whether two models
    fit TOGETHER, which is the configuration the judge-lifetime fix made
    routine."""
    import torch

    from steerlab_server.steering import model_loader as ml

    gib = 1 << 30
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda i: type("P", (), {"total_memory": 80 * gib})())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "A100")
    monkeypatch.setattr(torch.cuda, "device", lambda i: contextlib.nullcontext())
    monkeypatch.setattr(ml, "snapshot_size_bytes", lambda m, r: 24 * gib)

    # Empty device: 24 GiB of weights fits 80 GiB fine.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (78 * gib, 80 * gib))
    ml._assert_gpu_capacity("cuda:0", "org/judge", None)

    # Study model already resident: only 10 GiB left, so the judge does not
    # fit even though the DEVICE is large enough.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10 * gib, 80 * gib))
    with pytest.raises(ml.ModelLoadError, match="is free") as excinfo:
        ml._assert_gpu_capacity("cuda:0", "org/judge", None)
    message = str(excinfo.value)
    # The remedy differs from the absolute refusal: unload, not "bigger GPU".
    assert "another model is already resident" in message
    assert "use the study model as judge" in message
    assert "larger GPU" not in message
