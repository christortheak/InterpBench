"""The closed dtype vocabulary (external review round 4, finding 2).

Before this, `model_loader.load` mapped ANY unrecognized dtype to float32
silently. Combined with the freeze gate that REQUIRES a judge dtype pin,
that made the pin a false claim: a manifest could say "banana" (or a real
dtype the device could not honor), the load would run fp32, and the
judgment artifact recorded the REQUEST rather than the reality.

Three things close it: a closed vocabulary refused at load, a refusal when
a genuine pin is not honored, and an artifact that stamps the dtype the
model ACTUALLY ran in alongside the one it was asked for.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.steering import model_loader


# --- the vocabulary itself ------------------------------------------------------


def test_the_two_vocabularies_agree():
    """The manifest firewall duplicates the loader's vocabulary rather than
    importing torch to validate a manifest. They must not drift."""
    assert es.JUDGE_DTYPE_VOCABULARY == model_loader.DTYPE_VOCABULARY


@pytest.mark.parametrize("alias,canonical", [
    ("bfloat16", "bfloat16"), ("bf16", "bfloat16"), ("BF16", "bfloat16"),
    ("float16", "float16"), ("fp16", "float16"),
    ("float32", "float32"), ("fp32", "float32"),
    ("  bf16  ", "bfloat16"),
])
def test_aliases_normalize_identically_on_both_paths(alias, canonical):
    assert model_loader.normalize_dtype(alias) == canonical
    assert es.normalize_judge_dtype(alias) == canonical


@pytest.mark.parametrize("value", ["banana", "int8", "float64", "bfloat", ""])
def test_unknown_spellings_normalize_to_none(value):
    assert model_loader.normalize_dtype(value) is None
    assert es.normalize_judge_dtype(value) is None


def test_auto_and_none_are_resolution_not_a_pin():
    """'auto' means let the device decide — absence of a pin, not a bad one.
    The manifest side has no notion of auto: a judge either pins or omits."""
    assert model_loader.normalize_dtype("auto") is None
    assert model_loader.normalize_dtype(None) is None


# --- the loader refuses ---------------------------------------------------------


def test_load_refuses_an_unknown_dtype_before_touching_weights():
    with pytest.raises(model_loader.ModelLoadError, match="unknown dtype"):
        model_loader.load("org/nonexistent-model", dtype="banana")


def test_the_refusal_names_the_vocabulary_and_the_reason():
    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        model_loader.load("org/nonexistent-model", dtype="float64")
    message = str(excinfo.value)
    for name in model_loader.DTYPE_VOCABULARY:
        assert name in message
    assert "silently" in message


# --- the manifest firewall refuses, earlier -------------------------------------


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _study(root, judges, name="s"):
    concept = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    d = es.load_raw(name, root)
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = _write(
        os.path.join(root, "prompts", "rubrics", "r.md"), "Judge.")
    d["judges"] = judges
    es.save_raw(d, root)
    from steerlab_server.experiment.manifest import Manifest
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


def test_freeze_refuses_a_judge_dtype_outside_the_vocabulary(tmp_path):
    root = str(tmp_path)
    name = _study(root, [
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "local", "model": "other/judge-12b",
         "revision": "cafe01", "dtype": "banana"},
    ])
    with pytest.raises(es.ExperimentStoreError,
                       match="cannot load") as excinfo:
        es.freeze(name, root=root)
    assert "'judge-2' declares dtype 'banana'" in str(excinfo.value)


def test_a_bad_dtype_is_caught_even_on_a_study_model_judge(tmp_path):
    """A study-model judge needs no PIN, but a dtype it does declare still
    has to be loadable — the loader would refuse it on the compute node."""
    root = str(tmp_path)
    name = _study(root, [
        {"name": "judge-1", "kind": "local", "dtype": "int4"},
        {"name": "judge-2", "kind": "openrouter",
         "model": "anthropic/claude-opus-4", "provider": "Anthropic"},
    ])
    with pytest.raises(es.ExperimentStoreError, match="cannot load"):
        es.freeze(name, root=root)


def test_alias_spellings_are_accepted_by_the_gate(tmp_path):
    root = str(tmp_path)
    name = _study(root, [
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "local", "model": "other/judge-12b",
         "revision": "cafe01", "dtype": "bf16"},
    ])
    assert es.freeze(name, root=root)["status"] == "frozen"
