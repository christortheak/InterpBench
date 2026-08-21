"""Dual-root lookup for a concept's held-out validation.jsonl (2026-08-19).

A validation set has TWO possible homes and the RECIPE decides which is
canonical: paired recipes read ``prompts/concepts/<name>/``, the grand-mean
recipe reads ``prompts/emotions/<name>/``. Before this change a set filed
under the wrong recipe's root was treated as ABSENT — no hash pinned, no
error — so a measurement-side pin went quietly missing from the firewall.

The fix is lookup-only and deliberately narrow: the canonical home is read
first and always wins; the other home is a fallback; using the fallback (or
finding a file in both homes) is LOUD but never fatal. The three-state pin
semantics (hash / explicitly-absent / legacy-unpinned) are untouched.

Swift twin: ``ValidationDualRootTests``.
"""

import hashlib
import os

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import (
    Manifest, concept_validation_hash, resolve_validation_file,
    validation_lookup_advisory, validation_relpath)

VALIDATION_ROW = '{"text": "the shadows closed in", "expresses": true}\n'
OTHER_ROW = '{"text": "a different hidden scene", "expresses": false}\n'


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _paired_concept(root, name="fear", validation=None):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    if validation is not None:
        return _write(os.path.join(d, "validation.jsonl"), validation)
    return None


def _grand_mean_concept(root, name="fear", validation=None):
    d = os.path.join(root, "prompts", "emotions", name)
    _write(os.path.join(d, "stories.jsonl"),
           '{"concept": "%s", "text": "a long dread story"}\n' % name)
    if validation is not None:
        return _write(os.path.join(d, "validation.jsonl"), validation)
    return None


# --- resolution order ---------------------------------------------------------

def test_canonical_home_resolves_with_no_advisory(tmp_path):
    root = str(tmp_path)
    digest = _paired_concept(root, validation=VALIDATION_ROW)
    location = resolve_validation_file("fear", paired=True, root=root)
    assert location is not None
    assert location.relpath == "prompts/concepts/fear/validation.jsonl"
    assert not location.used_fallback and not location.both_present
    assert validation_lookup_advisory("fear", location) is None
    assert concept_validation_hash("fear", paired=True, root=root) == digest


def test_fallback_home_resolves_loudly_and_still_pins(tmp_path):
    """A grand-mean concept whose held-out set was filed under the PAIRED
    root: found, hashed, and advised about — never silently absent."""
    root = str(tmp_path)
    _grand_mean_concept(root)  # stories only, no validation beside them
    digest = _write(os.path.join(root, "prompts", "concepts", "fear",
                                 "validation.jsonl"), VALIDATION_ROW)
    location = resolve_validation_file("fear", paired=False, root=root)
    assert location is not None
    assert location.used_fallback and not location.both_present
    assert location.relpath == "prompts/concepts/fear/validation.jsonl"
    assert location.canonical_relpath == "prompts/emotions/fear/validation.jsonl"
    advisory = validation_lookup_advisory("fear", location)
    assert advisory is not None
    assert "OTHER recipe's root" in advisory
    assert "prompts/emotions/fear/validation.jsonl" in advisory
    # The pin is computed from the file actually FOUND.
    assert concept_validation_hash("fear", paired=False, root=root) == digest


def test_both_homes_present_canonical_wins_with_ambiguity_advisory(tmp_path):
    root = str(tmp_path)
    canonical = _grand_mean_concept(root, validation=VALIDATION_ROW)
    fallback = _write(os.path.join(root, "prompts", "concepts", "fear",
                                   "validation.jsonl"), OTHER_ROW)
    assert canonical != fallback
    location = resolve_validation_file("fear", paired=False, root=root)
    assert location is not None
    assert location.both_present and not location.used_fallback
    assert location.relpath == location.canonical_relpath
    advisory = validation_lookup_advisory("fear", location)
    assert advisory is not None and "BOTH" in advisory
    assert location.fallback_relpath in advisory
    assert concept_validation_hash("fear", paired=False, root=root) == canonical


def test_neither_home_stays_absent(tmp_path):
    root = str(tmp_path)
    _paired_concept(root)
    assert resolve_validation_file("fear", paired=True, root=root) is None
    assert validation_lookup_advisory("fear", None) is None
    assert concept_validation_hash("fear", paired=True, root=root) is None


def test_validation_relpath_is_the_documented_rule():
    assert validation_relpath("fear", paired=True) == \
        "prompts/concepts/fear/validation.jsonl"
    assert validation_relpath("fear", paired=False) == \
        "prompts/emotions/fear/validation.jsonl"


# --- lifecycle: attach pins it, verify accepts it, freeze advises about it ------

def test_attach_pins_misfiled_set_and_verify_stays_clean(tmp_path):
    root = str(tmp_path)
    _grand_mean_concept(root)
    digest = _write(os.path.join(root, "prompts", "concepts", "fear",
                                 "validation.jsonl"), VALIDATION_ROW)
    es.create("gm", model_id="org/m", revision="abc", root=root)
    d = es.attach("gm", ["fear"], method="emotionGrandMean", root=root)
    assert d["concepts"][0]["validationHash"] == digest
    assert Manifest.load("gm", root=root).verify(root) == []

    # Drift in the FALLBACK-home file is still drift.
    _write(os.path.join(root, "prompts", "concepts", "fear",
                        "validation.jsonl"), OTHER_ROW)
    violations = Manifest.load("gm", root=root).verify(root)
    assert any("validation.jsonl changed since pinning" in v
               for v in violations)


def test_freeze_advisory_names_the_misfiled_set(tmp_path):
    root = str(tmp_path)
    _grand_mean_concept(root)
    _write(os.path.join(root, "prompts", "concepts", "fear",
                        "validation.jsonl"), VALIDATION_ROW)
    es.create("gm", model_id="org/m", revision="abc", root=root)
    es.attach("gm", ["fear"], method="emotionGrandMean", root=root)
    advisories = es.freeze_advisories(es.load_raw("gm", root), root)
    assert any("OTHER recipe's root" in a
               and "prompts/emotions/fear/validation.jsonl" in a
               for a in advisories)


def test_freeze_advisory_silent_when_filing_is_canonical(tmp_path):
    root = str(tmp_path)
    _grand_mean_concept(root, validation=VALIDATION_ROW)
    es.create("gm", model_id="org/m", revision="abc", root=root)
    es.attach("gm", ["fear"], method="emotionGrandMean", root=root)
    advisories = es.freeze_advisories(es.load_raw("gm", root), root)
    assert not any("validation.jsonl" in a and "recipe" in a
                   for a in advisories)


def test_pinned_input_entries_name_the_file_actually_found(tmp_path):
    """The freeze snapshot / git gate / bundle packer share ONE pin-surface
    enumeration; a misfiled set has to travel from where it really is."""
    root = str(tmp_path)
    _grand_mean_concept(root)
    _write(os.path.join(root, "prompts", "concepts", "fear",
                        "validation.jsonl"), VALIDATION_ROW)
    es.create("gm", model_id="org/m", revision="abc", root=root)
    d = es.attach("gm", ["fear"], method="emotionGrandMean", root=root)
    entries = es.pinned_input_entries(d, root)
    validation = [e for e in entries if e.label.endswith("validation.jsonl")]
    assert validation, [e.label for e in entries]
    assert any(os.path.normpath(e.path).endswith(
        os.path.join("prompts", "concepts", "fear", "validation.jsonl"))
        for e in validation)


def test_data_check_surface_is_mac_authority():
    """This engine's `data check` is directory-driven (optvec / lora); the
    manifest-driven readiness checklist that reports a misfiled validation
    set lives on the Swift engine (`steerlab-cli data check <experiment>`).
    Pinned here so the twin is documented rather than assumed."""
    from steerlab_server import cli
    assert cli._DATA_TEMPLATES == ("optvec", "lora")
