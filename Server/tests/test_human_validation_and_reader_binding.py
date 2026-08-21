"""Review 2026-08-01, P1 pair: the human-validation row semantics and the
reader-pin binding.

Human validation: the engines used to read the SAME pinned file differently
(absent sampleIndex: wildcard in Swift, cell 0 here; duplicates: first-wins
there, last-wins here). The unified rule — exact-match-first, absent = an
explicit wildcard, duplicates refused — is pinned here and mirrored in the
Swift ``parseHumanValidation`` / ``humanAgreement`` tests.

Reader binding: a reader is activation-, model-, revision-, and
substrate-specific. Verify refuses an artifact that cannot say what it was
fitted on, and a ref whose concept disagrees with its artifact.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es, tasks
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --- human-validation rows ---------------------------------------------------

def _human_manifest(root, lines):
    digest = _write(os.path.join(root, "prompts", "human.jsonl"),
                    "\n".join(lines) + "\n")
    return SimpleNamespace(human_validation=SimpleNamespace(
        path="prompts/human.jsonl", hash=digest))


def test_wildcard_rows_cover_only_unclaimed_cells(tmp_path):
    root = str(tmp_path)
    manifest = _human_manifest(root, [
        json.dumps({"condition": "c", "promptID": "p", "outcome": "variant"}),
        json.dumps({"condition": "c", "promptID": "p", "outcome": "tie",
                    "sampleIndex": 1}),
    ])
    human = tasks._load_human_validation(manifest, root)
    assert set(human) == {("p", None, "c"), ("p", "1", "c")}
    judged = [("judge-1", {("p", "0", "c"): "baseline",
                           ("p", "1", "c"): "variant",
                           ("p", "2", "c"): "variant"})]
    resolved = tasks._materialize_human_validation(human, judged)
    # The exact row claims cell 1; the wildcard fills 0 and 2.
    assert resolved == {("p", "0", "c"): "variant",
                        ("p", "1", "c"): "tie",
                        ("p", "2", "c"): "variant"}


def test_duplicate_rows_refuse_exact_and_wildcard(tmp_path):
    root = str(tmp_path)
    exact = json.dumps({"condition": "c", "promptID": "p",
                        "outcome": "variant", "sampleIndex": 0})
    with pytest.raises(RuntimeError, match="duplicate"):
        tasks._load_human_validation(
            _human_manifest(root, [exact, exact]), root)
    wildcard = json.dumps({"condition": "c", "promptID": "p",
                           "outcome": "tie"})
    with pytest.raises(RuntimeError, match="duplicate"):
        tasks._load_human_validation(
            _human_manifest(root, [wildcard, wildcard]), root)


def test_sample_index_must_be_a_nonnegative_integer(tmp_path):
    root = str(tmp_path)
    for bad in ['"0"', "true", "-1", "1.5"]:
        line = ('{"condition": "c", "promptID": "p", "outcome": "tie", '
                f'"sampleIndex": {bad}}}')
        with pytest.raises(RuntimeError, match="sampleIndex"):
            tasks._load_human_validation(_human_manifest(root, [line]), root)


def test_missing_identity_fields_refuse(tmp_path):
    root = str(tmp_path)
    with pytest.raises(RuntimeError, match="condition and promptID"):
        tasks._load_human_validation(
            _human_manifest(root, ['{"outcome": "tie"}']), root)
    with pytest.raises(RuntimeError, match="no labeled rows"):
        tasks._load_human_validation(_human_manifest(root, [""]), root)


def test_numeric_identity_fields_refuse(tmp_path):
    """Review 2026-08-02 (P2): a numeric promptID was str()-coerced here
    while the Swift decoder refused it — the same file parsed on one engine
    and not the other. Non-empty JSON strings, both engines."""
    root = str(tmp_path)
    with pytest.raises(RuntimeError, match="non-empty strings"):
        tasks._load_human_validation(_human_manifest(root, [
            '{"condition": "c", "promptID": 5, "outcome": "tie"}']), root)


def test_manifest_verify_parses_the_pinned_rows(tmp_path):
    """A hash-clean pin over parser-refused rows (pasted JSON, a bundle) is
    a verify violation NOW, not an evaluate failure after the judging spend
    — verify reads the same parser evaluation uses."""
    root = str(tmp_path)
    digest = _write(os.path.join(root, "prompts", "hv.jsonl"),
                    '{"condition": "c", "promptID": 5, "outcome": "tie"}\n')
    es.create("hvv", model_id="org/m", revision="abc", root=root)
    d = es.load_raw("hvv", root)
    d["humanValidation"] = {"path": "prompts/hv.jsonl", "hash": digest}
    es.save_raw(d, root)
    violations = Manifest.load("hvv", root=root).verify(root)
    assert any("non-empty strings" in v for v in violations)


# --- reader binding ----------------------------------------------------------

def _reader_study(root, artifact, *, ref_concept="fair", revision="abc"):
    digest = _write(os.path.join(root, "prompts", "readers", "r.json"),
                    artifact)
    es.create("rb", model_id="org/m", revision=revision, root=root)
    d = es.load_raw("rb", root)
    d["readerRefs"] = [{"path": "prompts/readers/r.json", "hash": digest,
                        "concept": ref_concept}]
    es.save_raw(d, root)
    return Manifest.load("rb", root=root).verify(root)


def _artifact(**overrides):
    """A FULLY LOADABLE artifact (review 2026-08-02: verify now decodes
    through the real `repe_reader.load_reader`, so a stub missing
    template/probe is itself a violation — which is the point)."""
    base = {"artifactType": "repe-reader-lat", "schemaVersion": 1,
            "substrate": "python-hf-transformers",
            "modelID": "org/m", "revision": "abc", "concept": "fair",
            "layer": 2,
            "template": {"id": "t1", "conceptSlot": False,
                         "text": "Consider: {{stimulus}}",
                         "latToken": "final", "hash": "th"},
            "templateHash": "th", "datasetHash": "dh",
            "probe": {"direction": [1.0, 0.0], "projectionCenter": 0.0,
                      "projectionScale": 1.0, "orientation": 1.0,
                      "positiveMean": 1.0, "negativeMean": -1.0},
            "pc1ExplainedVariance": 0.5, "trainAccuracy": 0.9,
            "trainPairCount": 10, "heldOutPairCount": 2}
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def test_a_fully_bound_reader_verifies_clean(tmp_path):
    violations = _reader_study(str(tmp_path), _artifact())
    assert not [v for v in violations if "reader" in v]


def test_an_unloadable_artifact_is_a_violation_not_a_late_failure(tmp_path):
    """The scorer's loader is the verifier's loader: an artifact missing its
    probe used to pass the loose field inspection and die at evaluate."""
    violations = _reader_study(str(tmp_path), _artifact(probe=None))
    assert any("reader 'fair'" in v and "unreadable" in v.lower()
               or "reader 'fair'" in v and "probe" in v
               for v in violations)


def test_a_reader_without_a_revision_refuses(tmp_path):
    violations = _reader_study(str(tmp_path), _artifact(revision=None))
    assert any("no model revision" in v for v in violations)


def test_a_revision_mismatch_refuses(tmp_path):
    violations = _reader_study(str(tmp_path), _artifact(revision="beefcafe0000"))
    assert any("not the study's pinned" in v for v in violations)


def test_a_concept_mismatch_names_the_wrong_instrument(tmp_path):
    violations = _reader_study(str(tmp_path), _artifact(concept="anger"))
    assert any("wrong instrument" in v for v in violations)


def test_the_runtime_scorer_refuses_what_verify_flags(tmp_path):
    """Review 2026-08-02 (P1): the runtime checked only substrate, so on
    the permissive draft path (or under a forced freeze) it accepted
    readers verify would flag — including a reader with NO revision, and a
    ref scoring one reader while calling it another concept. Both now
    refuse through the SAME binding helper verify uses."""
    root = str(tmp_path)
    for overrides, match in [
        (dict(revision=None), "no model revision"),
        (dict(concept="anger"), "wrong instrument"),
        (dict(modelID="other/m"), "not the study model"),
    ]:
        rel = f"prompts/readers/{match.split()[0]}.json"
        _write(os.path.join(root, rel), _artifact(**overrides))
        manifest = SimpleNamespace(
            model_id="org/m", model_revision="abc",
            reader_refs=[SimpleNamespace(path=rel, concept="fair")])
        with pytest.raises(RuntimeError, match=match):
            tasks._reader_scorers(manifest, root)
