"""SAE candidate roster as pinned study data (proposal r2 §8 P1-6).

Two things are under test: the schema (a roster that cannot be re-read is not
evidence, so every violation is a LOAD error) and the manifest pin (the
roster joins markers / the reasoning-style taxonomy on the verify() drift
surface, additively — nothing already frozen changes status).

Offline by construction: no HF, no Neuronpedia, no model.
"""

import copy
import json
import os
import shutil

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import sae_candidates as sc
from steerlab_server.experiment.manifest import Manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "sae-candidates", "valid-candidates.json")
TEMPLATE = os.path.join(REPO_ROOT, "prompts", "templates", "sae-candidates",
                        "sae-candidates-template.json")

FOCAL = {
    "constructLabel": "Strict textual interpretation",
    "role": "focal",
    "model": "google/gemma-3-27b-it",
    "source": "gemmascope-2-res-65k",
    "layer": 40,
    "featureId": 62389,
    "neuronpediaUrl": "https://www.neuronpedia.org/gemma-3-27b-it/"
                      "40-gemmascope-2-res-65k/62389",
    "discovery": {"explanationText": "textualism", "accessDate": "2026-08-13"},
    "verification": {"status": "verifiedOnNeuronpedia", "date": "2026-08-13"},
    "status": "candidate",
}


def _roster(*entries, **extra):
    payload = {"schemaVersion": 1, "candidates": list(entries)}
    payload.update(extra)
    return payload


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return path


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


# --- schema: accept -------------------------------------------------------

def test_committed_fixture_and_template_both_validate():
    """The template is the shape researchers copy; if it rots, everyone's
    roster starts invalid."""
    for path in (FIXTURE, TEMPLATE):
        assert sc.CandidateManifest.from_bytes(_read(path)).candidates


def test_template_carries_the_proposal_roster():
    manifest = sc.CandidateManifest.from_bytes(_read(TEMPLATE))
    identities = {(c.layer, c.feature_id): c for c in manifest.candidates}
    assert set(identities) == {(40, 62389), (16, 11409), (40, 40802),
                               (40, 10346), (40, 56976), (40, 2214)}
    assert identities[(40, 62389)].role == "focal"
    assert identities[(40, 56976)].role == "domainControl"
    assert identities[(40, 10346)].verification_status == "unverified"
    assert identities[(40, 2214)].verification_status == "unverified"
    assert identities[(40, 62389)].verification_date == "2026-08-13"
    # The three unfilled control slots travel as pending constructs.
    assert {p.role for p in manifest.pending} == {
        "affectControl", "embodiedControl", "unrelatedTopicControl"}


def test_gemma_scope_block_records_the_exact_dictionary():
    """The optional block that makes a nomination dictionary-precise: it is
    the SAME pair an imported artifact records in gemmascopeSource, so no
    string mapping from the discovery surface's 'source' is needed."""
    entry = dict(FOCAL, gemmaScope={"release": "gemma-scope-2-27b-it-res",
                                    "saeID": "layer_40_width_65k_l0_medium"})
    candidate = sc.CandidateManifest.from_dict(_roster(entry)).candidates[0]
    assert candidate.gemma_scope.release == "gemma-scope-2-27b-it-res"
    assert candidate.gemma_scope.sae_id == "layer_40_width_65k_l0_medium"
    assert candidate.gemma_scope.describe() == \
        "gemma-scope-2-27b-it-res/layer_40_width_65k_l0_medium"
    summary = sc.CandidateManifest.from_dict(_roster(entry)).summary()
    assert summary["withGemmaScopeDictionary"] == 1


def test_a_roster_without_the_block_is_still_valid_and_dictionary_blind():
    """Additive: every roster written before this key keeps loading, and says
    so in the summary rather than being quietly upgraded."""
    manifest = sc.CandidateManifest.from_dict(_roster(FOCAL))
    assert manifest.candidates[0].gemma_scope is None
    assert manifest.summary()["withGemmaScopeDictionary"] == 0


@pytest.mark.parametrize("block, expected", [
    ({"release": "gemma-scope-2-27b-it-res"}, "'saeID' is required"),
    ({"saeID": "layer_40_width_65k_l0_medium"}, "'release' is required"),
    ({"release": "r", "saeID": "s", "width": "65k"}, "unknown key(s) width"),
    ("gemma-scope-2-27b-it-res", "'gemmaScope' must be an object"),
    ({"release": 2, "saeID": "s"}, "'release' must be a string"),
])
def test_a_half_declared_dictionary_refuses(block, expected):
    """Half a dictionary identifies no better than none, and would make the
    seating guard's match silently partial."""
    with pytest.raises(sc.CandidateManifestError) as err:
        sc.CandidateManifest.from_dict(_roster(dict(FOCAL, gemmaScope=block)))
    assert expected in str(err.value)


def test_one_feature_number_in_two_dictionaries_is_two_nominations():
    """65k F62389 and 262k F62389 at one layer are DIFFERENT features — with
    the dictionary declared, nominating both is legal, not a collision."""
    a = dict(FOCAL, gemmaScope={"release": "gemma-scope-2-27b-it-res",
                                "saeID": "layer_40_width_65k_l0_medium"})
    b = dict(FOCAL, constructLabel="Other construct",
             gemmaScope={"release": "gemma-scope-2-27b-it-res",
                         "saeID": "layer_40_width_262k_l0_medium"})
    assert len(sc.CandidateManifest.from_dict(_roster(a, b)).candidates) == 2
    # …and the SAME dictionary twice is still one feature, one entry.
    with pytest.raises(sc.CandidateManifestError) as err:
        sc.CandidateManifest.from_dict(_roster(a, dict(a, constructLabel="x")))
    assert "already nominated" in str(err.value)
    assert "layer_40_width_65k_l0_medium" in str(err.value)


def test_the_template_declares_a_dictionary_for_every_nomination():
    """New nominations should carry it; the shipped template shows how."""
    manifest = sc.CandidateManifest.from_bytes(_read(TEMPLATE))
    assert all(c.gemma_scope is not None for c in manifest.candidates)
    widths = {c.source: c.gemma_scope.sae_id for c in manifest.candidates}
    assert "width_65k" in widths["gemmascope-2-res-65k"]
    assert "width_262k" in widths["gemmascope-2-res-262k"]


def test_control_entry_needs_no_discovery_snapshot():
    control = dict(FOCAL, role="domainControl", featureId=56976)
    control.pop("discovery")
    manifest = sc.CandidateManifest.from_dict(_roster(control))
    assert manifest.candidates[0].discovery is None


def test_absent_verification_block_reads_as_unverified():
    entry = dict(FOCAL)
    entry.pop("verification")
    manifest = sc.CandidateManifest.from_dict(_roster(entry))
    assert manifest.candidates[0].verification_status == "unverified"
    assert manifest.candidates[0].verification_date is None


def test_empty_roster_is_legal():
    assert sc.CandidateManifest.from_dict(_roster()).candidates == ()


def test_summary_counts_roles_status_and_evidence():
    control = dict(FOCAL, role="domainControl", featureId=56976,
                   status="rejected")
    control.pop("discovery")
    summary = sc.CandidateManifest.from_dict(
        _roster(FOCAL, control)).summary()
    assert summary["count"] == 2
    assert summary["byRole"] == {"focal": 1, "domainControl": 1}
    assert summary["byStatus"] == {"candidate": 1, "rejected": 1}
    assert summary["withDiscoverySnapshot"] == 1
    assert summary["withQualification"] == 0


# --- schema: reject -------------------------------------------------------

@pytest.mark.parametrize("mutate, expected", [
    (lambda e: e.pop("discovery"), "must carry a 'discovery' snapshot"),
    (lambda e: e.update(role="vibes"), "'role' must be one of"),
    (lambda e: e.update(status="promoted"), "'status' must be one of"),
    (lambda e: e.update(featureId="62389"), "'featureId' must be an integer"),
    (lambda e: e.update(layer=-1), "'layer' must be >= 0"),
    (lambda e: e.update(neuronpediaUrl="neuronpedia.org/x"),
     "must be an http(s) URL"),
    (lambda e: e.pop("constructLabel"), "'constructLabel' is required"),
    (lambda e: e.update(featuerId=1), "unknown key(s) featuerId"),
    (lambda e: e.update(qualificationArtifact="/abs/qual.json"),
     "must be WORKSPACE-RELATIVE"),
    (lambda e: e.update(verification={"status": "verifiedOnNeuronpedia"}),
     "'date' is required when status is"),
    (lambda e: e.update(verification={"status": "verifiedOnNeuronpedia",
                                      "date": "13 Aug 2026"}),
     "must be an ISO date"),
    (lambda e: e["discovery"].update(accessDate="yesterday"),
     "must be an ISO date"),
    (lambda e: e["discovery"].update(topPositiveLogits="textual"),
     "must be an array of strings"),
])
def test_entry_violations_are_load_errors(mutate, expected):
    entry = copy.deepcopy(FOCAL)
    mutate(entry)
    with pytest.raises(sc.CandidateManifestError) as err:
        sc.CandidateManifest.from_dict(_roster(entry))
    assert expected in str(err.value)


def test_duplicate_feature_identity_refuses():
    twin = dict(FOCAL, constructLabel="Formalism (another label)")
    with pytest.raises(sc.CandidateManifestError) as err:
        sc.CandidateManifest.from_dict(_roster(FOCAL, twin))
    assert "already nominated" in str(err.value)


def test_same_feature_id_in_a_different_dictionary_is_not_a_duplicate():
    other = dict(FOCAL, source="gemmascope-2-res-262k")
    assert len(sc.CandidateManifest.from_dict(
        _roster(FOCAL, other)).candidates) == 2


@pytest.mark.parametrize("payload, expected", [
    (b"not json", "not valid JSON"),
    (b'{"schemaVersion": 1}', "'candidates' must be an array"),
    (b'{"schemaVersion": 2, "candidates": []}', "unsupported schemaVersion"),
    (b'{"schemaVersion": 1, "candidates": [], "candidate": []}',
     "unknown key(s) candidate"),
    (b'[]', "must be a JSON object"),
    (b'{"schemaVersion": 1, "candidates": [], "pendingConstructs": [{"constructLabel": "Fear", "role": "nope"}]}',
     "'role' must be one of"),
    (b"\xff\xfe", "not UTF-8 text"),
])
def test_document_level_violations(payload, expected):
    with pytest.raises(sc.CandidateManifestError) as err:
        sc.CandidateManifest.from_bytes(payload)
    assert expected in str(err.value)


# --- hashing --------------------------------------------------------------

def test_content_hash_is_the_byte_hash_and_deterministic():
    data = _read(FIXTURE)
    import hashlib
    assert sc.content_hash(data) == hashlib.sha256(data).hexdigest()
    assert sc.content_hash(data) == sc.content_hash(data)
    # A reformat with identical MEANING is still a different pin: the rule is
    # bytes, so no cross-engine canonicalization contract is needed.
    reformatted = json.dumps(json.loads(data), indent=4).encode("utf-8")
    assert sc.content_hash(reformatted) != sc.content_hash(data)


def test_live_hash_is_none_when_the_file_is_absent(tmp_path):
    assert sc.live_hash("prompts/sae/nope.json", str(tmp_path)) is None
    rel = "prompts/sae/candidates.json"
    shutil.copyfile(FIXTURE, _write(os.path.join(str(tmp_path), rel), "{}"))
    assert sc.live_hash(rel, str(tmp_path)) == sc.content_hash(_read(FIXTURE))


# --- the manifest pin -----------------------------------------------------

def _study(root, *, concept="fear"):
    d = os.path.join(root, "prompts", "concepts", concept)
    _write(os.path.join(d, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm"}\n')
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", [concept], root=root)


def _seed_roster(root, rel="prompts/sae/candidates.json"):
    dest = os.path.join(root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(FIXTURE, dest)
    return rel


def test_pin_stamps_path_and_hash_and_freeze_keeps_verifying(tmp_path):
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    d = es.pin_sae_candidates("s", rel, root=root)
    assert d["saeCandidates"]["path"] == rel
    assert d["saeCandidates"]["hash"] == sc.live_hash(rel, root)
    assert Manifest.load("s", root=root).verify(root) == []
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["saeCandidates"]["hash"] == sc.live_hash(rel, root)
    assert Manifest.load("s", root=root).verify(root) == []


def test_drift_after_freeze_is_a_verify_violation(tmp_path):
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    es.pin_sae_candidates("s", rel, root=root)
    es.freeze("s", force=True, root=root)

    payload = json.loads(_read(os.path.join(root, rel)))
    payload["candidates"][0]["status"] = "seated"
    _write(os.path.join(root, rel), payload)
    violations = Manifest.load("s", root=root).verify(root)
    assert any("SAE candidate manifest" in v and "changed since pinning" in v
               for v in violations)


def test_disappearance_after_pinning_is_a_verify_violation(tmp_path):
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    es.pin_sae_candidates("s", rel, root=root)
    os.remove(os.path.join(root, rel))
    violations = Manifest.load("s", root=root).verify(root)
    assert any("file missing" in v and "SAE candidate manifest" in v
               for v in violations)


def test_freeze_refuses_a_declared_block_whose_file_is_missing(tmp_path):
    """Force included: verify() of the pins is never skippable."""
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    es.pin_sae_candidates("s", rel, root=root)
    os.remove(os.path.join(root, rel))
    with pytest.raises(es.ExperimentStoreError) as err:
        es.freeze("s", force=True, root=root)
    assert "SAE candidate manifest" in str(err.value)
    assert "file missing" in str(err.value)


def test_half_pin_and_absolute_path_refuse(tmp_path):
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    digest = sc.live_hash(rel, root)

    d = es.load_raw("s", root)
    d["saeCandidates"] = {"path": rel}
    es.save_raw(d, root)
    assert any("pin is incomplete" in v
               for v in Manifest.load("s", root=root).verify(root))

    d["saeCandidates"] = {"path": os.path.join(root, rel), "hash": digest}
    es.save_raw(d, root)
    assert any("is absolute" in v
               for v in Manifest.load("s", root=root).verify(root))

    d["saeCandidates"] = {"path": rel, "hash": digest, "extra": 1}
    es.save_raw(d, root)
    assert any("unknown key(s) extra" in v
               for v in Manifest.load("s", root=root).verify(root))


def test_hash_clean_but_invalid_roster_is_a_violation(tmp_path):
    """The PinShapeValidation precedent: a pin over bytes the engine refuses
    to read certifies nothing."""
    root = str(tmp_path)
    _study(root)
    rel = "prompts/sae/candidates.json"
    _write(os.path.join(root, rel), _roster(dict(FOCAL, role="vibes")))
    d = es.load_raw("s", root)
    d["saeCandidates"] = {"path": rel, "hash": sc.live_hash(rel, root)}
    es.save_raw(d, root)
    violations = Manifest.load("s", root=root).verify(root)
    assert any("is not valid" in v and "'role' must be one of" in v
               for v in violations)


def test_pin_refuses_an_invalid_roster_and_a_frozen_manifest(tmp_path):
    root = str(tmp_path)
    _study(root)
    rel = "prompts/sae/candidates.json"
    _write(os.path.join(root, rel), _roster(dict(FOCAL, status="promoted")))
    with pytest.raises(es.ExperimentStoreError) as err:
        es.pin_sae_candidates("s", rel, root=root)
    assert "'status' must be one of" in str(err.value)
    assert "saeCandidates" not in es.load_raw("s", root)

    with pytest.raises(es.ExperimentStoreError) as err:
        es.pin_sae_candidates("s", "prompts/sae/absent.json", root=root)
    assert "no SAE candidate manifest at" in str(err.value)

    with pytest.raises(es.ExperimentStoreError) as err:
        es.pin_sae_candidates("s", os.path.join(root, rel), root=root)
    assert "is absolute" in str(err.value)

    good = _seed_roster(root, "prompts/sae/good.json")
    es.pin_sae_candidates("s", good, root=root)
    es.freeze("s", force=True, root=root)
    with pytest.raises(es.ExperimentStoreError) as err:
        es.pin_sae_candidates("s", good, root=root)
    assert "duplicate the experiment" in str(err.value)


# --- back-compat: a manifest without the key is untouched ------------------

def test_legacy_manifest_without_the_key_freezes_and_verifies_unchanged(tmp_path):
    root = str(tmp_path)
    _study(root)
    # A roster file sitting in the workspace UNPINNED must not be consulted:
    # nothing is pinned, so nothing can drift.
    _seed_roster(root)
    assert "saeCandidates" not in es.load_raw("s", root)
    assert Manifest.load("s", root=root).verify(root) == []
    frozen = es.freeze("s", force=True, root=root)
    # No key is added anywhere, so the canonical payload — and therefore the
    # freeze hash — is byte-identical to what a pre-feature engine produced.
    assert "saeCandidates" not in frozen
    assert frozen["freezeHash"] == Manifest.load("s", root=root).content_hash()
    assert Manifest.load("s", root=root).verify(root) == []
    payload = json.loads(_read(os.path.join(root, "prompts", "sae",
                                            "candidates.json")))
    payload["candidates"] = []
    _write(os.path.join(root, "prompts", "sae", "candidates.json"), payload)
    assert Manifest.load("s", root=root).verify(root) == []


def test_pinned_block_changes_the_content_hash(tmp_path):
    """Pinning is a manifest change like any other — it must not be invisible
    to the freeze hash."""
    root = str(tmp_path)
    _study(root)
    before = Manifest.load("s", root=root).content_hash()
    es.pin_sae_candidates("s", _seed_roster(root), root=root)
    assert Manifest.load("s", root=root).content_hash() != before


def test_pinned_roster_is_a_pinned_input_for_snapshot_and_bundles(tmp_path):
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    es.pin_sae_candidates("s", rel, root=root)
    d = es.load_raw("s", root)
    labels = {e.label for e in es.pinned_input_entries(d, root)}
    assert "SAE candidate manifest" in labels
    es.freeze("s", force=True, root=root)
    assert os.path.exists(os.path.join(root, "experiments", "s", "pinned", rel))


# --- CLI ------------------------------------------------------------------

def test_cli_check_prints_a_summary_and_gates_on_violations(tmp_path, capsys):
    from steerlab_server import cli
    root = str(tmp_path)
    rel = _seed_roster(root)
    assert cli.main(["--root", root, "sae", "candidates", "check", rel]) == 0
    out = capsys.readouterr().out
    assert "focal 1" in out and "domainControl 1" in out
    assert "pending slot  Hunger" in out

    assert cli.main(["--root", root, "sae", "candidates", "check", rel,
                     "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["byRole"] == {"focal": 1, "domainControl": 1}
    assert payload["sha256"] == sc.live_hash(rel, root)

    bad = "prompts/sae/bad.json"
    _write(os.path.join(root, bad), _roster(dict(FOCAL, role="vibes")))
    assert cli.main(["--root", root, "sae", "candidates", "check", bad]) == 2
    assert "'role' must be one of" in capsys.readouterr().err

    assert cli.main(["--root", root, "sae", "candidates"]) == 64


def test_cli_pin_stamps_the_manifest(tmp_path, capsys):
    from steerlab_server import cli
    root = str(tmp_path)
    _study(root)
    rel = _seed_roster(root)
    assert cli.main(["--root", root, "sae", "candidates", "pin", "s", rel]) == 0
    assert es.load_raw("s", root)["saeCandidates"]["path"] == rel
    assert "pinned" in capsys.readouterr().out
    assert cli.main(["--root", root, "sae", "candidates", "pin", "s",
                     "prompts/sae/absent.json"]) == 2
