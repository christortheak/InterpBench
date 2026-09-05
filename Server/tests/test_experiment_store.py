"""Experiment authoring: create/attach/condition/duplicate/freeze + the
frozen-immutability guard and freeze gating."""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest


def _concept(root, name="french"):
    d = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "positive.jsonl"), "w").write('{"text": "bonjour"}\n')
    open(os.path.join(d, "negative.jsonl"), "w").write('{"text": "hello"}\n')


def test_create_attach_condition(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("My Study", model_id="org/m", root=root)
    d = es.attach("my-study", ["french"], method="lat", pool_from_token=50, root=root)
    assert d["concepts"][0]["name"] == "french"
    assert d["concepts"][0]["stimulusSetHash"]
    assert d["concepts"][0]["options"]["method"] == "lat"
    # readingPosition is the Swift Codable enum shape so the Mac app can read it
    assert d["concepts"][0]["options"]["readingPosition"] == {"meanFromToken": {"_0": 50}}
    d = es.add_condition("my-study", {"name": "fear-L5", "slots": [
        {"concept": "french", "layer": 5, "alpha": 2.0}], "bandWidth": 3,
        "alphaInNormUnits": False}, root=root)
    assert d["conditions"][0]["name"] == "fear-L5"


def test_frozen_is_read_only(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)  # verify needs a concept/variant
    # force-freeze (skips validate-evidence gate; verify still runs)
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["status"] == "frozen"
    assert frozen["frozenBy"] == "server" and frozen["freezeHash"]
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"temperature": 0.9}, root=root)


def test_freeze_writes_generated_preregistration_when_path_is_free(tmp_path):
    """The unchanged common case: nobody authored a preregistration, so the
    freeze-time settings summary lands at its historical path, carrying the
    marker line the destination rule keys on — and no displaced-summary file
    appears."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    es.freeze("s", force=True, root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    with open(os.path.join(exp_dir, "preregistration.md"), encoding="utf-8") as h:
        text = h.read()
    assert "# Preregistration: s" in text
    assert es.PREREG_GENERATED_MARKER in text
    assert not os.path.exists(
        os.path.join(exp_dir, es.PREREG_FROZEN_SETTINGS_FILENAME))


def test_freeze_preserves_hand_authored_preregistration(tmp_path):
    """Field incident 2026-08-29: a researcher-authored ANALYSIS
    preregistration at experiments/<name>/preregistration.md — commitments
    written before any data existed — was silently destroyed by the first
    freeze. It must survive byte-for-byte, with the generated settings
    summary landing beside it under its own name."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    authored = ("# Analysis preregistration\n\n"
                "We commit to the paired-difference estimator before any "
                "data exists.\n")
    with open(os.path.join(exp_dir, "preregistration.md"), "w",
              encoding="utf-8") as h:
        h.write(authored)
    es.freeze("s", force=True, root=root)
    with open(os.path.join(exp_dir, "preregistration.md"), encoding="utf-8") as h:
        assert h.read() == authored  # preserved untouched
    with open(os.path.join(exp_dir, es.PREREG_FROZEN_SETTINGS_FILENAME),
              encoding="utf-8") as h:
        displaced = h.read()
    assert "# Preregistration: s" in displaced
    assert es.PREREG_GENERATED_MARKER in displaced


def test_freeze_overwrites_stale_generated_preregistration(tmp_path):
    """LEGACY structural path: a file with no stamp to check it against is the
    freeze's own prior output only when it LOOKS exactly like one — generated
    header on line 1, marker line last. Such a file (e.g. copied in by hand
    from another study, or written by a freeze predating the stamps) is
    regenerated in sync with THIS freeze instead of displacing the summary."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    with open(os.path.join(exp_dir, "preregistration.md"), "w",
              encoding="utf-8") as h:
        h.write("# Preregistration: other-study\n\nstale facts\n\n"
                + es.PREREG_GENERATED_MARKER + " Duplicate the experiment to "
                "change anything.*\n")
    es.freeze("s", force=True, root=root)
    with open(os.path.join(exp_dir, "preregistration.md"), encoding="utf-8") as h:
        text = h.read()
    assert "stale facts" not in text
    assert "# Preregistration: s" in text
    assert not os.path.exists(
        os.path.join(exp_dir, es.PREREG_FROZEN_SETTINGS_FILENAME))


def test_freeze_preserves_authored_preregistration_that_quotes_the_footer(tmp_path):
    """Review 2026-08-29, P1: the marker test was a SUBSTRING check, so the
    most natural way to write a real preregistration — start from the
    generated summary, add the commitments above it, leave the footer alone —
    was classified as generated and destroyed. Position, not presence: the
    marker is only ours when it is the final non-empty line AND the generated
    header is line 1."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    authored = ("# Analysis preregistration\n\n"
                "We commit to the paired-difference estimator.\n\n"
                "The settings summary a previous freeze wrote follows:\n\n"
                + es.PREREG_GENERATED_MARKER + " Duplicate the experiment to "
                "change anything.*\n\n"
                "## Deviations\n\nNone so far.\n")
    with open(os.path.join(exp_dir, "preregistration.md"), "w",
              encoding="utf-8") as h:
        h.write(authored)
    frozen = es.freeze("s", force=True, root=root)
    with open(os.path.join(exp_dir, "preregistration.md"), encoding="utf-8") as h:
        assert h.read() == authored  # preserved, footer quote and all
    assert os.path.exists(os.path.join(exp_dir, es.PREREG_FROZEN_SETTINGS_FILENAME))
    assert frozen[es.PREREG_AUTHORED_HASH_KEY] == hashlib.sha256(
        authored.encode("utf-8")).hexdigest()


def test_freeze_stamps_and_snapshots_the_preserved_preregistration(tmp_path):
    """Review 2026-08-29, P1: preserving the file froze NOTHING about it. The
    freeze now stamps its sha256, snapshots it into pinned/ for the no-git
    floor, and puts it on the pin surface."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    authored = "# Analysis preregistration\n\nOne estimator, chosen now.\n"
    with open(os.path.join(exp_dir, "preregistration.md"), "w",
              encoding="utf-8") as h:
        h.write(authored)
    frozen = es.freeze("s", force=True, root=root)
    digest = hashlib.sha256(authored.encode("utf-8")).hexdigest()
    assert frozen[es.PREREG_AUTHORED_HASH_KEY] == digest
    # The stamp is a FREEZE stamp: outside the canonical payload, so it
    # cannot disturb the freeze hash it is written beside.
    with open(os.path.join(exp_dir, "freeze-canonical.json"),
              encoding="utf-8") as h:
        canonical = json.load(h)
    assert es.PREREG_AUTHORED_HASH_KEY not in canonical
    assert es.PREREG_GENERATED_HASH_KEY not in canonical
    assert Manifest.load("s", root).content_hash() == frozen["freezeHash"]
    with open(os.path.join(exp_dir, "pinned", "preregistration.md"),
              encoding="utf-8") as h:
        assert h.read() == authored  # no-git reproducibility floor
    surface = {e.label for e in es.pinned_input_entries(frozen, root)}
    assert "researcher-authored preregistration" in surface


def test_editing_a_preserved_preregistration_after_freeze_fails_verify(tmp_path):
    """The whole point of the stamp: an authored preregistration is a frozen
    artifact, so a post-freeze edit is drift like any other pinned input."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    prereg = os.path.join(root, "experiments", "s", "preregistration.md")
    with open(prereg, "w", encoding="utf-8") as h:
        h.write("# Analysis preregistration\n\nWe predict a positive shift.\n")
    es.freeze("s", force=True, root=root)
    assert Manifest.load("s", root).verify(root) == []
    with open(prereg, "w", encoding="utf-8") as h:
        h.write("# Analysis preregistration\n\nWe predicted the shift we got.\n")
    violations = Manifest.load("s", root).verify(root)
    assert any("researcher-authored preregistration" in v for v in violations)
    os.remove(prereg)
    assert any("researcher-authored preregistration" in v
               for v in Manifest.load("s", root).verify(root))


def test_legacy_frozen_experiment_verifies_without_the_preregistration_stamp(tmp_path):
    """Frozen directories are immutable, so experiments frozen before the
    stamp existed have none: absence is not a violation, and their authored
    preregistration is free to sit there unhashed."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    frozen = es.freeze("s", force=True, root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    # Roll the manifest back to what a pre-stamp freeze wrote, and put an
    # authored preregistration beside it.
    frozen.pop(es.PREREG_AUTHORED_HASH_KEY, None)
    frozen.pop(es.PREREG_GENERATED_HASH_KEY, None)
    with open(os.path.join(exp_dir, "experiment.json"), "w",
              encoding="utf-8") as h:
        json.dump(frozen, h)
    with open(os.path.join(exp_dir, "preregistration.md"), "w",
              encoding="utf-8") as h:
        h.write("# Analysis preregistration\n\nWritten in 2026.\n")
    assert Manifest.load("s", root).verify(root) == []


def test_freeze_refreshes_its_own_stamped_generated_preregistration(tmp_path):
    """Provenance-plus-hash: when the manifest carries the stamp of the
    summary this instrument generated and the bytes still match it, the file
    is provably ours and is refreshed rather than displaced — no structural
    guessing required."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    exp_dir = os.path.join(root, "experiments", "s")
    frozen = es.freeze("s", force=True, root=root)
    with open(os.path.join(exp_dir, "preregistration.md"), "rb") as h:
        generated = h.read()
    assert frozen[es.PREREG_GENERATED_HASH_KEY] == hashlib.sha256(
        generated).hexdigest()
    # Re-open the study as a draft carrying its stamps (what copying an
    # experiment directory by hand produces) and freeze again: the untouched
    # generated file is recognized by its hash.
    frozen["status"] = "draft"
    with open(os.path.join(exp_dir, "experiment.json"), "w",
              encoding="utf-8") as h:
        json.dump(frozen, h)
    refrozen = es.freeze("s", force=True, root=root)
    assert es.PREREG_AUTHORED_HASH_KEY not in refrozen
    assert not os.path.exists(
        os.path.join(exp_dir, es.PREREG_FROZEN_SETTINGS_FILENAME))


def test_preregistration_classification_rule(tmp_path):
    """The classifier itself, at the boundaries the freeze tests cannot
    isolate: a stored hash decides when there is one, structure decides when
    there is not, and doubt always resolves to 'preserve'."""
    path = str(tmp_path / "preregistration.md")

    def _write(text):
        with open(path, "w", encoding="utf-8") as h:
            h.write(text)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    footer = es.PREREG_GENERATED_MARKER + " Duplicate the experiment to change anything.*"
    # 1. Stamped generated hash wins over a file that looks nothing like ours.
    digest = _write("not remotely our shape\n")
    assert es._preregistration_is_generated(
        path, {es.PREREG_GENERATED_HASH_KEY: digest})
    # 2. A stamp that does NOT match the bytes preserves, even when the file
    #    is structurally perfect: something edited it.
    _write("# Preregistration: s\n\nbody\n\n" + footer + "\n")
    assert not es._preregistration_is_generated(
        path, {es.PREREG_GENERATED_HASH_KEY: "0" * 64})
    # 3. The preserved-authored stamp names it as the researcher's.
    digest = _write("# Preregistration: s\n\nbody\n\n" + footer + "\n")
    assert not es._preregistration_is_generated(
        path, {es.PREREG_AUTHORED_HASH_KEY: digest})
    # 4. No stamps: structure decides. Header first, marker last → ours.
    _write("# Preregistration: s\n\nbody\n\n" + footer + "\n")
    assert es._preregistration_is_generated(path, {})
    # 5. …the marker quoted mid-document is NOT ours.
    _write("# Analysis preregistration\n\n" + footer + "\n\nmore\n")
    assert not es._preregistration_is_generated(path, {})
    # 6. …nor is one whose header is right but whose footer moved.
    _write("# Preregistration: s\n\n" + footer + "\n\ntrailing commitments\n")
    assert not es._preregistration_is_generated(path, {})
    # 7. …nor a footer-terminated file under someone else's heading.
    _write("# Analysis preregistration\n\nbody\n\n" + footer + "\n")
    assert not es._preregistration_is_generated(path, {})
    # 8. Unreadable bytes are authored — when in doubt, preserve.
    with open(path, "wb") as h:
        h.write(b"\xff\xfe not utf-8")
    assert not es._preregistration_is_generated(path, {})


def test_write_preregistration_noops_for_legacy_flat_manifest(tmp_path):
    """A legacy flat-file manifest has no experiments/<name>/ directory: the
    writer must not create one (guard unchanged by the destination rule)."""
    root = str(tmp_path)
    d = {"name": "flat", "status": "frozen", "modelID": "org/m",
         "frozenAt": "2026-08-29T00:00:00Z"}
    es._write_preregistration(d, root)
    assert not os.path.exists(os.path.join(root, "experiments", "flat"))


def test_set_protocol_refuses_unknown_keys_at_the_store(tmp_path):
    """The store itself refuses, not just the client CLI: the HTTP authoring
    route hands request bodies straight to ``set_protocol``, so a drop here
    would keep that surface silent. Nothing is written on refusal — the
    valid keys in the same call must not land either."""
    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.set_protocol("s", {"temperature": 0.7, "notAField": 1}, root=root)
    assert "'notAField'" in str(exc.value)
    assert "temperature" in str(exc.value)  # the vocabulary is listed
    assert exc.value.repair_action
    d = es.load_raw("s", root)
    assert d["temperature"] == 0.0  # create's default, not 0.7
    assert "notAField" not in d
    # The contract keys the Mac spells as verbs land as protocol fields.
    d = es.set_protocol(
        "s", {"outcomeInstruments": ["sampledText"],
              "sweep": {"selection": {"objective": {"kind": "markerDensity"}}}},
        root=root)
    assert d["outcomeInstruments"] == ["sampledText"]
    assert d["sweep"]["selection"]["objective"]["kind"] == "markerDensity"
    # …with the instrument vocabulary enforced at declaration (Swift
    # `setOutcomeInstruments` twin) and sweep required to be an object.
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"outcomeInstruments": ["sampledTxt"]}, root=root)
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"sweep": "markerDensity"}, root=root)


def test_set_protocol_gates_the_sampling_protocol_fields(tmp_path):
    """The six generation-protocol fields carry declaration-time value gates
    (Swift twins: ``ExperimentStore.setSamplingProtocol`` /
    ``setExclusionRules``). Two loss classes: an out-of-vocabulary
    promptMode/seedPolicy is read by equality tests downstream and silently
    behaves as the default, and a non-numeric temperature/maxTokens/
    samplesPerItem BRICKS the manifest — ``Manifest.from_dict`` raises on the
    next load, so every later verb fails before it can repair. Nothing is
    written on refusal, including the valid keys of the same call."""
    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    # The stochastic replication arm that motivated the writers (25 samples ×
    # T=0.7 × 1024 tokens): authorable, and the run loop's own decoder reads
    # it back.
    d = es.set_protocol(
        "s", {"temperature": 0.7, "maxTokens": 1024, "samplesPerItem": 25,
              "seedPolicy": "derivedSHA256", "promptMode": "chatAssistant"},
        root=root)
    assert d["samplesPerItem"] == 25 and d["seedPolicy"] == "derivedSHA256"
    from steerlab_server.experiment.manifest import Manifest
    manifest = Manifest.from_dict(d)
    assert manifest.samples_per_item == 25
    assert manifest.seed_policy == "derivedSHA256"
    for bad in ({"temperature": "hot"}, {"temperature": -0.5},
                {"temperature": True},
                {"maxTokens": 0}, {"maxTokens": "lots"},
                {"promptMode": "freestyle"},
                {"samplesPerItem": 0}, {"samplesPerItem": "three"},
                {"seedPolicy": "diceRoll"},
                {"exclusionRules": [{"rule": "outOfRange"}]},
                {"exclusionRules": "outOfRange"}):
        with pytest.raises(es.ExperimentStoreError) as exc:
            es.set_protocol("s", dict(bad, taskDescription="x"), root=root)
        assert exc.value.repair_action, bad
    d = es.load_raw("s", root)
    assert "taskDescription" not in d  # the valid co-key never landed
    assert d["samplesPerItem"] == 25  # the earlier declaration survives
    # A JSON null clears like an absent key on decode, so None passes.
    es.set_protocol("s", {"seedPolicy": None}, root=root)
    assert "seedPolicy" not in es.load_raw("s", root)
    # Valid exclusion rules land verbatim; an unknown rule id refuses with
    # the engine's own wording and the Mac verb named in the repair.
    d = es.set_protocol(
        "s", {"exclusionRules": [
            {"rule": "unparseableEndpoint"},
            {"rule": "outOfRange", "min": 0, "max": 600}]}, root=root)
    assert [r["rule"] for r in d["exclusionRules"]] == [
        "unparseableEndpoint", "outOfRange"]
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.set_protocol("s", {"exclusionRules": [{"rule": "outOfRnge"}]},
                        root=root)
    assert "not recognized" in str(exc.value)
    assert "set-exclusions" in exc.value.repair_action


def test_an_explicit_json_null_clears_a_protocol_field(tmp_path):
    """Review round 10, finding 1. Every value gate reads
    ``fields.get(k) is not None``, so a null passes them all ungated — and the
    persistence loop used to WRITE it. ``Manifest.from_dict`` then raised
    ``TypeError`` on the next load and every later verb died before it could
    name the problem, verify included: a client verb that BRICKS the manifest.

    The site's own comment claimed "a JSON null clears like an absent key on
    decode"; the loop now makes that claim true by popping the key, which is
    also the symmetric affordance to the Swift writers' ``""`` clears."""
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    written = {"temperature": 0.7, "maxTokens": 1024, "samplesPerItem": 25,
               "seedPolicy": "derivedSHA256", "promptMode": "chatAssistant",
               "exclusionRules": [{"rule": "unparseableEndpoint"}],
               "outcomeInstruments": ["sampledText"]}
    es.set_protocol("s", dict(written), root=root)
    for key in written:
        # Write a value, null it, and the key is GONE — not present as null.
        assert key in es.load_raw("s", root), key
        d = es.set_protocol("s", {key: None}, root=root)
        assert key not in d, key
        assert key not in es.load_raw("s", root), key
        # …and the manifest still loads, which is the whole point: the
        # bricked state can no longer be produced.
        Manifest.from_dict(es.load_raw("s", root))

    # Nulling every field at once, on a fresh draft, is the same story.
    es.create("t", model_id="org/m", root=root)
    es.set_protocol("t", dict(written), root=root)
    d = es.set_protocol("t", {key: None for key in written}, root=root)
    assert not (set(written) & set(d))
    manifest = Manifest.from_dict(es.load_raw("t", root))
    assert manifest.temperature is None or manifest.temperature is not False
    # Nulling a key that was never there is a no-op, not a KeyError.
    es.set_protocol("t", {"temperature": None}, root=root)
    assert "temperature" not in es.load_raw("t", root)
    # An unknown key is still refused BEFORE any of this — a null does not
    # buy a way past the vocabulary.
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("t", {"notAField": None}, root=root)


def test_a_null_sweep_clears_like_every_other_protocol_field(tmp_path):
    """Review round 11, finding 4. ``sweep`` was the one field in the
    vocabulary whose shape gate spelled ``"sweep" in fields`` instead of
    ``fields.get("sweep") is not None``, so an explicit null was refused —
    "sweep must be an object, got NoneType" — before it could reach the
    null-clears loop that the round-10 fix made the promise for. A declared
    grid was removable only by hand-editing the manifest.

    The non-dict, non-null refusal is untouched: a string still cannot be a
    sweep block, and a refusal still writes nothing."""
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    d = es.set_protocol(
        "s", {"sweep": {"selection": {"instrument": "sampledText"}}},
        root=root)
    assert d["sweep"]["selection"]["instrument"] == "sampledText"

    d = es.set_protocol("s", {"sweep": None}, root=root)
    assert "sweep" not in d
    assert "sweep" not in es.load_raw("s", root)
    Manifest.from_dict(es.load_raw("s", root))

    # Nulling a sweep that was never declared is a no-op, not a refusal.
    es.create("t", model_id="org/m", root=root)
    es.set_protocol("t", {"sweep": None}, root=root)
    assert "sweep" not in es.load_raw("t", root)

    # A non-dict, non-null sweep still refuses — and writes nothing, so the
    # valid co-key in the same call does not land either.
    es.set_protocol("s", {"sweep": {"selection": {}}}, root=root)
    for bad in ("everything", ["selection"], 3):
        with pytest.raises(es.ExperimentStoreError) as exc:
            es.set_protocol("s", {"sweep": bad, "taskDescription": "x"},
                            root=root)
        assert "sweep must be an object" in str(exc.value), bad
        assert exc.value.repair_action, bad
    saved = es.load_raw("s", root)
    assert saved["sweep"] == {"selection": {}}
    assert "taskDescription" not in saved


def test_freeze_requires_revision_without_force(tmp_path):
    root = str(tmp_path)
    es.create("s2", model_id="org/m", root=root)  # no revision
    with pytest.raises(es.ExperimentStoreError):
        es.freeze("s2", force=False, cached_revision=lambda m: None, root=root)


def test_freeze_requires_validate_run_when_concepts_present(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s3", model_id="org/m", revision="abc", root=root)
    es.attach("s3", ["french"], root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.freeze("s3", force=False, root=root)
    assert "validate" in str(exc.value)


def test_freeze_rejects_bare_scope_file_and_accepts_complete_evidence(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s4", model_id="org/m", revision="abc", root=root)
    es.attach("s4", ["french"], root=root)
    from steerlab_server.experiment.manifest import Manifest
    scope = Manifest.load("s4", root=root).validation_scope_hash()

    # A run with only a bare scope file is NOT acceptable evidence.
    bare = os.path.join(root, "runs", "x-exp-s4-validate")
    os.makedirs(bare)
    open(os.path.join(bare, "validation-scope-hash.txt"), "w").write(scope)
    with pytest.raises(es.ExperimentStoreError):
        es.freeze("s4", force=False, root=root)

    # A complete validate run (evidence + report with content) IS acceptable.
    full = os.path.join(root, "runs", "y-exp-s4-validate")
    os.makedirs(full)
    json.dump({"schemaVersion": 1, "task": "validate", "validationScopeHash": scope},
              open(os.path.join(full, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"french": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(full, "validation-report.json"), "w"))
    frozen = es.freeze("s4", force=False, root=root)
    assert frozen["status"] == "frozen"


def test_resolve_paths(tmp_path):
    from steerlab_server.experiment import paths
    assert paths.resolve("/abs/path") == "/abs/path"
    assert paths.resolve("rel/x", root=str(tmp_path)) == os.path.join(str(tmp_path), "rel/x")
    assert paths.resolve("") == ""


def test_duplicate_makes_draft(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("orig", model_id="org/m", revision="abc", root=root)
    es.attach("orig", ["french"], root=root)
    es.freeze("orig", force=True, root=root)
    copy = es.duplicate("orig", "orig-v2", root=root)
    assert copy["status"] == "draft"
    assert "freezeHash" not in copy and "frozenBy" not in copy
    # editing the duplicate is allowed
    es.set_protocol("orig-v2", {"temperature": 0.5}, root=root)


# --- the projection carries mode and controlType (2026-09-05) ----------------

def _legacy_conditions():
    """Conditions in the shape the projection has ALWAYS stored for an
    add-by-omission arm: exactly {concept, layer, alpha} per slot, no
    controlType key. Hand-written, so the test cannot follow the code."""
    return [
        {"name": "french-a1", "bandWidth": 1, "alphaInNormUnits": True,
         "slots": [{"concept": "french", "layer": 5, "alpha": 1.0}]},
        {"name": "french-a2", "bandWidth": 3, "alphaInNormUnits": True,
         "slots": [{"concept": "french", "layer": 5, "alpha": 2.0}]},
        {"name": "baseline", "bandWidth": 1, "alphaInNormUnits": True,
         "slots": []},
    ]


def test_an_ablation_condition_and_its_control_round_trip_through_add_conditions(tmp_path):
    """The 2026-09-05 drop: ``_condition_entry`` projected exactly
    {concept, layer, alpha} per slot and no controlType, so an ablation
    authored through add_conditions (the POST …/condition route, and
    ``control_matrix.ablation_control_conditions()`` fed to it) was stored as
    a STEERING condition at α = λ, and its random-direction control as an
    ordinary treatment cell. Same failure class as the 2026-07-27 variant-wire
    bug: the wire spells add by omission, so a dropped mode is
    indistinguishable from steering. The Swift ``ExperimentStore.Slot`` always
    round-tripped ``mode``, so the two engines' authoring verbs disagreed."""
    from steerlab_server.experiment.control_matrix import (
        ablation_control_conditions, control_matrix_conditions)
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    es.attach("s", ["french"], root=root)
    declared = (ablation_control_conditions("french")
                + control_matrix_conditions("french", 5, 2.0))
    stored = es.add_conditions("s", declared, root=root)
    by_name = {c["name"]: c for c in stored["conditions"]}
    # Stored bytes: mode rides on every ablating slot, controlType on the
    # two control cells, and NOTHING else grew a key.
    for name in ("french-ablate-l0p5", "french-ablate-l1", "french-ablate-random"):
        assert by_name[name]["slots"] == [
            {"concept": "french", "layer": 0,
             "alpha": by_name[name]["slots"][0]["alpha"], "mode": "ablate"}]
    assert by_name["french-ablate-random"]["controlType"] == "randomDirectionAblation"
    assert by_name["french-randomMatchedNorm-a2"]["controlType"] == "randomMatchedNorm"
    assert by_name["french-randomMatchedNorm-a2"]["slots"] == [
        {"concept": "french", "layer": 5, "alpha": 2.0}]
    for name in ("french-ablate-l0p5", "french-ablate-l1", "french-a1",
                 "french-a2", "french-neg-a2"):
        assert "controlType" not in by_name[name]
    # Read back through the manifest model — what the run path executes.
    manifest = Manifest.from_dict(es.load_raw("s", root))
    conditions = {c.name: c for c in manifest.conditions}
    assert conditions["french-ablate-l1"].slots[0].is_ablation
    assert conditions["french-ablate-l1"].slots[0].alpha == 1.0
    assert conditions["french-ablate-l1"].control_type is None
    assert conditions["french-ablate-random"].slots[0].is_ablation
    assert conditions["french-ablate-random"].control_type == "randomDirectionAblation"
    assert conditions["french-randomMatchedNorm-a2"].control_type == "randomMatchedNorm"
    assert not conditions["french-randomMatchedNorm-a2"].slots[0].is_ablation
    assert not conditions["french-a2"].slots[0].is_ablation
    assert conditions["french-a2"].control_type is None
    # Re-projecting what was stored is a fixed point: add_conditions replaces
    # by name (sweep re-projection relies on it), and a second pass must not
    # lose what the first wrote.
    again = es.add_conditions("s", list(stored["conditions"]), root=root)
    assert again["conditions"] == stored["conditions"]


def test_a_mode_less_condition_projects_to_its_legacy_bytes():
    """The manifest hash is the study identity, so an arm that never declared
    a mode must project to exactly what it always did — no ``mode`` key, no
    ``controlType`` key — and an EXPLICIT ``add`` is normalised to omission,
    as the Swift encoder does (``Slot.encode`` skips ``.add``)."""
    for legacy in _legacy_conditions():
        assert es._condition_entry(legacy) == legacy
        assert es._condition_entry(dict(legacy, controlType=None)) == legacy
    explicit_add = {"name": "french-a2", "bandWidth": 3, "alphaInNormUnits": True,
                    "slots": [{"concept": "french", "layer": 5, "alpha": 2.0,
                               "mode": "add"}]}
    assert es._condition_entry(explicit_add) == _legacy_conditions()[1]
    # Same document, same content hash, whichever spelling authored it.
    def _doc(conditions):
        return {"name": "s", "modelID": "org/m", "concepts": [],
                "conditions": conditions}
    authored = [es._condition_entry(c) for c in (
        _legacy_conditions()[:1] + [explicit_add] + _legacy_conditions()[2:])]
    assert (Manifest.from_dict(_doc(authored)).content_hash()
            == Manifest.from_dict(_doc(_legacy_conditions())).content_hash())


def test_a_frozen_study_of_add_by_omission_conditions_keeps_its_freeze_hash(tmp_path):
    """End to end through the store: author legacy-shaped arms, freeze, and
    check the stamped freezeHash against a document whose conditions are
    HAND-WRITTEN in the pre-2026-09-05 shape. A key the projection had grown
    on every add-by-omission arm would show up here as a different hash."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)
    es.add_conditions("s", _legacy_conditions(), root=root)
    frozen = es.freeze("s", force=True, root=root)
    stored = es.load_raw("s", root)
    assert stored["conditions"] == _legacy_conditions()
    assert all("mode" not in slot for c in stored["conditions"]
               for slot in c["slots"])
    assert all("controlType" not in c for c in stored["conditions"])
    expected = dict(stored)
    expected["conditions"] = _legacy_conditions()
    assert Manifest.from_dict(expected).content_hash() == frozen["freezeHash"]
    assert stored["freezeHash"] == frozen["freezeHash"]


def test_condition_projection_refuses_what_neither_engine_can_run():
    """Closed vocabularies, typed refusals with a repair, nothing written:
    an unknown ``mode`` would make the Mac engine unable to decode the
    manifest (``InterventionPlan.Mode`` is a closed enum); an unknown
    ``controlType`` runs as an ordinary treatment (Swift
    ``knownControlTypes`` twin); a control with no slots has nothing to
    substitute into; and a control paired with the wrong slot mode is a cell
    that duplicates the treatment on one engine or the other."""
    base = {"name": "arm", "bandWidth": 1, "alphaInNormUnits": False}
    steer = [{"concept": "french", "layer": 5, "alpha": 1.0}]
    ablate = [{"concept": "french", "layer": 0, "alpha": 1.0, "mode": "ablate"}]

    def refusal(condition):
        with pytest.raises(es.ExperimentStoreError) as excinfo:
            es._condition_entry(condition)
        assert excinfo.value.repair_action
        return str(excinfo.value)

    detail = refusal(dict(base, slots=[dict(steer[0], mode="remove")]))
    assert "'remove'" in detail and "add | ablate" in detail
    detail = refusal(dict(base, slots=steer, controlType="shuffledLabels"))
    assert "'shuffledLabels'" in detail
    assert "randomMatchedNorm | randomDirectionAblation" in detail
    detail = refusal(dict(base, slots=[], controlType="randomMatchedNorm"))
    assert "no slots" in detail
    detail = refusal(dict(base, slots=ablate, controlType="randomMatchedNorm"))
    assert "randomDirectionAblation" in detail
    detail = refusal(dict(base, slots=steer, controlType="randomDirectionAblation"))
    assert "randomMatchedNorm" in detail
    # The matched pairs are accepted, and only the declared keys appear.
    assert es._condition_entry(
        dict(base, slots=ablate, controlType="randomDirectionAblation")) == {
        "name": "arm", "bandWidth": 1, "alphaInNormUnits": False,
        "slots": [{"concept": "french", "layer": 0, "alpha": 1.0,
                   "mode": "ablate"}],
        "controlType": "randomDirectionAblation"}
    assert es._condition_entry(
        dict(base, slots=steer, controlType="randomMatchedNorm")) == {
        "name": "arm", "bandWidth": 1, "alphaInNormUnits": False,
        "slots": [{"concept": "french", "layer": 5, "alpha": 1.0}],
        "controlType": "randomMatchedNorm"}
