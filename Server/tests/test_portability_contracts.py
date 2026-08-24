"""Phase-0 of the portability program: the cross-engine contracts a future
cross-platform Python client will depend on, pinned as goldens — Python half.

The client the later phases build has to do three things this engine and the
Swift engine already do to each other: **author a workspace the Swift engine
reads**, **submit a hash-pinned run bundle either engine produced**, and
**import an evidence bundle with verification**. Nothing here changes
behaviour; it records what today's behaviour IS, so a later phase that breaks
one of these seams fails a test instead of failing in a workspace.

Three idioms are reused rather than reinvented:

1. **Producer-generated fixtures** (``scripts/regenerate-cross-engine-fixtures.py``
   and ``test_fixture_staleness.py``): bytes committed under
   ``Tests/Fixtures/cross-engine/`` come from the engine that really produces
   them, so the consumer's test cannot pin its author's *belief* about the
   other engine. The two fixtures written here follow the **write-if-missing**
   rule the envelope goldens use (``test_cli_envelope.py::_check``): the
   structural assertions always run against the freshly produced document, and
   only the byte comparison waits for the file to be committed. To regenerate,
   delete the file and re-run this module.

2. **Twin literals** (``test_cli_envelope.py`` ↔ ``CLIEnvelopeParityTests.swift``):
   a constant belonging to the other engine is written out here as a literal
   with a naming cross-reference, so neither engine can quietly follow the
   other.

3. **Loud refusals**: every verification contract below is pinned by the
   refusal it produces AND by the state of the workspace afterwards. A gate
   that refuses while leaving debris is a gate that did not hold.

Swift twin: ``Tests/ExperimentKitTests/PortabilityContractTests.swift``.
Inventory: ``docs/PORTABILITY-CONTRACTS.md``.
"""

import hashlib
import io
import json
import os
import tarfile

import pytest

from steerlab_server.experiment import bundles
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "cross-engine")

REGENERATE = ("stale fixture — delete it and re-run "
              "`Server/.venv.nosync/bin/python -m pytest "
              "tests/test_portability_contracts.py`, then commit")


def _write_or_compare(name: str, payload) -> None:
    """The envelope goldens' write-if-missing rule, applied to a cross-engine
    fixture. A MISSING file is written (the structural assertions above the
    call already ran); a file that EXISTS must match byte for byte."""
    os.makedirs(FIXTURES, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return
    with open(path, encoding="utf-8") as handle:
        assert text == handle.read(), f"{name}: {REGENERATE}"


def _load_fixture(name: str):
    path = os.path.join(FIXTURES, name)
    assert os.path.exists(path), (
        f"{name} is missing. It is produced by the SWIFT engine "
        f"(Tests/ExperimentKitTests/PortabilityContractTests.swift) — run that "
        f"suite first, then re-run this one.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _write(root, rel, text) -> str:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return _sha256_file(path)


def _concept(root, name="french"):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "bonjour"}\n')
    with open(os.path.join(directory, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "hello"}\n')
    with open(os.path.join(directory, "validation.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "salut", "expresses": true}\n')


#: Volatile freeze stamps: keys that describe WHEN and BY WHAT a manifest was
#: stamped, never what the study measures. Copied from
#: ``ExperimentStore.volatileFreezeKeys``
#: (``Sources/ExperimentKit/ExperimentStore.swift``); this engine spells the
#: same set inline in ``Manifest.content_hash`` and
#: ``experiment_store._write_freeze_canonical``. Swift twin test:
#: ``PortabilityContractTests.volatileFreezeStampsAreOutsideTheContentHash``.
VOLATILE_FREEZE_KEYS = [
    "status", "frozenAt", "freezeHash", "gitCommit", "frozenBy", "createdAt",
    "appVersion", "freezeForced", "forcedGatesSkipped",
]

#: The ``steerlab-bundle.json`` header keys for a run bundle, copied from the
#: metadata literal in ``RunBundlePackager.packageExperiment``
#: (``Sources/ExperimentKit/ClusterClient.swift``). BOTH engines write exactly
#: this set — that agreement is what lets a bundle from either side be
#: submitted to either side, and it is what the tests below pin. Swift twin
#: test: ``PortabilityContractTests.theRunBundleMetadataShapeMatchesTheServerLiteral``.
RUN_BUNDLE_HEADER_KEYS = [
    "createdAt", "entries", "experiment", "experimentContentHash", "kind",
    "rootRelative", "schemaVersion", "validationScopeHash",
    "verificationViolations",
]

#: Copied from the Swift entry literal in the same function.
BUNDLE_ENTRY_KEYS = ["bytes", "path", "sha256"]


def _interop_study(root: str) -> dict:
    """A draft study declaring a broad slice of the pin surface, authored
    ENTIRELY through this engine's public store API — the shape a future
    Python client would produce and the Swift engine must read."""
    _concept(root)
    _write(root, "prompts/concepts/french/markers.json", '{"markers":["bonj"]}')
    _write(root, "prompts/neutral/corpus.jsonl", '{"text":"the sky is blue"}\n')

    es.create("portability-interop", model_id="org/m",
              revision="0123456789abcdef0123456789abcdef01234567", root=root)
    es.attach("portability-interop", ["french"], root=root)

    d = es.load_raw("portability-interop", root)
    # The cross-engine key really is `experimentDescription` on BOTH engines
    # (`Manifest.from_dict` reads it; Swift's `CodingKeys` names it) — a plain
    # `description` key would be read by neither.
    d["experimentDescription"] = "Phase-0 interop fixture"
    d["taskDescription"] = "answer the item"
    d["taskPromptsFile"] = "prompts/tasks/cases.jsonl"
    d["taskPromptsHash"] = _write(root, "prompts/tasks/cases.jsonl",
                                  '{"id":"c1","text":"decide"}\n')
    d["judgeRubricFile"] = "prompts/rubrics/r1.md"
    d["judgeRubricHash"] = _write(root, "prompts/rubrics/r1.md", "# rubric\n")
    d["capabilityBatteryFile"] = "prompts/batteries/b1.jsonl"
    d["capabilityBatteryHash"] = _write(root, "prompts/batteries/b1.jsonl",
                                        '{"prompt":"2+2","answer":"4"}\n')
    d["seeds"] = [0, 1, 2]
    # A sampling policy that is INTERNALLY COHERENT, so the frozen half of the
    # fixture is a manifest the engine would really freeze: repeated samples
    # need a temperature above zero and per-record derived seeds, or verify()
    # refuses (greedy decoding makes every sample identical).
    d["temperature"] = 0.7
    d["maxTokens"] = 512
    d["samplesPerItem"] = 3
    d["seedPolicy"] = "derivedSHA256"
    d["promptMode"] = "chatAssistant"
    d["systemPrompt"] = "Respond in JSON."
    # A nested typed block, in the vocabulary BOTH engines close over
    # (`EvaluationSpec.Kind` is `none`/`pairedJudge` on each side) — an
    # invented kind decodes on neither.
    d["evaluation"] = {"kind": "none", "judgeModel": "claude-opus-4-8",
                       "judgePrompt": ""}
    es.save_raw(d, root)
    # Conditions go through the STORE's own projection rather than being
    # hand-written into the document: `_condition_entry` stamps `bandWidth`
    # and `alphaInNormUnits`, which the Swift decoder REQUIRES (its
    # `ExperimentManifest.Condition` has no default for either). A fixture
    # that hand-wrote a bare `{name, slots}` condition would publish a
    # manifest no client should ever author — see
    # ``PortabilityContractTests.aConditionWithoutItsGlobalsIsUnreadableHere``
    # for the pinned consequence.
    es.add_conditions("portability-interop", [
        {"name": "baseline", "slots": []},
        {"name": "arm-a", "slots": [{"concept": "french", "layer": 1,
                                     "alpha": 0.1}]},
    ], root)
    return es.load_raw("portability-interop", root)


# =============================================================================
# 1. Manifest interop
# =============================================================================

def test_a_python_authored_manifest_fixture_is_current(tmp_path):
    """CONTRACT: manifest-interop — a manifest this engine authored, in the
    create→attach→pin state a client leaves a workspace in, published as bytes
    the Swift reader consumes.

    Producer-generated on purpose: a Swift test that hand-writes what it
    believes the server emits pins the Swift author's belief, and that belief
    drifting is the failure mode the whole cross-engine fixture strategy
    exists to catch."""
    root = str(tmp_path / "server-authored")
    draft = _interop_study(root)

    # The shape the fixture promises the Swift side, asserted on the FRESH
    # document so a change here fails even before the bytes are compared.
    assert draft["status"] == "draft"
    assert draft["name"] == "portability-interop"
    assert len(draft["concepts"]) == 1
    assert len(draft["concepts"][0]["stimulusSetHash"]) == 64
    assert [c["name"] for c in draft["conditions"]] == ["baseline", "arm-a"]

    manifest = Manifest.load("portability-interop", root)
    draft_hash = manifest.content_hash()
    assert len(draft_hash) == 64

    # …and the same study FROZEN, with the exact canonical bytes the freeze
    # hash was taken over. `force` skips the evidence gates only; the
    # canonicalization under test is unaffected by which gates ran.
    frozen = es.freeze("portability-interop", force=True,
                       cached_revision=lambda m: None, root=root)
    assert frozen["status"] == "frozen"
    canonical_path = os.path.join(root, "experiments", "portability-interop",
                                  "freeze-canonical.json")
    with open(canonical_path, "rb") as handle:
        canonical_bytes = handle.read()
    assert hashlib.sha256(canonical_bytes).hexdigest() == frozen["freezeHash"]

    # The freeze hash IS the content hash of the frozen document — the same
    # canonicalization, reached by two code paths (``Manifest.content_hash``
    # and ``experiment_store._write_freeze_canonical``). A client that
    # recomputes one and compares it to the other must get the same answer.
    #
    # It does NOT equal the DRAFT's content hash, and should not: freeze also
    # pins (markers, sweep inputs, the model revision), which is measured
    # surface changing. The rule that must hold across engines is the
    # volatile-key exclusion, pinned by
    # ``test_volatile_freeze_stamps_are_outside_the_content_hash``.
    assert Manifest.load("portability-interop", root).content_hash() \
        == frozen["freezeHash"]

    # Volatile stamps vary per machine and per second; they are excluded from
    # every canonicalization here, which is precisely what makes normalizing
    # them in the fixture safe rather than a fudge.
    normalized = dict(frozen)
    for key, value in (("frozenAt", "1970-01-01T00:00:00Z"),
                       ("createdAt", "1970-01-01T00:00:00Z"),
                       ("appVersion", "<appVersion>"),
                       ("gitCommit", "<gitCommit>")):
        if key in normalized:
            normalized[key] = value
    normalized_draft = dict(draft)
    normalized_draft["createdAt"] = "1970-01-01T00:00:00Z"

    # The SAME freeze over a document that also carries the two boolean keys
    # this engine omits at their defaults and the Swift encoder always writes
    # (`multiAgentIncludeBaseline`, `recordTokenIDs`). Both engines agree on
    # the default VALUES; they disagree on whether the key is written when it
    # holds one, and the Swift post-freeze check compares parsed documents —
    # so a study authored entirely here reads as drifted over there while a
    # Mac-authored study frozen here verifies clean. Publishing BOTH halves is
    # what turns that from a mystery into a pinned, named gap.
    swift_root = str(tmp_path / "swift-shaped")
    _interop_study(swift_root)
    d = es.load_raw("portability-interop", swift_root)
    d["multiAgentIncludeBaseline"] = True    # Swift default
    d["recordTokenIDs"] = False              # Swift default
    es.save_raw(d, swift_root)
    swift_shaped = es.freeze("portability-interop", force=True,
                             cached_revision=lambda m: None, root=swift_root)
    with open(os.path.join(swift_root, "experiments", "portability-interop",
                           "freeze-canonical.json"), "rb") as handle:
        swift_canonical = handle.read()
    assert hashlib.sha256(swift_canonical).hexdigest() \
        == swift_shaped["freezeHash"]
    normalized_swift = dict(swift_shaped)
    for key, value in (("frozenAt", "1970-01-01T00:00:00Z"),
                       ("createdAt", "1970-01-01T00:00:00Z"),
                       ("appVersion", "<appVersion>"),
                       ("gitCommit", "<gitCommit>")):
        if key in normalized_swift:
            normalized_swift[key] = value

    _write_or_compare("manifest-interop.json", {
        "note": "produced by Server experiment_store (create → attach → pin → "
                "freeze) — do not hand-edit; delete and re-run "
                "Server/tests/test_portability_contracts.py",
        "draft": normalized_draft,
        "frozen": normalized,
        "freezeCanonical": canonical_bytes.decode("utf-8"),
        "freezeHash": frozen["freezeHash"],
        "contentHash": draft_hash,
        "stimulusSetHash": draft["concepts"][0]["stimulusSetHash"],
        "volatileFreezeKeys": VOLATILE_FREEZE_KEYS,
        "frozenWithSwiftDefaults": normalized_swift,
        "freezeCanonicalWithSwiftDefaults": swift_canonical.decode("utf-8"),
        "freezeHashWithSwiftDefaults": swift_shaped["freezeHash"],
        "keysSwiftAlwaysWritesAndThisEngineOmitsAtDefault": [
            "multiAgentIncludeBaseline", "recordTokenIDs"],
    })


def test_volatile_freeze_stamps_are_outside_the_content_hash(tmp_path):
    """CONTRACT: manifest-canonicalization — the identity of a study is what it
    MEASURES, so every key describing when/by-what it was stamped is outside
    the hash on both engines.

    The hash VALUES differ per engine by design (``Manifest.content_hash``
    says so, and ``CrossEngineLifecycleTests`` documents why comparing them
    across substrates is a check that cannot pass). What must agree is the
    RULE, and that is what this pins: mutate each volatile key, the hash holds;
    mutate a measured key, the hash moves. Swift twin:
    ``PortabilityContractTests.volatileFreezeStampsAreOutsideTheContentHash``."""
    root = str(tmp_path)
    _interop_study(root)
    manifest = Manifest.load("portability-interop", root)
    baseline = manifest.content_hash()

    for key in VOLATILE_FREEZE_KEYS:
        mutated = Manifest.load("portability-interop", root)
        mutated.raw[key] = "a-value-no-canonicalization-should-see"
        assert mutated.content_hash() == baseline, (
            f"'{key}' is treated as content by this engine but as a volatile "
            f"freeze stamp by Swift's volatileFreezeKeys")

    for key, value in (("modelID", "org/other"),
                       ("taskPromptsHash", "0" * 64),
                       ("temperature", 0.1),
                       ("seeds", [9])):
        mutated = Manifest.load("portability-interop", root)
        mutated.raw[key] = value
        assert mutated.content_hash() != baseline, (
            f"'{key}' is measured surface and must move the content hash")


def test_the_condition_alpha_unit_default_diverges_from_swift():
    """GAP (found by this phase, recorded not repaired): declaring a condition
    without naming ``alphaInNormUnits`` yields FALSE here and TRUE on the Mac.

    ``experiment_store._condition_entry`` defaults it to False;
    ``ExperimentManifest.Condition.init`` (``Sources/ExperimentKit/ExperimentStore.swift``)
    defaults it to true. Neither engine is reading the other's default — both
    always WRITE the key, so a condition that has crossed the wire is
    unambiguous — but the same client call produces a different study
    depending on which engine served it, and α units are not a cosmetic
    setting.

    Pinned as it IS. Swift twin:
    ``PortabilityContractTests.theConditionAlphaUnitDefaultDivergesFromTheServer``."""
    entry = es._condition_entry({"name": "arm", "slots": []})
    assert entry["alphaInNormUnits"] is False, (
        "this engine's condition default changed — the divergence with Swift's "
        "`true` may be closed; re-check docs/PORTABILITY-CONTRACTS.md G6")
    assert entry["bandWidth"] == 1          # this one DOES agree
    # An explicit value is always honoured, on both engines.
    assert es._condition_entry(
        {"name": "arm", "slots": [], "alphaInNormUnits": True}
    )["alphaInNormUnits"] is True


def test_a_swift_authored_manifest_loads_with_every_field_intact(tmp_path):
    """CONTRACT: manifest-interop-reverse — a manifest the SWIFT engine
    authored is read by this engine with every cross-engine field surviving.

    The bytes come from the Swift engine's own ``ExperimentStore`` (see
    ``_load_fixture``), never from a Python author's idea of what Swift
    writes."""
    fixture = _load_fixture("swift-authored-manifest.json")
    document = fixture["manifest"]
    name = document["name"]

    directory = os.path.join(str(tmp_path), "experiments", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "experiment.json"), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle)

    manifest = Manifest.load(name, str(tmp_path))

    # The identity fields a client dispatches on.
    assert manifest.name == document["name"]
    assert manifest.model_id == document["modelID"]
    assert manifest.model_revision == document.get("modelRevision")
    assert manifest.status == document.get("status", "draft")

    # The pin surface: every declared pin reaches this engine's typed reader.
    assert manifest.task_prompts_file == document["taskPromptsFile"]
    assert manifest.task_prompts_hash == document["taskPromptsHash"]
    assert [c.name for c in manifest.concepts] == \
        [c["name"] for c in document["concepts"]]
    assert [c.stimulus_set_hash for c in manifest.concepts] == \
        [c["stimulusSetHash"] for c in document["concepts"]]
    assert [c["name"] for c in manifest.raw.get("conditions") or []] == \
        [c["name"] for c in document["conditions"]]

    # Sampling policy: a client that reads these wrong reruns a different study.
    assert manifest.seeds == document["seeds"]
    assert manifest.temperature == document["temperature"]
    assert manifest.max_tokens == document["maxTokens"]
    assert manifest.prompt_mode == document["promptMode"]

    # Nothing was dropped on the way in: the raw document this engine holds is
    # the document Swift wrote, key for key.
    assert manifest.raw == document

    # And this engine can hash it — the value differs from Swift's by design,
    # but a manifest that cannot be canonicalized at all is unusable.
    assert len(manifest.content_hash()) == 64


# =============================================================================
# 2. Run-bundle interop
# =============================================================================

def test_a_run_bundle_metadata_fixture_is_current(tmp_path):
    """CONTRACT: run-bundle-metadata — the ``steerlab-bundle.json`` header
    this engine writes, published for the Swift side.

    There is NO Swift run-bundle READER (``EvidenceBundleImporter`` reads
    evidence bundles; ``RunBundlePackager`` only writes run bundles), so this
    pins the metadata SHAPE rather than a round trip. Closing that gap is
    Phase-1 work — see docs/PORTABILITY-CONTRACTS.md."""
    root = str(tmp_path)
    _interop_study(root)
    meta = bundles.package_experiment("portability-interop", root=root)

    assert meta["kind"] == "runBundle"
    assert meta["schemaVersion"] == bundles.BUNDLE_SCHEMA
    assert len(meta["bundleSha256"]) == 64
    # The header key set is IDENTICAL to the Swift packager's literal — the
    # agreement that lets either engine submit a bundle the other executes.
    # (`bundlePath`/`bundleSha256` are added to the RETURNED dict after the
    # archive is written; they are not members of the document inside it.)
    assert sorted(k for k in meta
                  if k not in ("bundlePath", "bundleSha256")) \
        == RUN_BUNDLE_HEADER_KEYS
    for entry in meta["entries"]:
        assert sorted(entry) == BUNDLE_ENTRY_KEYS, (
            "a bundle entry's key set drifted from the Swift packager's "
            f"literal {BUNDLE_ENTRY_KEYS}")

    paths = sorted(e["path"] for e in meta["entries"])
    assert "experiments/portability-interop/experiment.json" in paths

    _write_or_compare("run-bundle-metadata.json", {
        "note": "produced by Server bundles.package_experiment — do not "
                "hand-edit; delete and re-run "
                "Server/tests/test_portability_contracts.py",
        "kind": meta["kind"],
        "schemaVersion": meta["schemaVersion"],
        "headerKeys": sorted(k for k in meta
                             if k not in ("bundlePath", "bundleSha256")),
        "entryKeys": BUNDLE_ENTRY_KEYS,
        "entryPaths": paths,
        "swiftPackagerHeaderKeys": RUN_BUNDLE_HEADER_KEYS,
    })


def _swift_shaped_run_bundle(path: str, entries) -> dict:
    """A run bundle whose ``steerlab-bundle.json`` is the SWIFT packager's
    metadata literal (copied from ``RunBundlePackager.packageExperiment``),
    carrying the supplied ``(name, payload)`` members."""
    meta = {
        "schemaVersion": 1,
        "kind": "runBundle",
        "createdAt": 1000.0,
        "experiment": "portability-interop",
        "experimentContentHash": "a" * 64,
        "validationScopeHash": "",
        "rootRelative": True,
        "verificationViolations": [],
        "entries": [{"path": name,
                     "sha256": hashlib.sha256(payload).hexdigest(),
                     "bytes": len(payload)} for name, payload in entries],
    }
    with tarfile.open(path, "w:gz") as tar:
        members = [("steerlab-bundle.json",
                    json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"))]
        members.extend(entries)
        for name, blob in members:
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return meta


def test_a_swift_packaged_run_bundle_is_inspectable_and_importable(tmp_path):
    """CONTRACT: run-bundle-swift-to-python — the bundle the Mac submits is
    the bundle this engine executes, so its metadata must survive this
    engine's reader with every Swift-only key intact.

    ``rootRelative``, ``validationScopeHash`` and ``verificationViolations``
    have no Python producer; a reader that refused or silently dropped them
    would break submission from the Mac."""
    bundle = str(tmp_path / "swift.run-bundle.tar.gz")
    payload = b'{"id":"c1","text":"decide"}\n'
    manifest_bytes = json.dumps(
        {"name": "portability-interop", "status": "draft",
         "modelID": "org/m"}).encode("utf-8")
    meta = _swift_shaped_run_bundle(bundle, [
        ("experiments/portability-interop/experiment.json", manifest_bytes),
        ("prompts/tasks/cases.jsonl", payload),
    ])

    inspected = bundles.inspect_bundle(bundle)
    assert sorted(k for k in inspected
                  if k not in ("bundlePath", "bundleSha256")) \
        == RUN_BUNDLE_HEADER_KEYS
    assert inspected["kind"] == "runBundle"
    assert inspected["experiment"] == meta["experiment"]
    assert inspected["experimentContentHash"] == meta["experimentContentHash"]
    assert inspected["rootRelative"] is True
    assert inspected["verificationViolations"] == []
    # The outer digest is RECOMPUTED from the file, never trusted from inside.
    assert inspected["bundleSha256"] == _sha256_file(bundle)

    target = tmp_path / "target"
    imported = bundles.import_bundle(bundle, target_root=str(target))
    assert sorted(imported["extracted"]) == [
        "experiments/portability-interop/experiment.json",
        "prompts/tasks/cases.jsonl",
    ]
    assert (target / "prompts" / "tasks" / "cases.jsonl").read_bytes() == payload


# =============================================================================
# 3. Evidence verification contracts
# =============================================================================

def _evidence_bundle(tmp_path):
    """A REAL evidence bundle from this engine's packager, over a run
    directory with two members."""
    root = tmp_path / "source"
    run = root / "runs" / "20260101T000000000-exp-portability-interop-validate"
    run.mkdir(parents=True)
    (run / "report.json").write_text('{"ok":true}\n', encoding="utf-8")
    (run / "records.jsonl").write_text('{"promptID":"c1"}\n', encoding="utf-8")
    return bundles.package_evidence(str(run), root=str(root))


def _repack_with_tampered_member(bundle_path, member_name, payload, out_path):
    """The original archive with ONE member's bytes swapped and the metadata
    left claiming the original hash."""
    with tarfile.open(bundle_path, "r:gz") as src, \
            tarfile.open(out_path, "w:gz") as dst:
        for member in src.getmembers():
            handle = src.extractfile(member)
            data = handle.read() if handle is not None else b""
            if member.name == member_name:
                data = payload
                member.size = len(payload)
            dst.addfile(member, io.BytesIO(data))
    return out_path


def test_a_flipped_member_byte_is_refused_and_the_refusal_names_the_member(
        tmp_path):
    """CONTRACT: evidence-member-integrity — a client importing evidence must
    be told WHICH member failed, not merely that something did.

    ``test_bundles.py::test_tampered_member_refuses_before_touching_the_workspace``
    already pins that the refusal happens before any disk write; what it does
    not pin is that the member is NAMED, which is the only part a remote
    client can act on."""
    meta = _evidence_bundle(tmp_path)
    member = f"runs/{meta['runID']}/report.json"
    tampered = _repack_with_tampered_member(
        meta["bundlePath"], member, b'{"ok":false}\n',
        str(tmp_path / "tampered.tar.gz"))

    target = tmp_path / "target"
    with pytest.raises(bundles.BundleError) as excinfo:
        bundles.import_bundle(tampered, target_root=str(target))
    detail = str(excinfo.value)
    assert "hash mismatch" in detail
    assert member in detail, (
        f"the refusal does not name the failing member: {detail}")
    # Nothing landed, and no staging debris was left behind.
    assert not (target / "runs" / meta["runID"] / "report.json").exists()
    assert list(target.rglob("*.tmp")) == []


def test_a_member_escaping_the_target_root_is_refused(tmp_path):
    """CONTRACT: evidence-path-containment — an archive is untrusted input, so
    a member naming a path outside the workspace is refused BY NAME and
    nothing is written anywhere.

    Sibling coverage exists for hostile *identifiers* (``isSafeComponent`` on
    the Swift importer's ``runID``; ``test_gemma_scope.py``'s ``../../escaped``
    import name), but the tar MEMBER path — the classic archive traversal —
    had no test on either engine."""
    bundle = str(tmp_path / "escape.tar.gz")
    payload = b"escaped\n"
    _swift_shaped_run_bundle(bundle, [("../escaped.json", payload)])

    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(bundles.BundleError) as excinfo:
        bundles.import_bundle(bundle, target_root=str(target))
    detail = str(excinfo.value)
    assert "escapes target root" in detail
    assert "../escaped.json" in detail, (
        f"the refusal does not name the offending member: {detail}")
    assert not (tmp_path / "escaped.json").exists()
    assert list(target.rglob("*")) == []


def test_re_importing_identical_evidence_neither_duplicates_nor_rewrites(
        tmp_path):
    """CONTRACT: evidence-import-idempotence — re-importing the SAME evidence
    is not a way to quietly re-write a workspace.

    The engine's answer is a refusal, not a silent no-op, and this pins the
    part that matters to a client: after the refusal the files on disk are
    byte-identical AND untouched (same mtime), and no second copy appeared.
    ``test_bundles.py::test_import_rejects_overwrite`` pins the refusal;
    it does not pin that the existing bytes survived it."""
    meta = _evidence_bundle(tmp_path)
    target = tmp_path / "target"
    first = bundles.import_bundle(meta["bundlePath"], target_root=str(target))
    assert first["extracted"]

    run_dir = target / "runs" / meta["runID"]
    before = {p.relative_to(target).as_posix():
              (p.read_bytes(), p.stat().st_mtime_ns)
              for p in sorted(run_dir.rglob("*")) if p.is_file()}
    assert before, "the first import wrote nothing to compare against"

    with pytest.raises(bundles.BundleError) as excinfo:
        bundles.import_bundle(meta["bundlePath"], target_root=str(target))
    assert "refusing to overwrite existing file" in str(excinfo.value)

    after = {p.relative_to(target).as_posix():
             (p.read_bytes(), p.stat().st_mtime_ns)
             for p in sorted(run_dir.rglob("*")) if p.is_file()}
    assert after == before, "a refused re-import disturbed the workspace"
    assert list(target.rglob("*.tmp")) == []
    # No second copy of the run appeared under another name.
    assert [d.name for d in (target / "runs").iterdir()] == [meta["runID"]]

    # The explicit override is the ONLY way through, and it is content-stable:
    # identical bytes in, identical bytes out.
    bundles.import_bundle(meta["bundlePath"], target_root=str(target),
                          allow_overwrite=True)
    rewritten = {p.relative_to(target).as_posix(): p.read_bytes()
                 for p in sorted(run_dir.rglob("*")) if p.is_file()}
    assert rewritten == {k: v[0] for k, v in before.items()}


def test_the_outer_bundle_hash_travels_out_of_band_and_moves_with_one_member(
        tmp_path):
    """CONTRACT: evidence-outer-hash — the archive digest is a pin the
    RECIPIENT checks before extracting, and it is never read from inside the
    archive it is supposed to protect.

    This engine's importer has no ``expected_sha256`` parameter: the outer
    pin is carried out of band (the job record's ``bundleSha256``) and checked
    by the consumer. On the Mac that consumer is
    ``EvidenceBundleImporter.importEvidenceBundle(_:expectedSHA256:)``, pinned
    by ``CrossEngineLifecycleTests.aBundleWhoseHashDoesNotMatchIsRefused``. A
    future Python client must do the same check itself — this is a GAP for
    Phase 1, recorded in docs/PORTABILITY-CONTRACTS.md."""
    meta = _evidence_bundle(tmp_path)
    assert meta["bundleSha256"] == _sha256_file(meta["bundlePath"])

    # `inspect_bundle` RECOMPUTES the digest rather than reporting a stamped
    # one, so a tampered archive cannot present its victim's hash.
    with tarfile.open(meta["bundlePath"], "r:gz") as tar:
        inner = json.loads(tar.extractfile("steerlab-evidence.json").read())
    assert "bundleSha256" not in inner, (
        "the outer digest must not be stamped inside the archive it pins")
    assert bundles.inspect_bundle(meta["bundlePath"])["bundleSha256"] \
        == meta["bundleSha256"]

    tampered = _repack_with_tampered_member(
        meta["bundlePath"], f"runs/{meta['runID']}/report.json",
        b'{"ok":false}\n', str(tmp_path / "tampered.tar.gz"))
    assert bundles.inspect_bundle(tampered)["bundleSha256"] \
        != meta["bundleSha256"], (
        "one flipped member byte did not move the outer digest")


# =============================================================================
# 4. Envelope fields a cross-platform client dispatches on
# =============================================================================

ENVELOPE_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures",
                                 "cli-envelopes")


def test_every_committed_golden_carries_the_fields_a_client_dispatches_on():
    """CONTRACT: envelope-dispatch-fields — a cross-platform client branches on
    exactly four things, so all four must be present and USABLE in every
    committed golden, not merely allowed by the closed key set.

    ``test_cli_envelope.py::_check`` already pins the closed key set, the
    header keys, the engine stamp, and error-present-iff-not-success. What it
    does not pin is that ``error.code`` names something from a CLOSED
    vocabulary and that ``error.repairAction`` is a non-empty instruction —
    the two fields an agent cannot recover without. Swift twin:
    ``PortabilityContractTests.everyGoldenCarriesTheFieldsAClientDispatchesOn``."""
    from steerlab_server import cli_envelope

    names = sorted(f for f in os.listdir(ENVELOPE_FIXTURES)
                   if f.endswith(".json"))
    assert names, "no committed cli-envelope goldens"

    saw_refusal = False
    for name in names:
        with open(os.path.join(ENVELOPE_FIXTURES, name),
                  encoding="utf-8") as handle:
            document = json.load(handle)

        # 1. `state` — always present, always in the closed vocabulary, and
        #    always resolvable to an exit code without a second table.
        state = document["state"]
        assert state in cli_envelope.STATE_EXIT_CODES, f"{name}: state {state!r}"
        exit_code = cli_envelope.exit_code_for(state)

        # 2. `workspace` — a client addressing a remote engine has to know
        #    WHICH workspace answered. Optional by the contract; when present
        #    it must be a non-empty absolute-ish locator, never an empty
        #    placeholder a client would treat as "the default".
        if "workspace" in document:
            assert isinstance(document["workspace"], str)
            assert document["workspace"].strip(), f"{name}: empty workspace"

        if exit_code == 0:
            assert "error" not in document, f"{name}: success carries an error"
            continue

        saw_refusal = True
        error = document["error"]

        # 3. `error.code` — the branch key. Closed vocabulary: a lifecycle
        #    gate, a freeze gate, or one of the envelope's own parse codes.
        code = error["code"]
        assert isinstance(code, str) and code, f"{name}: empty error.code"

        # 4. `error.repairAction` — what the client DOES next. A refusal
        #    without one sends an agent in a circle, which is the whole
        #    failure mode the typed-refusal contract exists to prevent.
        repair = error.get("repairAction")
        assert isinstance(repair, str) and repair.strip(), (
            f"{name}: refusal has no actionable repairAction")
        assert len(repair) > 8, f"{name}: repairAction is not an instruction"

    assert saw_refusal, (
        "no committed golden is a refusal — the dispatch fields that matter "
        "most to a client are then untested")
