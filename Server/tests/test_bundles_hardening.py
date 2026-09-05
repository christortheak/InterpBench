"""Bundle-integrity hardening (external review, 2026-09-05).

Two defects, both in the seam between what a bundle's METADATA claims and what
the importer DERIVES from it. The transaction in ``bundles.py`` is careful
about members — containment, hash coverage, staging, rollback, a commit that
re-enforces what preflight decided — and both defects live just outside that
care.

* **ENG-01** — the portable pipeline ledger's destination is derived from the
  metadata (``<target>/runs/<runID>/pipeline-portable.json``) and was never
  held to the containment rule every archive MEMBER passes. A validly pinned
  bundle whose ``runID`` was ``../../outside`` therefore wrote outside the
  import root. The outer hash cannot help: it legitimately pins an archive
  whose metadata is unsafe.

* **ENG-04** — a member's pinned digest was checked for ABSENCE but never for
  SHAPE, and the comparison itself was guarded by a truthiness test
  (``if planned.expected and digest != planned.expected``). So ``"sha256": ""``
  passed preflight AND disabled the comparison: an unverified member landed in
  the canonical workspace.
"""

import hashlib
import io
import json
import os
import tarfile
import time

import pytest

from steerlab_server.experiment import bundles

from test_bundles import _bundle_with_members, _no_leftovers


LEDGER = "pipeline-portable.json"
_ABSENT = object()


def _ledger_bundle(path, *, run_id, members=(),
                   ledger=b'{"kind":"pipelinePortable"}\n', pin=_ABSENT):
    """A well-formed bundle carrying a hash-pinned portable ledger, whose
    ``runID`` — the only thing that decides where the ledger lands — is the
    caller's to choose."""
    extra = {"runID": run_id}
    extra["pipelinePortableSha256"] = (
        hashlib.sha256(ledger).hexdigest() if pin is _ABSENT else pin)
    return _bundle_with_members(path,
                                [*members, ("steerlab-pipeline.json", ledger)],
                                extra=extra)


def _target_with_runs(tmp_path):
    """A target root that already holds ``runs/`` — the ordinary state of a
    workspace, and what a traversal needs to walk back out of."""
    target = tmp_path / "target"
    (target / "runs").mkdir(parents=True)
    return target


def _outside(tmp_path, name="outside"):
    """A pre-existing directory OUTSIDE the import root. Its emptiness after
    every refusal is the whole point."""
    directory = tmp_path / name
    directory.mkdir()
    return directory


# ==========================================================================
# ENG-01 — the derived ledger destination must not escape the import root
# ==========================================================================


def test_a_traversal_run_id_cannot_place_the_ledger_outside_the_target(
        tmp_path):
    """THE reproduction. Every hash in this bundle is correct; the metadata
    is what is hostile. ``runs/../../outside`` resolved to a sibling of the
    import root and the ledger landed there."""
    target = _target_with_runs(tmp_path)
    outside = _outside(tmp_path)
    bundle = str(tmp_path / "traversal.tar.gz")
    _ledger_bundle(bundle, run_id="../../outside")

    with pytest.raises(bundles.BundleError, match="runID"):
        bundles.import_bundle(bundle, target_root=str(target))

    assert list(outside.iterdir()) == [], \
        "the ledger landed outside the import root"
    assert list((target / "runs").iterdir()) == []
    _no_leftovers(target)


def test_an_absolute_run_id_is_refused(tmp_path):
    """``os.path.join`` lets an ABSOLUTE component discard everything before
    it, so an absolute ``runID`` does not even need a ``..`` to leave."""
    target = _target_with_runs(tmp_path)
    outside = _outside(tmp_path, "absolute")
    bundle = str(tmp_path / "absolute.tar.gz")
    _ledger_bundle(bundle, run_id=str(outside))

    with pytest.raises(bundles.BundleError, match="runID"):
        bundles.import_bundle(bundle, target_root=str(target))

    assert list(outside.iterdir()) == []
    _no_leftovers(target)


def test_a_symlinked_run_directory_cannot_carry_the_ledger_outside(tmp_path):
    """The ``runID`` here is a perfectly ordinary single segment — it is the
    WORKSPACE that points elsewhere. Members are canonicalized with
    ``os.path.realpath`` precisely so a symlink cannot be a tunnel; the
    derived destination must be canonicalized the same way."""
    target = _target_with_runs(tmp_path)
    outside = _outside(tmp_path)
    os.symlink(str(outside), str(target / "runs" / "linked"))
    bundle = str(tmp_path / "symlinked.tar.gz")
    _ledger_bundle(bundle, run_id="linked")

    with pytest.raises(bundles.BundleError, match="outside the target root"):
        bundles.import_bundle(bundle, target_root=str(target))

    assert list(outside.iterdir()) == [], \
        "the ledger followed the symlink out of the import root"
    _no_leftovers(target)


def test_valid_hashes_do_not_license_an_escaping_destination(tmp_path):
    """Both pins verify — the out-of-band outer digest and the portable
    payload's own — and neither says anything about WHERE the payload is
    entitled to land. A correctly pinned bundle is not thereby a safe one."""
    target = _target_with_runs(tmp_path)
    outside = _outside(tmp_path)
    bundle = str(tmp_path / "pinned-traversal.tar.gz")
    _ledger_bundle(bundle, run_id="../../outside")
    outer = bundles.sha256_file(bundle)

    with pytest.raises(bundles.BundleError, match="runID"):
        bundles.import_bundle(bundle, target_root=str(target),
                              expected_sha256=outer)

    assert list(outside.iterdir()) == []
    _no_leftovers(target)


def test_an_escaping_ledger_leaves_both_trees_exactly_as_it_found_them(
        tmp_path):
    """The transaction's promise, extended to the derived destination: the
    refusal is decided in PREFLIGHT, so the bundle's perfectly good ordinary
    members never land either, and nothing waits in staging."""
    target = _target_with_runs(tmp_path)
    outside = _outside(tmp_path)
    bundle = str(tmp_path / "mixed.tar.gz")
    _ledger_bundle(bundle, run_id="../../outside",
                   members=[("runs/keep/first.jsonl", b'{"n":1}\n'),
                            ("runs/keep/second.jsonl", b'{"n":2}\n')])

    with pytest.raises(bundles.BundleError, match="runID"):
        bundles.import_bundle(bundle, target_root=str(target))

    assert not (target / "runs" / "keep").exists(), \
        "a good member landed anyway — the refusal was not a preflight one"
    assert list(outside.iterdir()) == []
    _no_leftovers(target)


def test_a_portable_ledger_with_no_run_id_is_refused(tmp_path):
    """A bundle that carries a verified ledger and no ``runID`` to place it
    used to import "successfully" while silently dropping the ledger. There
    is no safe destination to derive, so there is no import."""
    target = _target_with_runs(tmp_path)
    bundle = str(tmp_path / "no-run-id.tar.gz")
    _ledger_bundle(bundle, run_id="")

    with pytest.raises(bundles.BundleError, match="runID"):
        bundles.import_bundle(bundle, target_root=str(target))
    _no_leftovers(target)


@pytest.mark.parametrize("run_id", [".", "..", "a/b", "a\\b", 17, None])
def test_every_unsafe_run_id_shape_is_refused(tmp_path, run_id):
    """One segment, and one that names something. Separators, the two
    relative names, and anything that is not a string at all."""
    target = _target_with_runs(tmp_path)
    bundle = str(tmp_path / "shape.tar.gz")
    _ledger_bundle(bundle, run_id=run_id)

    with pytest.raises(bundles.BundleError, match="runID"):
        bundles.import_bundle(bundle, target_root=str(target))
    _no_leftovers(target)


def test_an_ordinary_pipeline_bundle_still_lands_its_ledger(tmp_path):
    """The guard on the repair: the honest bundle — a single-segment run ID
    naming a directory the import itself creates — is untouched."""
    target = tmp_path / "target"
    target.mkdir()
    bundle = str(tmp_path / "pipeline.tar.gz")
    ledger = b'{"kind":"pipelinePortable"}\n'
    _ledger_bundle(bundle, run_id="pipeline-2026", ledger=ledger,
                   members=[("runs/pipeline-2026/pipeline.json",
                             b'{"kind":"pipeline"}\n')])

    result = bundles.import_bundle(bundle, target_root=str(target))
    assert f"runs/pipeline-2026/{LEDGER}" in result["extracted"]
    assert (target / "runs" / "pipeline-2026" / LEDGER).read_bytes() == ledger
    _no_leftovers(target)


# ==========================================================================
# ENG-04 — a member's pinned digest must be a digest, and must be compared
# ==========================================================================


def _digest_bundle(path, name, payload, digest, *, extra=None):
    """A bundle whose single member's DECLARED digest is whatever the caller
    hands over — including nothing at all (``_ABSENT``)."""
    entry = {"path": name, "bytes": len(payload)}
    if digest is not _ABSENT:
        entry["sha256"] = digest
    meta = {
        "schemaVersion": bundles.BUNDLE_SCHEMA,
        "kind": "runBundle",
        "createdAt": time.time(),
        "experiment": "digest-shape",
        "entries": [entry],
    }
    meta.update(extra or {})
    with tarfile.open(path, "w:gz") as tar:
        for member_name, blob in (("steerlab-bundle.json",
                                   json.dumps(meta).encode("utf-8")),
                                  (name, payload)):
            info = tarfile.TarInfo(member_name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return meta


MEMBER = "runs/ok/data.jsonl"
PAYLOAD = b'{"n":1}\n'


def test_a_valid_digest_still_imports_and_is_compared(tmp_path):
    """The baseline both directions: the matching digest lands, and the
    mismatching one — well formed, simply wrong — still refuses."""
    target = tmp_path / "target"
    good = str(tmp_path / "good.tar.gz")
    _digest_bundle(good, MEMBER, PAYLOAD, hashlib.sha256(PAYLOAD).hexdigest())
    assert bundles.import_bundle(good, target_root=str(target))["extracted"] \
        == [MEMBER]
    assert (target / "runs" / "ok" / "data.jsonl").read_bytes() == PAYLOAD

    other = tmp_path / "t2"
    bad = str(tmp_path / "wrong.tar.gz")
    _digest_bundle(bad, MEMBER, PAYLOAD, "0" * 64)
    with pytest.raises(bundles.BundleError, match="hash mismatch"):
        bundles.import_bundle(bad, target_root=str(other))
    assert not (other / "runs").exists()


def test_an_uppercase_digest_is_accepted_and_compared_case_insensitively(
        tmp_path):
    """The case policy, made explicit: a digest is hex, hex has no case, and
    a bundle whose producer stamped uppercase is verified rather than
    refused for a difference that carries no information."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "upper.tar.gz")
    _digest_bundle(bundle, MEMBER, PAYLOAD,
                   hashlib.sha256(PAYLOAD).hexdigest().upper())
    assert bundles.import_bundle(bundle,
                                 target_root=str(target))["extracted"] == [MEMBER]
    assert (target / "runs" / "ok" / "data.jsonl").read_bytes() == PAYLOAD

    # …and the comparison is still a comparison: uppercase does not become a
    # licence to carry different bytes.
    other = tmp_path / "t2"
    tampered = str(tmp_path / "upper-wrong.tar.gz")
    _digest_bundle(tampered, MEMBER, b"different\n",
                   hashlib.sha256(PAYLOAD).hexdigest().upper())
    with pytest.raises(bundles.BundleError, match="hash mismatch"):
        bundles.import_bundle(tampered, target_root=str(other))
    assert not (other / "runs").exists()


def test_an_empty_digest_is_refused_and_nothing_lands(tmp_path):
    """THE ENG-04 reproduction. ``""`` is not ``None``, so preflight passed
    it; ``if planned.expected and …`` is False for ``""``, so the comparison
    never ran. The member's bytes were arbitrary and landed unverified."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "empty-digest.tar.gz")
    _digest_bundle(bundle, MEMBER, b"whatever this is, nothing checked it\n",
                   "")

    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target))

    assert not (target / "runs").exists(), \
        "an unverified member reached the workspace"
    _no_leftovers(target)


@pytest.mark.parametrize("digest", [
    "   ",                       # blank, but truthy
    "0" * 63,                    # one short
    "0" * 65,                    # one long
    "z" * 64,                    # right length, not hex
    "0" * 32,                    # an MD5-shaped pin
    12345,                       # not a string at all
    None,                        # present in the entry, explicitly null
    ["0" * 64],                  # a container that would compare unequal
])
def test_every_malformed_digest_is_refused_before_anything_lands(tmp_path,
                                                                 digest):
    """Shape is checked in PREFLIGHT, so every one of these is a refusal
    with the workspace exactly as it was — not a refusal that happens to
    fall out of a comparison downstream, and never a silent pass."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "malformed.tar.gz")
    _digest_bundle(bundle, MEMBER, PAYLOAD, digest)

    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target))

    assert not (target / "runs").exists()
    _no_leftovers(target)


def test_a_member_absent_from_the_entry_list_is_still_refused(tmp_path):
    """Unchanged, and distinct from a malformed pin: a member the entry list
    does not NAME is unverifiable for a different reason, and says so."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "unlisted.tar.gz")
    _digest_bundle(bundle, MEMBER, PAYLOAD, _ABSENT)
    # The entry exists but carries no digest key at all — an entry list that
    # names the member and pins nothing.
    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target))
    assert not (target / "runs").exists()

    # …and a member with no entry at all keeps its own refusal.
    orphan = str(tmp_path / "orphan.tar.gz")
    _digest_bundle(orphan, MEMBER, PAYLOAD,
                   hashlib.sha256(PAYLOAD).hexdigest())
    with tarfile.open(orphan, "r:gz") as source:
        meta = json.loads(source.extractfile("steerlab-bundle.json").read())
    meta["entries"] = []
    with tarfile.open(orphan, "w:gz") as tar:
        for name, blob in (("steerlab-bundle.json",
                            json.dumps(meta).encode("utf-8")),
                           (MEMBER, PAYLOAD)):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    with pytest.raises(bundles.BundleError, match="not listed"):
        bundles.import_bundle(orphan, target_root=str(tmp_path / "t2"))


def test_a_correctly_outer_pinned_archive_with_a_bad_member_digest_refuses(
        tmp_path):
    """The outer pin proves the archive is the one the job record named. It
    proves nothing about whether that archive's own entry list verifies its
    members — so it must not be a way past this check."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "outer-pinned.tar.gz")
    _digest_bundle(bundle, MEMBER, b"unverified bytes\n", "")
    outer = bundles.sha256_file(bundle)

    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target),
                              expected_sha256=outer)
    assert not (target / "runs").exists()
    _no_leftovers(target)


# ==========================================================================
# ENG-04, the audit: the same missing-vs-empty question, asked of every other
# hash field in the module
# ==========================================================================


def test_an_uppercase_portable_pin_verifies_rather_than_refusing(tmp_path):
    """The portable payload's pin gets the same case policy as a member's."""
    target = tmp_path / "target"
    ledger = b'{"kind":"pipelinePortable"}\n'
    bundle = str(tmp_path / "upper-pin.tar.gz")
    _ledger_bundle(bundle, run_id="pipeline-2026", ledger=ledger,
                   pin=hashlib.sha256(ledger).hexdigest().upper(),
                   members=[("runs/pipeline-2026/pipeline.json",
                             b'{"kind":"pipeline"}\n')])

    result = bundles.import_bundle(bundle, target_root=str(target))
    assert f"runs/pipeline-2026/{LEDGER}" in result["extracted"]
    _no_leftovers(target)


@pytest.mark.parametrize("pin", ["0" * 63, "z" * 64, 12345])
def test_a_malformed_portable_pin_is_refused_as_malformed(tmp_path, pin):
    """A pin that cannot be a digest is a broken bundle, not a tampered one,
    and the refusal should say which."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "bad-pin.tar.gz")
    _ledger_bundle(bundle, run_id="pipeline-2026", pin=pin)

    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target))
    _no_leftovers(target)


def test_a_blank_portable_pin_with_no_ledger_member_is_still_incomplete(
        tmp_path):
    """The closure check asked ``if meta.get(…)``, so a bundle could declare
    an EMPTY portable pin, ship no ledger, and import clean — the pin
    stamped, the evidence gone. Presence of the field is the claim; the
    member must be there to answer it."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "no-ledger.tar.gz")
    _bundle_with_members(bundle, [("runs/ok/data.jsonl", PAYLOAD)],
                         extra={"runID": "ok", "pipelinePortableSha256": ""})

    with pytest.raises(bundles.BundleError, match="portable pipeline ledger"):
        bundles.import_bundle(bundle, target_root=str(target))
    assert not (target / "runs").exists()
    _no_leftovers(target)


def test_a_malformed_outer_pin_is_refused_as_malformed_not_as_a_mismatch(
        tmp_path):
    """The out-of-band pin, same distinction. A caller who passes something
    that is not a digest has a broken call, not a substituted archive, and
    reporting it as a hash mismatch sends them hunting the wrong thing."""
    target = tmp_path / "target"
    bundle = str(tmp_path / "ordinary.tar.gz")
    _digest_bundle(bundle, MEMBER, PAYLOAD, hashlib.sha256(PAYLOAD).hexdigest())

    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target),
                              expected_sha256="not-a-digest")
    assert not (target / "runs").exists()

    # A non-string pin used to be an AttributeError from `.strip()`, which is
    # a crash rather than a refusal.
    with pytest.raises(bundles.BundleError, match="SHA-256 digest"):
        bundles.import_bundle(bundle, target_root=str(target),
                              expected_sha256=b"\x00" * 64)
    assert not (target / "runs").exists()
    _no_leftovers(target)
