"""POLE MIRRORING for steering-vector artifacts — the other end of a
contrastive direction, minted as a properly provenanced artifact of its own.

A CAA direction points from its negative file's pole toward its positive
file's pole (``mean(pos) − mean(neg)``). A researcher who wants to inject the
OTHER pole has, today, nothing but a negative α — which every downstream
surface (sweep grids, alpha ladders, the norm-unit dose policy, the
playground) treats as "less of the concept" rather than "the opposite
concept", and which no artifact records. Mirroring writes the negation down:
the tensors multiplied by −1 at every layer, under a NEW concept name, with a
derivation stamp that names the bytes it came from.

Three decisions carry the honesty of the result.

**The negation is a SIGN-BIT FLIP, not arithmetic.** The ``.safetensors``
bytes are copied and each float's IEEE-754 sign bit is XORed, so nothing is
decoded and re-encoded through a lossy path, ``-0.0`` round-trips as ``-0.0``,
and the transform is an INVOLUTION: mirroring a mirror returns the parent's
tensor bytes byte-for-byte (``test_pole_mirror.py``). Only the ``layer_<i>``
tensors are flipped — ``neutral_mean_layer_<i>`` is the residual stream's own
mean at that layer, an absolute activation statistic that has nothing to do
with which pole the concept vector points at, and negating it would corrupt
the ablation mean-centring that reads it.

**A new concept name is REQUIRED.** Two artifacts under one concept name
pointing in opposite directions is a hazard nothing downstream can detect:
every selector, pin, and promotion matcher addresses a direction by concept.

**``stimulusSetHash`` is PRESERVED and qualified.** The mirrored concept's
stimuli are the same two files as the source's with the positive/negative
roles swapped. Minting a fresh hash would claim different bytes were read;
carrying the source's hash silently would claim the same recipe. So the hash
travels and ``polesSwappedFromSource: True`` says what changed about its
meaning.

Swift twin: ``Sources/ExperimentKit/PoleMirror.swift``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

#: ``layer_<i>`` and nothing else. ``neutral_mean_layer_<i>`` deliberately
#: fails this match (see the module docstring).
_LAYER_TENSOR = re.compile(r"^layer_(\d+)$")

#: Element widths in bytes for the FLOAT dtypes safetensors names. Integer
#: dtypes are absent on purpose: two's-complement negation is not a sign-bit
#: flip, so an integer tensor is refused rather than silently mangled.
_FLOAT_ELEMENT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}

#: The provenance key a mirrored sidecar carries, and the flag that qualifies
#: its inherited ``stimulusSetHash``. Pinned cross-engine contract.
NEGATED_FROM_KEY = "negatedFrom"
POLES_SWAPPED_KEY = "polesSwappedFromSource"

#: What the CLI prints after a successful mint, and the reason this module
#: writes NOTHING into ``prompts/concepts/``: the mirrored pole's held-out
#: evidence is a file only the researcher can author, and an engine that
#: invented it would be manufacturing the very evidence the gate exists to
#: demand. Swift twin: ``PoleMirror.validationAuthoringNote``.
def validation_authoring_note(concept: str) -> str:
    return (f"to validate the mirrored pole, author "
            f"prompts/concepts/{concept}/validation.jsonl — the source "
            "concept's rows with every expresses label inverted are the "
            "natural starting point")


class PoleMirrorError(ValueError):
    """A typed mirroring refusal: which gate declined, why, and the repair.

    ``kind`` is the stable machine code the CLI puts in ``error.code``; the
    Swift twin's ``PoleMirror.MirrorError.Kind`` carries the same strings.
    """

    def __init__(self, kind: str, reason: str, repair_action: str):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.repair_action = repair_action


@dataclass
class MirrorResult:
    run_directory: str
    artifact_id: str                # "<run_directory>/<name>" (catalog id)
    vectors_path: str
    sidecar_path: str
    concept: str
    source_artifact: str
    source_concept: str
    source_vectors_hash: str
    source_sidecar_hash: str
    layer_count: int


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _iso8601(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def negate_layer_tensors(payload: bytes) -> bytes:
    """Return ``payload`` with every ``layer_<i>`` float's sign bit flipped.

    Bit-exact and involutive by construction: no float is decoded, so no
    rounding, no NaN-payload rewrite, and ``-0.0`` survives as ``-0.0``.
    Every other byte — the 8-byte header length, the JSON header, and any
    non-``layer_`` tensor (notably ``neutral_mean_layer_<i>``) — is copied
    unchanged. Swift twin: ``PoleMirror.negatedTensorBytes``.
    """
    if len(payload) < 8:
        raise PoleMirrorError(
            "unreadableArtifact",
            "safetensors payload is shorter than its 8-byte header length",
            "re-extract the source artifact; its .safetensors is truncated")
    (header_length,) = struct.unpack_from("<Q", payload, 0)
    header_start = 8
    data_start = header_start + header_length
    if header_length > len(payload) - header_start:
        raise PoleMirrorError(
            "unreadableArtifact",
            f"safetensors header claims {header_length} bytes but only "
            f"{len(payload) - header_start} follow",
            "re-extract the source artifact; its .safetensors is truncated")
    try:
        header = json.loads(payload[header_start:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoleMirrorError(
            "unreadableArtifact",
            f"safetensors header is not readable JSON: {exc}",
            "re-extract the source artifact; its .safetensors is corrupt"
        ) from exc

    out = bytearray(payload)
    flipped = 0
    for key, entry in sorted(header.items()):
        if not _LAYER_TENSOR.match(key) or not isinstance(entry, dict):
            continue
        dtype = entry.get("dtype")
        width = _FLOAT_ELEMENT_BYTES.get(dtype)
        if width is None:
            raise PoleMirrorError(
                "unreadableArtifact",
                f"tensor {key!r} has dtype {dtype!r} — mirroring flips IEEE "
                "sign bits and is defined only for float tensors "
                f"({', '.join(sorted(_FLOAT_ELEMENT_BYTES))})",
                "re-extract the source artifact as float32")
        start, stop = entry["data_offsets"]
        if (stop - start) % width or stop > len(payload) - data_start:
            raise PoleMirrorError(
                "unreadableArtifact",
                f"tensor {key!r} data_offsets [{start}, {stop}] do not "
                f"describe whole {dtype} elements inside the payload",
                "re-extract the source artifact; its .safetensors is corrupt")
        # The sign bit is the MSB of the LAST byte of each little-endian
        # element, for every IEEE width safetensors carries.
        for offset in range(data_start + start + width - 1,
                            data_start + stop, width):
            out[offset] ^= 0x80
        flipped += 1
    if not flipped:
        raise PoleMirrorError(
            "unreadableArtifact",
            "safetensors payload carries no layer_<i> tensors — it is not a "
            "steering-vector artifact",
            "pass the base path of a vector artifact written by "
            "`steerlab-server experiment extract <name>`")
    return bytes(out)


def mirrored_sidecar(original: dict, *, concept: str, source_artifact: str,
                     source_vectors_hash: str, source_sidecar_hash: str,
                     date: datetime | None = None) -> dict:
    """The mirrored artifact's sidecar: the source's, field for field, with
    the concept renamed and the derivation stamped.

    Everything sign-invariant is preserved verbatim, and most of the sidecar
    IS sign-invariant:

    * ``normsPerLayer`` / ``residualNormPerLayer`` and the whole
      ``residualNorm*`` denominator family — an L2 norm does not change when
      the vector it measures is negated, so a mirrored artifact's α in norm
      units means exactly the dose the source's did;
    * ``readingPosition`` / ``readingPositionResolution`` /
      ``extractionRendering`` — where and how the activations were read;
    * ``coversModelDepth``, ``layerCount``, ``hiddenSize`` — the shape is
      untouched;
    * ``modelID`` / ``revision`` / ``substrate`` — the same bytes on the same
      model;
    * ``extractionMethod`` / ``recipeMethod`` / ``signConvention`` and the
      reader-, SAE- and OptVec-provenance blocks — these describe how the
      SOURCE direction was produced, which is still the true answer to "where
      did these numbers come from"; ``negatedFrom`` names the artifact they
      describe.

    Exactly one field is DROPPED: ``recipeIdentityHash``. That hash is an
    identity claim about THESE bytes ("running this recipe produces this
    artifact"), its canonical form includes the concept name, and promotion
    matches candidates on it. Carrying the source's hash onto a renamed,
    negated artifact would let the matcher treat the mirror as the source
    recipe's output — a wrong answer rather than a missing one.
    """
    sidecar = dict(original)
    sidecar["concept"] = concept
    sidecar[NEGATED_FROM_KEY] = {
        "path": source_artifact,
        "sha256TensorHash": source_vectors_hash,
        "sha256SidecarHash": source_sidecar_hash,
        "concept": str(original.get("concept", "")),
        "date": _iso8601(date or datetime.now(timezone.utc)),
    }
    # The inherited stimulusSetHash, qualified rather than reminted: same two
    # files, roles swapped (module docstring).
    sidecar[POLES_SWAPPED_KEY] = True
    sidecar.pop("recipeIdentityHash", None)
    return sidecar


def mirror_poles(vector_dir: str, name: str, *, concept: str,
                 run_directory: str, output_name: str | None = None,
                 date: datetime | None = None) -> MirrorResult:
    """Mint the mirrored pole of ``<vector_dir>/<name>`` into
    ``run_directory``. The source artifact is never modified (runs are
    immutable). Returns the new artifact's identity."""
    base = os.path.join(vector_dir, name)
    vectors_source = base + ".safetensors"
    sidecar_source = base + ".json"
    if not name or not (os.path.isfile(vectors_source)
                        and os.path.isfile(sidecar_source)):
        raise PoleMirrorError(
            "sourceNotFound", source_not_found_reason(base),
            source_not_found_repair("steerlab-server"))

    concept = (concept or "").strip()
    if not concept:
        raise PoleMirrorError(
            "conceptRequired", concept_required_reason(""),
            concept_required_repair("steerlab-server", base))

    with open(vectors_source, "rb") as handle:
        vector_bytes = handle.read()
    with open(sidecar_source, "rb") as handle:
        sidecar_bytes = handle.read()
    try:
        original = json.loads(sidecar_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoleMirrorError(
            "unreadableArtifact", f"unreadable sidecar {sidecar_source}: {exc}",
            "pass the base path of a vector artifact — <runDir>/<name> with "
            "no extension") from exc
    if not isinstance(original, dict) or "layerCount" not in original:
        raise PoleMirrorError(
            "unreadableArtifact",
            f"{base!r} is not a steering-vector artifact",
            "pass the base path of a vector artifact — <runDir>/<name> with "
            "no extension")

    source_concept = str(original.get("concept", ""))
    if source_concept == concept:
        raise PoleMirrorError(
            "conceptRequired", concept_required_reason(source_concept),
            concept_required_repair("steerlab-server", base))

    # Double mirror: this artifact is ALREADY the negation of the concept
    # being asked for, so the thing being requested exists and has a name.
    existing = original.get(NEGATED_FROM_KEY)
    if isinstance(existing, dict) and str(existing.get("concept", "")) == concept:
        raise PoleMirrorError(
            "doubleMirror",
            double_mirror_reason(base, concept, str(existing.get("path", ""))),
            double_mirror_repair(str(existing.get("path", ""))))

    out_name = output_name or concept
    if not out_name or "/" in out_name or os.sep in out_name \
            or out_name in (".", ".."):
        raise PoleMirrorError(
            "conceptRequired",
            f"--output-name {out_name!r} must be a plain file-name component",
            "pass --output-name <name> with no path separators, or omit it "
            "and the mirrored concept name is used")

    os.makedirs(run_directory, exist_ok=True)
    vectors_path = os.path.join(run_directory, f"{out_name}.safetensors")
    sidecar_path = os.path.join(run_directory, f"{out_name}.json")
    # No-replace, the house rule for artifacts: a mirror that overwrote one
    # would destroy provenance nothing else records.
    for path in (vectors_path, sidecar_path):
        if os.path.exists(path):
            raise PoleMirrorError(
                "destinationOccupied", destination_occupied_reason(path),
                destination_occupied_repair())

    mirrored_bytes = negate_layer_tensors(vector_bytes)
    sidecar = mirrored_sidecar(
        original, concept=concept, source_artifact=base,
        source_vectors_hash=_sha256_hex(vector_bytes),
        source_sidecar_hash=_sha256_hex(sidecar_bytes), date=date)

    with open(vectors_path, "wb") as handle:
        handle.write(mirrored_bytes)
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)

    return MirrorResult(
        run_directory=run_directory,
        artifact_id=os.path.join(run_directory, out_name),
        vectors_path=vectors_path, sidecar_path=sidecar_path,
        concept=concept, source_artifact=base, source_concept=source_concept,
        source_vectors_hash=sidecar[NEGATED_FROM_KEY]["sha256TensorHash"],
        source_sidecar_hash=sidecar[NEGATED_FROM_KEY]["sha256SidecarHash"],
        layer_count=int(original["layerCount"]))


# --- refusal texts (cross-engine literals; Swift twin: PoleMirror) -------------
#
# One function per refusal, on both engines, because these sentences are the
# product: an agent reading two engines' logs must read one sentence, and a
# sentence duplicated at its two throw sites drifts.

ARTIFACT_SHAPE = ("a vector artifact is <runDir>/<name>.safetensors PLUS its "
                  "<runDir>/<name>.json sidecar")


def source_not_found_reason(base: str) -> str:
    return (f"no vector artifact at '{base}' — {ARTIFACT_SHAPE}, and the "
            "reference is the base path they share, with no extension")


def source_not_found_repair(program: str) -> str:
    return (f"pass <runDir>/<name> as `{program} vectors mirror-poles` prints "
            "it, or list the run directories under runs/")


def concept_required_reason(source_concept: str) -> str:
    same = (f" (you passed '{source_concept}', which is the source's own name)"
            if source_concept else "")
    return (f"--concept <newName> is required and must differ from the "
            f"source's concept{same} — the mirrored pole is a DIFFERENT "
            "concept. A contrastive direction points from its negative file's "
            "pole toward its positive file's pole, so its negation points at "
            "the opposite pole; writing that under the source's name would "
            "leave two artifacts with one concept name pointing opposite "
            "ways, and every selector, pin, and promotion matcher addresses a "
            "direction by concept")


def concept_required_repair(program: str, base: str) -> str:
    return (f"{program} vectors mirror-poles {base} --concept <a name for the "
            "opposite pole>")


def destination_occupied_reason(path: str) -> str:
    return (f"'{path}' already exists — mirroring never replaces an artifact "
            "(run directories are immutable)")


def destination_occupied_repair() -> str:
    return ("write the mirror into a fresh run directory, or pass "
            "--output-name <name> to give it a name that is free there")


def double_mirror_reason(base: str, concept: str, parent: str) -> str:
    return (f"'{base}' is ITSELF the mirror of concept '{concept}' — its "
            f"negatedFrom names '{parent}'. Mirroring it back would mint a "
            "third copy of a direction that already exists on disk")


def double_mirror_repair(parent: str) -> str:
    return f"use the original artifact at '{parent}' instead of mirroring this one"
