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
import math
import os
import re
import shutil
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .vector_math import ExtractionMethod

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

#: The repair every unreadable-payload refusal names, on both engines. One
#: literal, because a repair sentence duplicated at its throw sites drifts —
#: which is exactly what had happened here (this engine said "truncated" where
#: the Swift twin said "corrupt", for the same two refusals).
#: Swift twin: ``PoleMirror.truncatedRepair``.
_TRUNCATED_REPAIR = "re-extract the source artifact; its .safetensors is corrupt"


def mirrorable_methods() -> list[ExtractionMethod]:
    """The extraction methods a mirrored pole is DEFINED for, decided from the
    methods' own properties rather than from a list that drifts as methods are
    added: PAIRED (two authored stimulus files, so swapping their roles is what
    the negation means) AND source-concept-bearing (so the swapped files, and a
    validation.jsonl over them, exist somewhere a study could pin). Today that
    is the CAA family, ``meanDifference`` and ``pairedDifferencePCA``.

    Why the OTHER source-concept-bearing methods are excluded — the question
    this restriction had to answer (external review round 8, finding 2):

    * ``designatedReference`` is source-concept-bearing but UNPAIRED. Its
      direction is mean(concept stories) − mean(a designated REFERENCE
      corpus's stories), so its negation is "the reference corpus minus the
      concept" — a different comparison, not the concept's opposite pole. The
      reference is a baseline the study designated, not a pole a researcher
      authored as the concept's other end, and nothing in the sidecar's
      ``designatedReference {name, hash}`` schema can even express a swap: the
      ``stimulusSetHash`` is the concept's own stories hash, and a mirrored
      artifact would have to claim the reference corpus is now a concept with
      held-out scenarios of its own. Not obviously yes, so: excluded, and the
      refusal says why.
    * ``emotionGrandMean`` negates to "the population mean minus the concept",
      which is generic negation with no second pole anywhere.
    * ``optvec``, ``gemmaScopeSAE`` and ``repeReaderLAT`` have no source
      concept at all — no stimulus files to swap, and a ``validation.jsonl``
      under the mirrored name is a file ``attach_artifact`` pins EXPLICITLY
      NULL, so the success message used to promise a workflow attach forbids.

    Swift twin: ``PoleMirror.mirrorableMethods``.
    """
    return [method for method in ExtractionMethod
            if method.is_paired and method.has_source_concept]


def mirrorable_method_list() -> str:
    """Those methods' values, sorted — the vocabulary the refusal names."""
    return ", ".join(sorted(method.value for method in mirrorable_methods()))

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


def _layer_count_reason(base: str, value: float) -> str:
    """A sidecar ``layerCount`` that is a number but not a layer count. The
    value is NAMED — a refusal about a field has to say what it read. Swift
    twin: ``PoleMirror.layerCountReason``, byte-for-byte, including how the
    offending number is spelled."""
    if isinstance(value, float) and math.isnan(value):
        spelled = "NaN"
    elif isinstance(value, float) and math.isinf(value):
        spelled = "-Infinity" if value < 0 else "Infinity"
    elif float(value) == int(float(value)) and abs(float(value)) < 1e15:
        spelled = str(int(value))
    else:
        spelled = str(float(value))
    return (f"'{base}' records layerCount {spelled} — a steering-vector "
            "artifact's layer count is a whole number of layers, 1 or more")


def _remove_quietly(path: str) -> None:
    """Delete a path we own, swallowing "it was never there". Used only on
    failure paths, where a second exception would replace the real one."""
    try:
        os.remove(path)
    except OSError:
        pass


def _commit_no_replace(staged: str, dest: str) -> None:
    """Promote a staged file onto ``dest``, incapable of overwriting it.

    ``os.link`` + ``os.remove``, not ``os.replace``. Replace clobbers, and the
    ``destinationOccupied`` preflight above runs BEFORE the tensors are
    negated and both temporaries are written — so between the check and the
    promotion there is a window in which a second mirror of the same concept,
    or the same call run twice, can create the file. ``link`` raises
    ``FileExistsError`` in exactly that case, which turns the race into the
    refusal the preflight already gives instead of a silent overwrite of
    somebody's artifact (review round 9, finding 7; the same class as the
    runner's round-5 commit fix). It also commits bytes already on the disk
    rather than copying them a second time.

    The O_EXCL reservation is the fallback for a filesystem without hardlinks,
    where it is slower and just as unable to overwrite anything — with the
    same stated residual as its twins: on that path the destination NAME is
    visible, empty then partial, for as long as the copy takes. Nothing can
    overwrite it, and a failed copy removes it again.

    The TWIN of ``experiment.bundles._commit_no_replace`` and
    ``client.runner._commit_no_replace`` — same primitive, same fallback, same
    reasoning — mirrored a third time rather than shared, for the reason those
    two are mirrored from each other: this module reaches nothing but the
    standard library, and a change to any of them belongs in all of them.

    Raises ``FileExistsError`` when ``dest`` exists; leaves ``staged`` in
    place for the caller's cleanup when it does.

    BOTH-OR-NEITHER (review round 10, finding 8): ``dest`` is this call's
    reservation, and a commit that cannot FINISH must not leave it standing.
    Dropping the staging name is the last step, and it can fail on its own
    (a read-only staging directory, an interrupt) — after ``dest`` has
    landed. The caller's cleanup owns the TEMPORARIES, not the destination,
    so the propagating error used to leave a half-final artifact under the
    final name: the exact state ``destinationOccupied`` then refuses to
    repair. The removal is therefore wrapped, and a failure takes ``dest``
    back out before it propagates.
    """
    try:
        os.link(staged, dest)
    except FileExistsError:
        raise
    except OSError:
        handle_fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        landed = False
        try:
            try:
                landing = os.fdopen(handle_fd, "wb")
            except BaseException:
                os.close(handle_fd)
                raise
            with landing, open(staged, "rb") as source:
                shutil.copyfileobj(source, landing)
            landed = True
        finally:
            if not landed:
                # A copy that died half way must not leave a short file
                # wearing the destination's name — the reservation comes
                # back out.
                _remove_quietly(dest)
    try:
        os.remove(staged)
    except BaseException:
        _remove_quietly(dest)
        raise


def _iso8601(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _readable_offsets(offsets: object) -> bool:
    """Whether a header entry's ``data_offsets`` is the shape the Swift twin's
    typed ``[Int]`` decode accepts: a list (or tuple) of plain integers.
    ``bool`` is excluded deliberately — it is an ``int`` subclass in Python and
    is not one in the contract."""
    return (isinstance(offsets, (list, tuple))
            and all(isinstance(value, int) and not isinstance(value, bool)
                    for value in offsets))


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
            _TRUNCATED_REPAIR)
    (header_length,) = struct.unpack_from("<Q", payload, 0)
    header_start = 8
    data_start = header_start + header_length
    if header_length > len(payload) - header_start:
        raise PoleMirrorError(
            "unreadableArtifact",
            f"safetensors header claims {header_length} bytes but only "
            f"{len(payload) - header_start} follow",
            _TRUNCATED_REPAIR)
    try:
        header = json.loads(payload[header_start:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoleMirrorError(
            "unreadableArtifact",
            f"safetensors header is not readable JSON: {exc}",
            _TRUNCATED_REPAIR
        ) from exc

    out = bytearray(payload)
    flipped = 0
    for key, entry in sorted(header.items()):
        if not _LAYER_TENSOR.match(key) or not isinstance(entry, dict):
            continue
        # The header is UNTRUSTED input — a `.safetensors` file is bytes on
        # disk, and this transform writes into the payload at offsets the
        # header names. The Swift twin decodes the entry through a typed
        # `HeaderEntry` (a `String` dtype and an `[Int]` data_offsets) and then
        # guards the pair; this engine has to spell the same checks out, or a
        # malformed header escapes as a raw `KeyError`/`TypeError` instead of
        # the typed refusal, and two of the shapes are worse than untidy:
        # `[8, 0]` flips nothing while counting as a flipped tensor (Python's
        # `-8 % 4` is 0, so the modulo test passes and the empty range is a
        # silent no-op), and a NEGATIVE start addresses backwards out of the
        # payload into the header. Swift twin: `PoleMirror.negatedTensorBytes`.
        dtype = entry.get("dtype")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or not _readable_offsets(offsets):
            raise PoleMirrorError(
                "unreadableArtifact",
                f"tensor {key!r} has no readable dtype/data_offsets",
                _TRUNCATED_REPAIR)
        width = _FLOAT_ELEMENT_BYTES.get(dtype)
        if width is None:
            raise PoleMirrorError(
                "unreadableArtifact",
                f"tensor {key!r} has dtype {dtype!r} — mirroring flips IEEE "
                "sign bits and is defined only for float tensors "
                f"({', '.join(sorted(_FLOAT_ELEMENT_BYTES))})",
                "re-extract the source artifact as float32")
        if len(offsets) != 2:
            raise PoleMirrorError(
                "unreadableArtifact",
                f"tensor {key!r} has no readable dtype/data_offsets",
                _TRUNCATED_REPAIR)
        start, stop = offsets
        if (start < 0 or stop < start or (stop - start) % width
                or stop > len(payload) - data_start):
            raise PoleMirrorError(
                "unreadableArtifact",
                f"tensor {key!r} data_offsets [{start}, {stop}] do not "
                f"describe whole {dtype} elements inside the payload",
                _TRUNCATED_REPAIR)
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
    # `layerCount` is read as a NUMBER here, before a single byte is written —
    # it used to be converted after both files had landed, so a sidecar whose
    # layerCount was a string ("2", say) stranded a complete artifact pair on
    # disk and then raised a bare ValueError past every typed refusal. Swift
    # twin: the same numeric guard in `PoleMirror.mirrorPoles`.
    layer_count_value = original.get("layerCount") \
        if isinstance(original, dict) else None
    if not isinstance(layer_count_value, (int, float)) \
            or isinstance(layer_count_value, bool):
        raise PoleMirrorError(
            "unreadableArtifact",
            f"{base!r} is not a steering-vector artifact",
            "pass the base path of a vector artifact — <runDir>/<name> with "
            "no extension")
    # A NUMBER is not yet a layer count (review round 10, finding 9). `2.5`
    # truncated to 2 and stamped a mirror claiming a depth its source never
    # had; `0` and `-3` stamped an impossible one; `nan`/`inf` reach `int()`
    # and raise a bare ValueError/OverflowError past every typed refusal
    # (the Swift twin TRAPS there, which is why both engines check finiteness
    # and integrality BEFORE converting). No upper bound is invented: no other
    # sidecar reader on either engine bounds this key above.
    if (not math.isfinite(layer_count_value)
            or float(layer_count_value) != int(float(layer_count_value))
            or layer_count_value < 1):
        raise PoleMirrorError(
            "unreadableArtifact",
            _layer_count_reason(base, layer_count_value),
            "pass the base path of a vector artifact — <runDir>/<name> with "
            "no extension")
    layer_count = int(layer_count_value)

    # Which methods HAVE an opposite pole (see :func:`mirrorable_methods`).
    # This gate is the reason the success message's validation-authoring note
    # is now always honest: it can only be printed for a method whose mirrored
    # concept can pin a validation.jsonl at attach.
    recorded_method = str(original.get("extractionMethod") or "").strip()
    try:
        source_method: ExtractionMethod | None = ExtractionMethod(recorded_method)
    except ValueError:
        source_method = None
    if source_method is None or not source_method.is_paired \
            or not source_method.has_source_concept:
        raise PoleMirrorError(
            "unmirrorableMethod",
            unmirrorable_method_reason(
                base, recorded_method,
                source_method.label if source_method else None),
            UNMIRRORABLE_METHOD_REPAIR)

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
    sidecar_text = json.dumps(sidecar, sort_keys=True, indent=2)

    # An artifact is a PAIR, so it is written as one. Both files land under
    # temporary names in the destination directory and are promoted only once
    # both are on disk: a failure between the two writes — a full disk, a
    # permission change, an interrupt — used to strand a tensor with no
    # sidecar, which the catalog reads as an unreadable artifact and which the
    # `destinationOccupied` rule then refuses to replace. The cleanup removes
    # exactly the two temporary names, on EVERY failure path (bare `except`,
    # re-raised): a cleanup that only ran for typed refusals would miss the
    # very failures this exists for. Swift twin: `PoleMirror.mirrorPoles`.
    #
    # The promotion is `_commit_no_replace`, not `os.replace` (review round 9,
    # finding 7). The occupancy check above is a PREFLIGHT: it runs before the
    # tensors are negated and both temporaries are written, and `os.replace`
    # would have silently destroyed anything that arrived in that window — in
    # the name of a rule that had just refused exactly that. A destination
    # that appears now is the same `destinationOccupied` refusal it would have
    # been a moment earlier, and everything THIS call created comes back out.
    token = uuid.uuid4().hex
    vectors_temp = f"{vectors_path}.{token}.partial"
    sidecar_temp = f"{sidecar_path}.{token}.partial"
    try:
        with open(vectors_temp, "wb") as handle:
            handle.write(mirrored_bytes)
        with open(sidecar_temp, "w", encoding="utf-8") as handle:
            handle.write(sidecar_text)
        try:
            _commit_no_replace(vectors_temp, vectors_path)
        except FileExistsError as exc:
            raise PoleMirrorError(
                "destinationOccupied",
                destination_occupied_reason(vectors_path),
                destination_occupied_repair()) from exc
        try:
            _commit_no_replace(sidecar_temp, sidecar_path)
        except FileExistsError as exc:
            # The tensor is already promoted; take it back out so a losing
            # writer leaves nothing of its own behind. Removing a name THIS
            # call created — the sidecar that beat us is untouched.
            _remove_quietly(vectors_path)
            raise PoleMirrorError(
                "destinationOccupied",
                destination_occupied_reason(sidecar_path),
                destination_occupied_repair()) from exc
        except BaseException:
            _remove_quietly(vectors_path)
            raise
    except BaseException:
        _remove_quietly(vectors_temp)
        _remove_quietly(sidecar_temp)
        raise

    return MirrorResult(
        run_directory=run_directory,
        artifact_id=os.path.join(run_directory, out_name),
        vectors_path=vectors_path, sidecar_path=sidecar_path,
        concept=concept, source_artifact=base, source_concept=source_concept,
        source_vectors_hash=sidecar[NEGATED_FROM_KEY]["sha256TensorHash"],
        source_sidecar_hash=sidecar[NEGATED_FROM_KEY]["sha256SidecarHash"],
        layer_count=layer_count)


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


def unmirrorable_method_reason(base: str, method: str,
                               label: str | None) -> str:
    if not method:
        recorded = "records no extractionMethod"
    elif label:
        recorded = f"records extractionMethod '{method}' ({label})"
    else:
        recorded = (f"records extractionMethod '{method}', which this engine "
                    "does not know")
    return (f"'{base}' {recorded} — mirror-poles mints the opposite pole only "
            "for a PAIRED, source-concept-bearing contrast "
            f"({mirrorable_method_list()}), where the two poles ARE two "
            "authored stimulus files and swapping their roles is exactly what "
            "the negation MEANS. Every other direction negates GENERICALLY, "
            "with no method-specific evidence semantics: a grand-mean or "
            "class-vs-reference direction negates to 'the population (or the "
            "designated reference corpus) minus the concept', which is a "
            "different comparison and not the concept's opposite pole, and a "
            "direction with no source concept has no stimulus files to swap at "
            "all — so the validation.jsonl a mirrored pole is told to author "
            "would be a file attach pins as EXPLICITLY ABSENT, and authoring "
            "it later is drift")


UNMIRRORABLE_METHOD_REPAIR = (
    "inject the opposite end of this direction with a NEGATIVE α in a study "
    "condition — the sign flip is available there, needs no new artifact, and "
    "claims no evidence the method cannot supply")
