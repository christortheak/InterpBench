"""Durable J-lens artifact records.

A lens is an imported **measuring instrument**, not a steering vector and not a
model: it is the fitted ``J_l`` map from a source-layer residual into the final
pre-normalization basis. See
``docs/GEMMA3-27B-JLENS-VECTORS-AND-ONLINE-READOUT-PLAN.md`` §4.2.

Two provenance roots are tracked separately and must never be conflated:

* the **upstream** artifact — repository, folder, exact filenames, resolved
  commit, and per-file SHA-256 — which is what a reader can independently
  re-download and verify; and
* the **converted** per-layer form this engine writes and actually reads at
  generation time (plan §4.1), which is a derived cache with its own hash.

The upstream configuration names a model but pins **no fit-time revision**.
That is recorded as unknown and stays unknown: substituting the runtime
revision would turn an absence of evidence into a false claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

ARTIFACT_TYPE = "jlens-imported"

#: Stamped on records this engine writes. Lens artifacts are PyTorch/HF-native;
#: activations do not transfer across substrates, so a lens imported here is
#: never valid for the MLX engine (CLAUDE.md, hard requirement).
SUBSTRATE = "python-hf-transformers"

#: Every readout convention this engine can stamp. The canonical readout is
#: ``softcap(U · RMSNorm(J_l h))`` with the model's final-norm gain ``g``
#: folded into the token row; see plan §2. Verified against the reference on
#: 2026-07-27 (Stage 1a). ``g`` is ``1 + weight`` on offset-parameterized
#: RMSNorms (Gemma 1/2/3, Qwen3.5, Qwen3-Next, …) and ``weight`` on direct
#: ones (Llama, Qwen2/3, OLMo, GPT-OSS, …); which one a model uses is
#: OBSERVED from its norm module (:mod:`norm_convention`), never assumed from
#: its name, and stamped wherever the gain is used.
CANONICAL_READOUT = ("softcap(U.RMSNorm(J_l h)); final-norm gain g folded into "
                     "token row (g=1+w offset RMSNorm, g=w direct RMSNorm; "
                     "observed per model)")
DIRECTION_CONVENTION = "J_l^T (g . u_t)"


class JLensError(Exception):
    """Typed failure for every lens lifecycle operation."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iso(when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SourceRef:
    """The upstream artifact, exactly as fetched.

    ``folder``/``tensorFile``/``configFile`` come verbatim from the curated
    table (plan §3.1) or from the published folder's own contents — the
    ``config.yaml`` that names the model and the one ``*_jacobian_lens.pt``
    beside it — and are never synthesized from the model id: upstream names
    the tensor after the *checkpoint variant*, so ``gemma-3-12b/`` ships
    ``gemma-3-12b-pt_jacobian_lens.pt`` and any naming rule is wrong for it.
    """

    repo: str
    folder: str
    tensorFile: str
    configFile: str
    commit: str | None = None
    tensorSHA256: str | None = None
    configSHA256: str | None = None


@dataclass
class ConvertedRef:
    """The per-layer safetensors form this engine reads at generation time.

    Exists because the reference loader promotes every layer to float32 at
    construction — 3.53 GB of fp16 becoming ~6.6 GiB resident, all layers,
    whether one is armed or sixty-one (plan §4.1). Derived and rebuildable:
    if it is missing, re-import from the hash-pinned upstream.
    """

    path: str
    dtype: str
    sha256: str | None = None
    layerCount: int = 0


@dataclass
class FitProvenance:
    """What the upstream configuration does and does not establish."""

    modelID: str | None = None
    #: None means UNKNOWN, and unknown is the honest value: the published
    #: configs carry no base-model revision. Never fill this from the runtime.
    revision: str | None = None
    revisionKnown: bool = False
    dtype: str | None = None
    corpus: str | None = None
    promptsFitted: int | None = None
    maxSeqLen: int | None = None


@dataclass
class Qualification:
    """One acceptance of these exact bytes against one exact runtime.

    Keyed by revision AND numeric configuration, because geometry cannot see
    dtype: a float16 or quantized model has the same layer count, hidden size,
    vocabulary, and head shape as the bf16 one while presenting different
    numerics to the same Jacobian (plan §3.3).
    """

    qualificationID: str
    modelID: str
    revision: str
    dtype: str
    quantization: str | None = None
    tier: str = "testing"
    passed: bool = False
    #: The lens BYTES this acceptance was measured against — upstream tensor
    #: and the converted per-layer cache. Absent on records written before
    #: 2026-08-16, which are matched leniently (a legacy record cannot prove
    #: which bytes it saw, and refusing them all would silently invalidate
    #: history); present ones must agree, so a re-import under the same lensID
    #: cannot inherit an acceptance measured on different matrices.
    lensSHA256: str | None = None
    convertedSHA256: str | None = None
    #: The source layers actually exercised. A qualification that tested three
    #: mid-stack layers says nothing about a study arming layer 3.
    layers: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    qualifiedAt: str = field(default_factory=_iso)

    @property
    def runtime_key(self) -> str:
        return f"{self.modelID}@{self.revision}/{self.dtype}/{self.quantization or 'none'}"


@dataclass
class JLensRecord:
    """The durable record for one imported lens (plan §4.2)."""

    lensID: str
    source: SourceRef
    fit: FitProvenance
    sourceLayers: list[int]
    dModel: int
    targetLayer: int
    nPrompts: int
    converted: ConvertedRef | None = None
    configHash: str | None = None
    referencePackage: str | None = None
    referenceCommit: str | None = None
    #: The evidence tier this lens's model holds IN THIS PROJECT, and where
    #: that came from: ``"curated"`` when the model has a row in
    #: ``importer.SUPPORTED``, ``"declared"`` when the researcher declared it
    #: at import (``jlens import --tier``) because the model has a published
    #: lens but no curated row. A curated row always wins over a declaration.
    #: ``None`` on records written before 2026-09-05, which resolve through
    #: the curated table alone, exactly as they always did.
    tier: str | None = None
    tierSource: str | None = None
    readoutConvention: str = CANONICAL_READOUT
    directionConvention: str = DIRECTION_CONVENTION
    targetLayerConvention: str = "n_layers-1 (final block output); transport there is the identity"
    artifactType: str = ARTIFACT_TYPE
    substrate: str = SUBSTRATE
    importedAt: str = field(default_factory=_iso)
    qualifications: list[Qualification] = field(default_factory=list)
    schemaVersion: int = 1

    def qualification_for(self, model_id: str, revision: str, dtype: str,
                          quantization: str | None = None, *,
                          layers: list | None = None,
                          qualification_id: str | None = None
                          ) -> Qualification | None:
        """The passing qualification for an exact runtime, or None.

        Absent is never a match: a runtime whose numeric configuration cannot
        be established is refused rather than assumed (plan §3.3).

        Three bindings beyond the runtime key (external review 2026-08-16):

        * **The lens bytes.** A record carrying ``lensSHA256`` must match this
          record's current source hash. Re-importing from a different upstream
          commit keeps the lensID and changes every number, and without this
          the old acceptance would still answer.
        * **The tested layers.** When the caller names the layers it intends
          to arm, a qualification that exercised a different set does not
          cover them.
        * **Newest first.** Entries are appended, so iterating forward
          returned the OLDEST passing record — the one most likely to predate
          whatever changed. The newest acceptance is the one that describes
          the current instrument.
        """
        if not (model_id and revision and dtype):
            return None
        wanted = set(int(l) for l in (layers or []))
        candidates = [q for q in self.qualifications
                      if self._binds(q, model_id, revision, dtype,
                                     quantization, wanted)]
        if qualification_id:
            # An EXPLICIT pin resolves that exact record. Newest-first is the
            # right default for exploratory work, but it broke frozen studies:
            # a study pinning a still-valid q1 refused the moment the runtime
            # was re-qualified as q2, because the resolver answered q2 and the
            # freeze gate compared identities (external review round 2).
            for q in candidates:
                if q.qualificationID == qualification_id:
                    return q
            return None
        # Unpinned: the newest acceptance describes the current instrument.
        return candidates[-1] if candidates else None

    def _binds(self, q: "Qualification", model_id: str, revision: str,
               dtype: str, quantization: str | None, wanted: set) -> bool:
        """Whether one qualification covers this exact request.

        **Fail CLOSED.** Every binding must be present AND match. A record
        written before the bindings existed carries none, and the earlier
        lenient reading ("absent means unconstrained") let such a record
        license any lens bytes and any layer — the fail-open the second review
        found. Those records remain readable history; they cannot license.
        There are none in production: no lens has been qualified against a
        real runtime yet, so strictness costs nothing and buys the invariant.
        """
        if not (q.passed and q.modelID == model_id and q.revision == revision
                and q.dtype == dtype
                and (q.quantization or None) == (quantization or None)):
            return False
        # The lens bytes this acceptance was measured against.
        if not q.lensSHA256 or q.lensSHA256 != self.source.tensorSHA256:
            return False
        converted = self.converted.sha256 if self.converted else None
        if not q.convertedSHA256 or q.convertedSHA256 != converted:
            return False
        # The layers it exercised must COVER the layers being armed. A
        # qualification that tested three mid-stack layers says nothing about
        # a study arming layer 3.
        tested = set(int(l) for l in (q.layers or []))
        if not tested:
            return False
        return not wanted or wanted <= tested

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JLensRecord":
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in d.items() if k in known}
        payload["source"] = SourceRef(**(payload.get("source") or {}))
        payload["fit"] = FitProvenance(**(payload.get("fit") or {}))
        conv = payload.get("converted")
        payload["converted"] = ConvertedRef(**conv) if conv else None
        payload["qualifications"] = [
            Qualification(**q) for q in (payload.get("qualifications") or [])]
        return cls(**payload)


def write_record(record: JLensRecord, path: str) -> str:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def read_record(path: str) -> JLensRecord:
    try:
        with open(path, encoding="utf-8") as handle:
            return JLensRecord.from_dict(json.load(handle))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise JLensError(f"unreadable lens record '{path}': {exc}") from exc
