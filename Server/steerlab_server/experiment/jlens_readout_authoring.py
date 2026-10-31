"""Author a draft study's J-lens run readout (``jlensReadout``).

``experiment set-jlens-readout <name> <lens-id>`` on both clients. The engine
has always read, priced, and enforced this block (``run_readouts``,
``jlens.readout``, ``jlens.trace``), but no verb wrote it, and a manifest is
never hand-edited, so no researcher could arm a readout at all.

**What the researcher declares, and what is derived.** The researcher picks
the lens, the layers, a top-k width and/or a token watchlist, whether the
logit-lens companion rides along, an optional qualification, and optional
budget ceilings. Everything else is a pin, and a pin is never typed on this
client: ``lensSHA256`` and the fitted layers come from the workspace's
imported lens record, ``tokenizerHash`` from the study model's tokenizer files
in the local model cache, the newest passing qualification from the same
record, and ``configHash`` from the engine's own ``readout_config_payload``,
so it is the hash run start recomputes.

**The Mac command line takes the pins as flags.** The Mac engine never
resolves or loads a lens (server-only by rule), and the lens record is the
Python engine's schema, so ``steerlab-cli`` does not read it. This module's
``--dry-run`` reports the pins and the exact ``steerlab-cli`` spelling
(``result.pins``, ``result.macCommand``), and the Swift twin
(``JLensReadoutAuthoring``) builds the same block from them, byte for byte —
the cross-client parity test holds the two together.

**Budget.** The engine prices a readout for the whole study at the token cap
(``maxTokens``) and refuses at run start when it does not fit. The verb prices
ONE response with the same function, reports how many responses the declared
budget covers, and refuses a declaration that cannot cover even one. A
declaration guaranteed to reach a run-start refusal is malformed (64), as an
instrument scope that selects zero rows is.

Light by contract: stdlib plus the engine's torch-free lens schema and readout
pricing, so the client's authoring path still imports no torch.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict
from types import SimpleNamespace

from . import command_vocabulary as vocabulary
from . import experiment_store as store
from . import lifecycle_gates
from ..jlens import lens_store
from ..jlens.readout import Budget, ReadoutConfig, preflight
from ..jlens.schemas import JLensError
from ..jlens.trace import config_hash, readout_config_payload

VERB = "set-jlens-readout"
#: The manifest key this verb owns.
MANIFEST_KEY = "jlensReadout"

#: The ranges a declaration may name. ``topK`` and the watchlist are capped at
#: the engine's own default ceilings (``Budget.maxTopK``, ``maxWatchlist``):
#: the cap is a property of the instrument, not a budget a study raises here.
MAX_TOP_K = Budget().maxTopK
MAX_WATCHLIST = Budget().maxWatchlist
#: A token id fits a signed 32-bit integer on every vocabulary in use.
MAX_TOKEN_ID = 2**31 - 1
#: Budget values are whole numbers up to 10^15. The bound keeps every value
#: exactly representable on both clients, so the configuration hash they
#: compute cannot differ by a rounding.
MAX_BUDGET_VALUE = 10**15

#: The budget ceilings a study may declare, flag → ``Budget`` field, in the
#: order the canonical spelling lists them.
BUDGET_FLAGS: tuple[tuple[str, str], ...] = (
    ("--max-armed-layers", "maxArmedLayers"),
    ("--max-observations", "maxObservations"),
    ("--max-trace-bytes", "maxTraceBytes"),
    ("--max-full-vocab-projections", "maxFullVocabProjections"),
)

#: The ceilings that scale with the number of responses, each with the
#: per-response quantity it bounds, in the order a binding one is named.
PROPORTIONAL_CEILINGS: tuple[tuple[str, str], ...] = (
    ("maxFullVocabProjections", "fullVocabProjections"),
    ("maxTraceBytes", "projectedTraceBytes"),
    ("maxObservations", "observations"),
)

_LAYER_ITEM = re.compile(r"([0-9]{1,4})(?:-([0-9]{1,4}))?")
_TOKEN_ID = re.compile(r"[0-9]{1,10}")
_WHOLE = re.compile(r"[0-9]{1,16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def usage(name: str, program: str = vocabulary.CLIENT_PROGRAM) -> str:
    """The declaration's spelling, for a repair."""
    return (f"{program} experiment {VERB} {name} <lens-id> --layers <l,…> "
            "[--top-k <k>] [--watchlist <token-id,…>]  (\"\" as the lens id "
            "clears the readout)")


def _malformed(reason: str, name: str, program: str,
               repair: str | None = None) -> store.MeasurementDeclarationError:
    return store.MeasurementDeclarationError(
        reason, repair=repair or usage(name, program))


# --- parsing (Swift twin: JLensReadoutAuthoring.parse*) ------------------------

def parse_layers(text: str, flag: str, *, name: str,
                 program: str = vocabulary.CLIENT_PROGRAM) -> list[int]:
    """Comma-separated layer indices or inclusive ranges (``20-24``), sorted
    and de-duplicated."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise _malformed(f"{flag} is empty — name at least one layer",
                         name, program)
    layers: set[int] = set()
    for part in parts:
        match = _LAYER_ITEM.fullmatch(part)
        if match is None:
            raise _malformed(
                f"{flag} expects comma-separated layer indices or ranges such "
                f"as 20-24 — got '{part}'", name, program)
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) is not None else low
        if low > high:
            raise _malformed(
                f"{flag} range '{part}' runs backwards — write the lower "
                "layer first", name, program)
        layers.update(range(low, high + 1))
    return sorted(layers)


def parse_watchlist(text: str, *, name: str,
                    program: str = vocabulary.CLIENT_PROGRAM) -> list[int]:
    """Comma-separated token ids, in the order given. ``""`` is no watchlist."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    ids: list[int] = []
    for part in parts:
        if _TOKEN_ID.fullmatch(part) is None or int(part) > MAX_TOKEN_ID:
            raise _malformed(
                f"--watchlist expects comma-separated token ids — got '{part}'",
                name, program)
        if int(part) in ids:
            raise _malformed(
                f"--watchlist names token {int(part)} twice — each watched "
                "token is one column of the trace", name, program)
        ids.append(int(part))
    if len(ids) > MAX_WATCHLIST:
        raise _malformed(
            f"--watchlist names {len(ids)} tokens; a readout watches at most "
            f"{MAX_WATCHLIST}", name, program)
    return ids


def parse_whole(text: str, flag: str, *, low: int, high: int, name: str,
                program: str = vocabulary.CLIENT_PROGRAM) -> int:
    if _WHOLE.fullmatch(text.strip()) is None or not (
            low <= int(text.strip()) <= high):
        raise _malformed(
            f"{flag} must be a whole number from {low} to {high} — got "
            f"'{text}'", name, program)
    return int(text.strip())


def parse_sha256(text: str, flag: str, *, name: str,
                 program: str = vocabulary.CLIENT_PROGRAM) -> str:
    if _SHA256.fullmatch(text.strip()) is None:
        raise _malformed(
            f"{flag} must be a SHA-256 digest: 64 lower-case hexadecimal "
            f"characters — got '{text}'", name, program)
    return text.strip()


def compact_layers(layers) -> str:
    """``[0, 1, 2, 5]`` → ``"0-2,5"``: the spelling the layer flags accept."""
    out: list[str] = []
    ordered = sorted(set(int(x) for x in layers))
    index = 0
    while index < len(ordered):
        start = end = ordered[index]
        while index + 1 < len(ordered) and ordered[index + 1] == end + 1:
            index += 1
            end = ordered[index]
        out.append(str(start) if start == end else f"{start}-{end}")
        index += 1
    return ",".join(out)


# --- the settings ---------------------------------------------------------------

class Settings:
    """What the researcher declared, already parsed and range-checked."""

    def __init__(self, *, layers, top_k=0, top_k_layers=None, watchlist=(),
                 companion=True, budget=None, qualification_id=None,
                 tokenizer_hash=None, record_token_ids=False):
        self.layers = list(layers)
        self.top_k = int(top_k)
        self.top_k_layers = (None if top_k_layers is None
                             else list(top_k_layers))
        self.watchlist = list(watchlist)
        self.companion = bool(companion)
        self.budget = dict(budget or {})
        self.qualification_id = qualification_id or None
        self.tokenizer_hash = tokenizer_hash or None
        self.record_token_ids = bool(record_token_ids)


def check_settings(settings: Settings, *, name: str,
                   program: str = vocabulary.CLIENT_PROGRAM) -> None:
    """The rules between settings. Swift twin: ``checkSettings``."""
    if settings.top_k_layers is not None:
        if settings.top_k <= 0:
            raise _malformed(
                "--top-k-layers chooses which layers carry the top-k readout, "
                "and no --top-k width is declared", name, program)
        outside = sorted(set(settings.top_k_layers) - set(settings.layers))
        if outside:
            raise _malformed(
                f"--top-k-layers names {compact_layers(outside)}, which "
                "--layers does not arm — top-k is read only at armed layers",
                name, program)
    if not settings.watchlist and settings.top_k <= 0:
        raise _malformed(
            "the readout declares neither a --watchlist nor a --top-k width, "
            "so it would record nothing", name, program)


def build_block(settings: Settings, *, lens_id: str, lens_sha256: str) -> dict:
    """The ``jlensReadout`` block exactly as both clients write it.

    Absent settings are absent keys, never ``null``: the block is opaque to
    the design identity, so a ``null`` would be hashed as data. ``configHash``
    is written only beside a ``tokenizerHash``, because run start hashes the
    tokenizer it finds and a hash computed without one could never match.
    """
    block: dict = {
        "lensID": lens_id,
        "lensSHA256": lens_sha256,
        "layers": list(settings.layers),
        "topK": settings.top_k,
        "watchlist": list(settings.watchlist),
        "logitLensCompanion": settings.companion,
    }
    if settings.top_k_layers is not None:
        block["topKLayers"] = list(settings.top_k_layers)
    if settings.budget:
        block["budget"] = dict(settings.budget)
    if settings.qualification_id:
        block["qualificationID"] = settings.qualification_id
    if settings.tokenizer_hash:
        block["tokenizerHash"] = settings.tokenizer_hash
        block["configHash"] = configuration_hash(block)
    return block


def readout_config(block: dict) -> ReadoutConfig:
    """The configuration run start builds from this block (``run_readouts``)."""
    return ReadoutConfig(
        layers=[int(x) for x in (block.get("layers") or [])],
        watchlist=[int(x) for x in (block.get("watchlist") or [])],
        topK=int(block.get("topK") or 0),
        topKLayers=[int(x) for x in (block.get("topKLayers") or [])],
        logitLensCompanion=bool(block.get("logitLensCompanion", True)))


def readout_budget(block: dict) -> Budget:
    """The budget run start builds from this block (``run_readouts``)."""
    return Budget(**{k: v for k, v in (block.get("budget") or {}).items()
                     if k in Budget.__dataclass_fields__})


def configuration_hash(block: dict) -> str:
    """The hash run start computes for this block, by the engine's own payload
    (``trace.readout_config_payload``), so the pin cannot disagree with it."""
    lens = SimpleNamespace(lensID=block["lensID"],
                           source=SimpleNamespace(tensorSHA256=block["lensSHA256"]))
    return config_hash(readout_config_payload(
        readout_config(block), record=lens,
        qualification_id=block.get("qualificationID"),
        budget=readout_budget(block),
        tokenizer_hash=block.get("tokenizerHash")))


def estimate(block: dict, max_tokens: int) -> dict:
    """ONE response priced at the token cap by the engine's ``preflight``, and
    how many responses the declared budget covers. Swift twin: ``estimate``."""
    config, budget = readout_config(block), readout_budget(block)
    one = preflight(config, generations=1, max_new_tokens=max_tokens,
                    budget=budget)
    per_response = {key: one[key] for key in
                    ("observations", "fullVocabProjections",
                     "projectedTraceBytes")}
    allowed, limited_by = None, None
    for ceiling, quantity in PROPORTIONAL_CEILINGS:
        need = per_response[quantity]
        if need <= 0:
            continue
        fits = getattr(budget, ceiling) // need
        if allowed is None or fits < allowed:
            allowed, limited_by = fits, ceiling
    return {
        "maxTokens": int(max_tokens),
        "armedLayers": one["armedLayers"],
        "topKLayers": len(config.armed_topk_layers()),
        "perResponse": per_response,
        "budget": asdict(budget),
        "responsesAllowed": allowed,
        "limitedBy": limited_by,
        "problems": list(one["problems"]),
    }


def budget_refusal(priced: dict, *, name: str,
                   program: str = vocabulary.CLIENT_PROGRAM
                   ) -> store.MeasurementDeclarationError | None:
    """The refusal for a declaration the engine would refuse at run start
    whatever the study's size, or None."""
    if not priced["problems"]:
        return None
    return _malformed(
        "the J-lens readout cannot fit its declared budget even for one "
        f"response at the {priced['maxTokens']}-token cap: "
        + "; ".join(priced["problems"]), name, program,
        repair=(f"{program} experiment {VERB} {name} <lens-id> … with fewer "
                "--layers or --top-k-layers, or with a raised ceiling: "
                + ", ".join(flag for flag, _field in BUDGET_FLAGS)))


def mac_command(name: str, block: dict, *, lens_model: str,
                fitted_layers, record_token_ids: bool = False) -> str:
    """The ``steerlab-cli`` spelling that writes this exact block."""
    parts = [vocabulary.MAC_PROGRAM, "experiment", VERB, name, block["lensID"],
             "--lens-sha256", block["lensSHA256"],
             "--lens-model", lens_model,
             "--fitted-layers", compact_layers(fitted_layers),
             "--layers", compact_layers(block["layers"])]
    if block.get("topK"):
        parts += ["--top-k", str(block["topK"])]
    if "topKLayers" in block:
        parts += ["--top-k-layers", compact_layers(block["topKLayers"])]
    if block.get("watchlist"):
        parts += ["--watchlist", ",".join(str(t) for t in block["watchlist"])]
    if not block.get("logitLensCompanion", True):
        parts.append("--no-companion")
    if block.get("qualificationID"):
        parts += ["--qualification", block["qualificationID"]]
    if block.get("tokenizerHash"):
        parts += ["--tokenizer-hash", block["tokenizerHash"]]
    for flag, field in BUDGET_FLAGS:
        if field in (block.get("budget") or {}):
            parts += [flag, str(block["budget"][field])]
    if record_token_ids:
        parts.append("--record-token-ids")
    return " ".join(parts)


# --- the tokenizer pin (torch-free twin of jlens.derive.tokenizer_identity_hash)

#: The files that define tokenizer identity, in hashing order. The SAME tuple
#: as ``jlens.derive._TOKENIZER_FILES``; a test holds the two equal.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json")


def _hub_directory() -> str:
    """The model cache's hub directory, resolved as ``huggingface_hub`` does
    (``steering.model_loader.hf_hub_dir``, which imports torch, so it is
    restated here as ``api.housekeeping`` restates it)."""
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return hub
    home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def tokenizer_hash(model_id: str, revision: str | None = None) -> str | None:
    """SHA-256 over the cached snapshot's tokenizer files, or None when the
    snapshot (or every tokenizer file) is absent. The same bytes and order as
    ``jlens.derive.tokenizer_identity_hash``: a pinned revision reads only
    that revision's snapshot, and no revision reads the newest one."""
    base = os.path.join(_hub_directory(),
                        "models--" + model_id.replace("/", "--"), "snapshots")
    if revision:
        snapshot = os.path.join(base, revision)
        if not os.path.isdir(snapshot):
            return None
    else:
        try:
            found = sorted(os.path.join(base, entry)
                           for entry in os.listdir(base))
        except OSError:
            return None
        if not found:
            return None
        snapshot = found[-1]
    digest = hashlib.sha256()
    present = False
    for filename in TOKENIZER_FILES:
        path = os.path.join(snapshot, filename)
        if not os.path.exists(path):
            continue
        present = True
        digest.update(filename.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest() if present else None


# --- the verb -------------------------------------------------------------------

def _require_draft(document: dict, name: str) -> None:
    """Refuse a non-draft BEFORE the lens is read, so a frozen study answers
    with its one repair whatever else is wrong. Swift twin: ``updateDraft``."""
    status = document.get("status")
    if status != "draft":
        raise store.ExperimentStoreError(
            f"experiment '{name}' is {status} — duplicate it to iterate",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(vocabulary.authoring(
                        f"experiment duplicate {name} {name}-v2",
                        f"experiment {VERB} {name}-v2 <lens-id> …")
                    + "  (a frozen study is immutable; the duplicate is a "
                      "draft again)"))


def _lens_record(lens_id: str, name: str, root: str | None):
    try:
        return lens_store.resolve(lens_id, root)
    except JLensError as exc:
        raise store.ExperimentStoreError(
            f"no imported J-lens '{lens_id}' in this workspace "
            f"(runs/jlens-lenses/{lens_id}/lens.json): {exc}",
            gate=lifecycle_gates.MISSING_PREREQUISITE,
            repair=(f"{vocabulary.ENGINE_PROGRAM} jlens list  (the lenses "
                    "this workspace has), or import one with "
                    f"{vocabulary.ENGINE_PROGRAM} jlens import <model-id>; "
                    f"then {usage(name)}")) from exc


def unfitted_reason(unfitted, lens_id: str, fitted) -> str:
    """Swift twin: ``unfittedReason``."""
    one = len(unfitted) == 1
    return (f"{'layer' if one else 'layers'} {compact_layers(unfitted)} "
            f"{'is not a fitted source layer' if one else 'are not fitted source layers'}"
            f" of '{lens_id}' (it fitted {compact_layers(fitted)})")


def _check_lens(record, document: dict, settings: Settings, *, name: str,
                lens_id: str) -> None:
    """The lens must be armable, belong to this study's model, and have
    fitted every requested layer."""
    if record.converted is None or not record.source.tensorSHA256:
        missing = ("no converted per-layer artifact"
                   if record.converted is None else "no tensor hash")
        raise store.ExperimentStoreError(
            f"J-lens '{lens_id}' records {missing}, so a run could not arm it "
            "and the study could not pin its bytes",
            gate=lifecycle_gates.MISSING_PREREQUISITE,
            repair=(f"{vocabulary.ENGINE_PROGRAM} jlens import "
                    f"{record.fit.modelID or '<model-id>'}  (re-import the "
                    f"lens), then {usage(name)}"))
    model_id = document.get("modelID") or ""
    revision = document.get("modelRevision") or None
    fitted_on = record.fit.modelID
    if not fitted_on or fitted_on != model_id:
        raise store.ExperimentStoreError(
            f"J-lens '{lens_id}' was fitted on "
            f"{repr(fitted_on) if fitted_on else 'an unrecorded model'}, and "
            f"this study measures '{model_id}' — a lens reads only the model "
            "it was fitted on",
            gate=lifecycle_gates.ARTIFACT_PIN,
            repair=(f"{vocabulary.ENGINE_PROGRAM} jlens list  (find a lens "
                    f"fitted on {model_id}), then "
                    f"{vocabulary.CLIENT_PROGRAM} experiment {VERB} {name} "
                    "<that-lens-id> …"))
    fitted_revision = record.fit.revision if record.fit.revisionKnown else None
    if fitted_revision and revision and fitted_revision != revision:
        raise store.ExperimentStoreError(
            f"J-lens '{lens_id}' was fitted on {model_id}@{fitted_revision}, "
            f"and this study pins revision {revision}",
            gate=lifecycle_gates.ARTIFACT_PIN,
            repair=(f"{vocabulary.CLIENT_PROGRAM} experiment pin-revision "
                    f"{name} {fitted_revision}, or name a lens fitted on "
                    f"revision {revision}"))
    unfitted = sorted(set(settings.layers) - set(record.sourceLayers))
    if unfitted:
        raise _malformed(
            unfitted_reason(unfitted, lens_id, record.sourceLayers)
            + f" — the target layer {record.targetLayer} has no Jacobian by "
              "construction", name, vocabulary.CLIENT_PROGRAM,
            repair=(f"{vocabulary.CLIENT_PROGRAM} experiment {VERB} {name} "
                    f"{lens_id} --layers <a subset of "
                    f"{compact_layers(record.sourceLayers)}> …"))


def _resolve_qualification(record, document: dict, settings: Settings, *,
                           name: str, lens_id: str) -> str | None:
    """A named qualification must pass for this study's runtime and cover the
    armed layers. With none named, the newest passing one is pinned — the one
    run start would otherwise resolve, so the configuration hash names it."""
    model_id = document.get("modelID") or ""
    revision = document.get("modelRevision") or ""
    dtype = document.get("dtype") or ""
    found = record.qualification_for(
        model_id, revision, dtype, document.get("quantization"),
        layers=settings.layers, qualification_id=settings.qualification_id)
    if settings.qualification_id and found is None:
        raise _malformed(
            f"qualification '{settings.qualification_id}' of '{lens_id}' does "
            f"not pass for {model_id}@{revision or '?'}/{dtype or '?'} over "
            f"layers {compact_layers(settings.layers)} — a pinned "
            "qualification must be bound to these lens bytes and cover the "
            "armed layers", name, vocabulary.CLIENT_PROGRAM,
            repair=(f"{vocabulary.ENGINE_PROGRAM} jlens inspect {lens_id}  "
                    "(its qualifications), then name a passing one with "
                    "--qualification, or leave the flag out"))
    return found.qualificationID if found is not None else None


def notes_for(document: dict, block: dict, *, tier: str | None = None,
              tokenizer_found: bool = True) -> list[str]:
    """What the researcher should know before running. Never a refusal."""
    notes: list[str] = []
    model_id = document.get("modelID") or "the study model"
    if "tokenizerHash" not in block:
        notes.append(
            f"The tokenizer of {model_id} is not in this machine's model "
            "cache, so tokenizerHash and configHash are not pinned. The study "
            "can run as exploratory, and freeze refuses until both are "
            "pinned. Install the model's files here, or pass --tokenizer-hash "
            "from the machine that runs the study, and run this verb again."
            if not tokenizer_found else
            "tokenizerHash and configHash are not pinned, so freeze refuses "
            "until they are.")
    if "qualificationID" not in block:
        notes.append(
            "No qualification is pinned: no passing qualification of this "
            "lens covers this runtime and these layers yet, so a run is "
            "exploratory and freeze refuses. Qualify the runtime "
            f"({vocabulary.ENGINE_PROGRAM} jlens qualify), then run this "
            "verb again so the pin names the qualification.")
    if tier is not None and tier != "evidence":
        notes.append(
            f"This lens is {tier}-tier for {model_id}: it exercises the "
            "readout and produces no evidence, so freeze refuses it.")
    if not (document.get("modelRevision") and document.get("dtype")):
        notes.append(
            "The study pins no model revision or dtype, and a qualification "
            "is keyed by both: pin them before freezing.")
    if not document.get("recordTokenIDs"):
        notes.append(
            "recordTokenIDs is off, so the exact sampled tokens are not kept "
            "and the run cannot be replayed exactly later. Add "
            "--record-token-ids to keep them.")
    return notes


def declare(name: str, lens_id: str, settings: Settings, *,
            root: str | None = None, dry_run: bool = False) -> dict:
    """Validate, pin, price, and (unless ``dry_run``) write the readout."""
    document = store.load_raw(name, root)
    _require_draft(document, name)
    record = _lens_record(lens_id, name, root)
    _check_lens(record, document, settings, name=name, lens_id=lens_id)
    settings.qualification_id = _resolve_qualification(
        record, document, settings, name=name, lens_id=lens_id)
    model_id = document.get("modelID") or ""
    local = tokenizer_hash(model_id, document.get("modelRevision") or None)
    if settings.tokenizer_hash and local and settings.tokenizer_hash != local:
        raise _malformed(
            f"--tokenizer-hash {settings.tokenizer_hash[:12]}… disagrees with "
            f"the tokenizer of {model_id} in this machine's model cache "
            f"({local[:12]}…) — every watched token id would name a "
            "different piece", name, vocabulary.CLIENT_PROGRAM,
            repair=f"leave --tokenizer-hash out: this client reads it from "
                   f"the cache. {usage(name)}")
    settings.tokenizer_hash = settings.tokenizer_hash or local
    block = build_block(settings, lens_id=lens_id,
                        lens_sha256=record.source.tensorSHA256)
    # The engine's own default for an absent cap (``Manifest.max_tokens``).
    cap = document.get("maxTokens")
    priced = estimate(block, int(cap) if cap is not None else 512)
    refusal = budget_refusal(priced, name=name)
    if refusal is not None:
        raise refusal
    from ..jlens import importer
    tier = importer.tier_of(model_id, record)[0]
    if not dry_run:
        document[MANIFEST_KEY] = block
        if settings.record_token_ids:
            document["recordTokenIDs"] = True
        store.save_raw(document, root)
    lens_model = (f"{record.fit.modelID}@{record.fit.revision}"
                  if record.fit.revisionKnown and record.fit.revision
                  else record.fit.modelID)
    view = dict(document)
    if settings.record_token_ids:
        view["recordTokenIDs"] = True
    return {
        "document": document,
        "block": block,
        "estimate": priced,
        "tier": tier,
        "pins": {
            "lensID": lens_id,
            "lensSHA256": block["lensSHA256"],
            "lensModel": lens_model,
            "fittedLayers": compact_layers(record.sourceLayers),
            "tokenizerHash": block.get("tokenizerHash"),
            "qualificationID": block.get("qualificationID"),
        },
        "macCommand": mac_command(
            name, block, lens_model=lens_model,
            fitted_layers=record.sourceLayers,
            record_token_ids=settings.record_token_ids),
        "notes": notes_for(view, block, tier=tier,
                           tokenizer_found=local is not None),
    }


def clear(name: str, *, root: str | None = None,
          dry_run: bool = False) -> tuple[dict, bool]:
    """Remove the readout. Returns the document and whether one was there.
    ``recordTokenIDs`` is left as it is: keeping tokens is harmless, and it is
    a separate setting."""
    document = store.load_raw(name, root)
    _require_draft(document, name)
    existed = MANIFEST_KEY in document
    if existed and not dry_run:
        document.pop(MANIFEST_KEY, None)
        store.save_raw(document, root)
    return document, existed
