"""Frozen LoRA dataset objects: strict loaders, split integrity, tokenization
and label masking (the data half of ``docs/CLUSTER-LORA-READINESS.md``
§2.1–2.3).

The exploratory ingestion this supersedes
(:func:`lora_train.ingest_documents` + ``_chunk``) concatenated every source
file into one token stream, cut it into fixed windows, and called the trailing
10% "validation" — so document boundaries became training examples, the
researcher-authored ``validation/`` folder was ignored, and the split depended
on input order. None of that is evidence-grade. This module is the replacement:

- **rows are examples.** Each JSONL row is tokenized on its own and never
  concatenated with its neighbour. A row too long for the declared maximum is
  either split into deterministic, provenance-stamped windows or refused —
  chosen by declared policy, never silently.
- **splits are researcher-authored inputs**, verified by SHA-256 against the
  bytes the dataset package pinned, with intra-split duplicate and cross-split
  leakage refusals.
- **refusals are loud and name ``path:line``.** The Swift trainer
  (``FineTuneTrainer.instructionExample(from:)``) silently drops a malformed
  row; the server refuses it (plan §2.3.5). A dropped row is a dataset whose
  hash no longer describes what trained.
- **instruction rows train on the assistant only**: every prompt, system,
  template-control and padding label is ``-100`` (plan §2.3).

Stdlib only — no torch, no transformers. :func:`tokenize_rows` takes a
duck-typed tokenizer (anything with ``__call__``, ``apply_chat_template``,
``eos_token_id`` and ``chat_template`` — an HF tokenizer satisfies it), so
every rule here is testable CPU-only against a fixture tokenizer.

Row identity (contract §2, and the reason two engines can agree on a dataset
without shipping its bytes):

- ``rowHash`` = SHA-256 of the canonical JSON of the NORMALIZED row;
- a file's ``rowsRoot`` = SHA-256 of the newline-joined row hashes in file
  order (so row order is part of the identity, not just row content);
- a file's ``sha256`` = SHA-256 of its raw bytes (identical to
  ``scripts/lora_dataset_utils.sha256_file``, which is what the dataset
  builders stamp into the package manifests).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field

#: The two evidence-grade training modes. ``legacy_inline`` is the trainer's
#: exploratory mode and never reaches this module.
DOCUMENT = "document"
INSTRUCTION_CHAT = "instruction_chat"
TRAINING_MODES = (DOCUMENT, INSTRUCTION_CHAT)

#: What to do with a row longer than ``max_sequence_tokens``. Never truncate:
#: a truncated example trains on a sentence fragment whose hash still claims
#: to be the whole row.
SPLIT = "split"
REFUSE = "refuse"
LONG_DOCUMENT_POLICIES = (SPLIT, REFUSE)

#: Strict key sets (contract §1). Anything else in a row refuses: an unknown
#: key is either a typo that silently dropped its content or a field this
#: loader is not honouring, and both are dataset bugs.
DOCUMENT_KEYS = frozenset({"text", "id"})
INSTRUCTION_KEYS = frozenset({"system", "user", "assistant", "prompt",
                              "completion", "id"})

#: Ingestion aliases, matching the Swift trainer (FineTuneTrainer.swift:385-392).
#: Normalized/persisted rows use only the canonical spellings.
INSTRUCTION_ALIASES = {"prompt": "user", "completion": "assistant"}

#: Label id for "this position does not contribute to the loss" (the HF
#: cross-entropy ignore_index).
IGNORE_LABEL = -100

ASSISTANT_ONLY = "assistantOnly"
FULL_SEQUENCE = "fullSequence"


class LoRADataError(ValueError):
    """A dataset refusal: bad row schema, hash mismatch, split leakage, or a
    row the declared masking/length policy cannot honour. Always names the
    offending file (and line, where a row is at fault)."""


# --- hashing ---------------------------------------------------------------


def canonical_json(value: object) -> str:
    """The canonical encoding row hashes are taken over (contract §2)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """SHA-256 of a file's raw bytes (twin of
    ``scripts/lora_dataset_utils.sha256_file`` — the dataset builders' stamp)."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def rows_root(row_hashes: list[str] | tuple[str, ...]) -> str:
    """Ordered rows root for one file: SHA-256 over the newline-joined row
    hashes IN FILE ORDER. Reordering a file changes its root, which is the
    point — the deterministic dataloader consumes rows in file order."""
    return _sha256_text("\n".join(row_hashes))


# --- rows ------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One normalized dataset row plus where it came from.

    ``fields`` holds only canonical spellings: ``{"text"}`` for document rows,
    ``{"user", "assistant"}`` (+ ``"system"`` when nonempty) for instruction
    rows, plus ``"id"`` when the source row carried one. String values are
    whitespace-trimmed, matching the Swift trainer.
    """

    path: str                # declared (workspace-relative) path, for messages
    line: int                # 1-based line number in that file
    fields: dict
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_hash",
                           _sha256_text(canonical_json(self.fields)))

    @property
    def row_id(self) -> str | None:
        value = self.fields.get("id")
        return value if isinstance(value, str) else None

    @property
    def content_key(self) -> str:
        """The row's TEXT identity, ignoring ``id``. Cross-split leakage is
        about content: the same passage under two ids is still the same
        passage in both splits."""
        if "text" in self.fields:
            return self.fields["text"]
        return "\x00".join((self.fields.get("system", ""),
                            self.fields.get("user", ""),
                            self.fields.get("assistant", "")))

    def with_assistant(self, assistant: str) -> "Row":
        """A copy carrying a different assistant reply (the control-arm
        transform). The row hash is recomputed, so a shuffled control's rows
        root can never collide with its treatment arm's."""
        fields = dict(self.fields)
        fields["assistant"] = assistant
        return Row(path=self.path, line=self.line, fields=fields)


@dataclass(frozen=True)
class DatasetFile:
    """One declared split file. ``expected_sha256`` is the package manifest's
    pin — ``None`` only in exploratory mode (evidence-grade loading refuses an
    unpinned file)."""

    path: str
    expected_sha256: str | None = None
    role: str = "train"      # "train" | "validation"


@dataclass(frozen=True)
class LoRADatasetSpec:
    """The frozen description of a training dataset: which files, in which
    mode, under which length policy. Everything here is manifest DATA — the
    trainer resolves it, it never guesses."""

    training_mode: str
    train_files: list[DatasetFile]
    validation_files: list[DatasetFile]
    max_sequence_tokens: int
    long_document_policy: str = SPLIT
    chunk_overlap_tokens: int = 64
    bundle_id: str | None = None
    manifest_path: str | None = None
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if self.training_mode not in TRAINING_MODES:
            raise LoRADataError(
                f"unknown trainingMode {self.training_mode!r} — expected one "
                f"of {', '.join(TRAINING_MODES)}")
        if self.long_document_policy not in LONG_DOCUMENT_POLICIES:
            raise LoRADataError(
                f"unknown longDocumentPolicy {self.long_document_policy!r} — "
                f"expected one of {', '.join(LONG_DOCUMENT_POLICIES)}")
        if self.max_sequence_tokens < 1:
            raise LoRADataError(
                f"maxSequenceTokens must be positive (got "
                f"{self.max_sequence_tokens})")
        if self.chunk_overlap_tokens < 0:
            raise LoRADataError(
                f"chunkOverlapTokens must not be negative (got "
                f"{self.chunk_overlap_tokens})")
        if self.chunk_overlap_tokens >= self.max_sequence_tokens:
            # stride = max - overlap; a non-positive stride windows forever.
            raise LoRADataError(
                f"chunkOverlapTokens ({self.chunk_overlap_tokens}) must be "
                f"smaller than maxSequenceTokens ({self.max_sequence_tokens})")


@dataclass(frozen=True)
class LoadedFile:
    path: str
    role: str
    sha256: str
    rows: tuple[Row, ...]
    rows_root: str

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256,
                "rows": len(self.rows), "rowsRoot": self.rows_root}


@dataclass(frozen=True)
class LoadedRows:
    """Both splits, normalized, hash-verified and leakage-checked."""

    spec: LoRADatasetSpec
    evidence_grade: bool
    train_files: tuple[LoadedFile, ...]
    validation_files: tuple[LoadedFile, ...]

    @property
    def train_rows(self) -> tuple[Row, ...]:
        return tuple(row for file in self.train_files for row in file.rows)

    @property
    def validation_rows(self) -> tuple[Row, ...]:
        return tuple(row for file in self.validation_files for row in file.rows)

    @property
    def counts(self) -> dict:
        return {"trainFiles": len(self.train_files),
                "validationFiles": len(self.validation_files),
                "trainRows": len(self.train_rows),
                "validationRows": len(self.validation_rows)}


# --- strict row parsing ----------------------------------------------------


def _require_string(value: object, *, path: str, line: int, key: str) -> str:
    if not isinstance(value, str):
        raise LoRADataError(
            f"{path}:{line}: '{key}' must be a JSON string (got "
            f"{type(value).__name__})")
    text = value.strip()
    if not text:
        raise LoRADataError(f"{path}:{line}: '{key}' is empty")
    return text


def _document_fields(obj: dict, *, path: str, line: int) -> dict:
    unknown = sorted(set(obj) - DOCUMENT_KEYS)
    if unknown:
        raise LoRADataError(
            f"{path}:{line}: unknown key(s) {', '.join(unknown)} in a document "
            f"row — allowed: {', '.join(sorted(DOCUMENT_KEYS))}")
    if "text" not in obj:
        raise LoRADataError(f"{path}:{line}: document row has no 'text'")
    fields = {"text": _require_string(obj["text"], path=path, line=line,
                                      key="text")}
    if "id" in obj:
        fields["id"] = _require_string(obj["id"], path=path, line=line, key="id")
    return fields


def _instruction_fields(obj: dict, *, path: str, line: int) -> dict:
    unknown = sorted(set(obj) - INSTRUCTION_KEYS)
    if unknown:
        raise LoRADataError(
            f"{path}:{line}: unknown key(s) {', '.join(unknown)} in an "
            f"instruction/chat row — allowed: "
            f"{', '.join(sorted(INSTRUCTION_KEYS))}")
    # Both spellings of one field is ambiguous, not a convenience: the Swift
    # reader would silently prefer the canonical one and train on data the
    # author may not have meant.
    for alias, canonical in INSTRUCTION_ALIASES.items():
        if alias in obj and canonical in obj:
            raise LoRADataError(
                f"{path}:{line}: row carries both '{alias}' and '{canonical}' "
                "— use one spelling")
    fields: dict = {}
    for canonical in ("user", "assistant"):
        alias = next(a for a, c in INSTRUCTION_ALIASES.items() if c == canonical)
        key = canonical if canonical in obj else (alias if alias in obj else None)
        if key is None:
            raise LoRADataError(
                f"{path}:{line}: instruction/chat row has no '{canonical}' "
                f"(alias '{alias}')")
        fields[canonical] = _require_string(obj[key], path=path, line=line,
                                            key=key)
    if "system" in obj:
        # Optional, but a present-and-empty system field is an authoring bug
        # worth surfacing rather than quietly ignoring.
        fields["system"] = _require_string(obj["system"], path=path, line=line,
                                           key="system")
    if "id" in obj:
        fields["id"] = _require_string(obj["id"], path=path, line=line, key="id")
    return fields


def parse_rows(text: str, *, path: str, training_mode: str) -> tuple[Row, ...]:
    """Parse one JSONL file's TEXT into normalized rows under the strict
    schema. Blank lines are skipped (they carry no content); everything else
    must be a JSON object obeying the mode's key set."""
    rows: list[Row] = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as exc:
            raise LoRADataError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise LoRADataError(f"{path}:{number}: expected a JSON object")
        if training_mode == DOCUMENT:
            fields = _document_fields(obj, path=path, line=number)
        else:
            fields = _instruction_fields(obj, path=path, line=number)
        rows.append(Row(path=path, line=number, fields=fields))
    return tuple(rows)


def _load_file(file: DatasetFile, *, spec: LoRADatasetSpec,
               evidence_grade: bool, root: str | None) -> LoadedFile:
    resolved = file.path if os.path.isabs(file.path) or root is None \
        else os.path.join(root, file.path)
    if not os.path.isfile(resolved):
        raise LoRADataError(
            f"{file.path}: {file.role} file not found (resolved to {resolved})")
    digest = sha256_file(resolved)
    if file.expected_sha256:
        if digest != file.expected_sha256.lower():
            raise LoRADataError(
                f"{file.path}: sha256 mismatch — the dataset pinned "
                f"{file.expected_sha256}, the file on disk is {digest}. The "
                "training data drifted after it was pinned; re-pin "
                "deliberately or restore the frozen bytes.")
    elif evidence_grade:
        raise LoRADataError(
            f"{file.path}: evidence-grade training requires a pinned sha256 "
            "for every split file — an unpinned file cannot be shown to be "
            "the bytes that trained")
    with open(resolved, "r", encoding="utf-8") as handle:
        text = handle.read()
    rows = parse_rows(text, path=file.path, training_mode=spec.training_mode)
    if not rows:
        raise LoRADataError(f"{file.path}: no rows")
    return LoadedFile(path=file.path, role=file.role, sha256=digest, rows=rows,
                      rows_root=rows_root([row.row_hash for row in rows]))


def load_split_rows(spec: LoRADatasetSpec, *, evidence_grade: bool,
                    root: str | None = None) -> LoadedRows:
    """Load, verify and leakage-check both splits.

    Verifies in order: file presence, pinned SHA-256, strict row schema,
    intra-split duplicate row hashes, cross-split exact overlap (row hash OR
    content). ``evidence_grade`` additionally requires a nonempty validation
    split and a pinned hash on every file.

    ``root`` resolves workspace-relative declared paths (``None`` = paths are
    used as given). The DECLARED path — not the resolved absolute one — is
    what refusals and provenance name, so a bundle reads the same on the Mac
    and on the cluster.
    """
    if evidence_grade and not spec.validation_files:
        raise LoRADataError(
            "evidence-grade training requires an explicit validation split — "
            "the legacy fractional split is exploratory only (plan §2.1)")
    if not spec.train_files:
        raise LoRADataError("no training files declared")

    declared: dict[str, str] = {}
    for file in list(spec.train_files) + list(spec.validation_files):
        if file.path in declared:
            raise LoRADataError(
                f"{file.path}: declared twice ({declared[file.path]} and "
                f"{file.role}) — a file cannot be in both splits, and listing "
                "it twice in one split double-counts every row")
        declared[file.path] = file.role

    train_files = tuple(
        _load_file(DatasetFile(f.path, f.expected_sha256, "train"), spec=spec,
                   evidence_grade=evidence_grade, root=root)
        for f in spec.train_files)
    validation_files = tuple(
        _load_file(DatasetFile(f.path, f.expected_sha256, "validation"),
                   spec=spec, evidence_grade=evidence_grade, root=root)
        for f in spec.validation_files)

    for role, files in (("train", train_files), ("validation", validation_files)):
        seen: dict[str, Row] = {}
        for file in files:
            for row in file.rows:
                first = seen.get(row.row_hash)
                if first is not None:
                    raise LoRADataError(
                        f"{row.path}:{row.line}: duplicate row (identical to "
                        f"{first.path}:{first.line}) in the {role} split — a "
                        "repeated row silently reweights the example it "
                        "repeats")
                seen[row.row_hash] = row

    train_by_hash = {row.row_hash: row for file in train_files for row in file.rows}
    train_by_content = {row.content_key: row for file in train_files
                        for row in file.rows}
    for file in validation_files:
        for row in file.rows:
            other = train_by_hash.get(row.row_hash) or \
                train_by_content.get(row.content_key)
            if other is not None:
                raise LoRADataError(
                    f"{row.path}:{row.line}: also present in the training "
                    f"split at {other.path}:{other.line} — validation measured "
                    "on trained rows measures memorization, not held-out "
                    "behaviour")

    return LoadedRows(spec=spec, evidence_grade=evidence_grade,
                      train_files=train_files,
                      validation_files=validation_files)


# --- tokenization and masking ----------------------------------------------


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    row_hash: str
    row_id: str | None = None
    chunk_index: int | None = None      # None = the row fit whole
    chunk_count: int | None = None
    chunk_overlap_tokens: int | None = None

    @property
    def target_token_count(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_LABEL)

    def provenance(self) -> dict:
        out: dict = {"rowHash": self.row_hash, "tokens": len(self.input_ids),
                     "targetTokens": self.target_token_count}
        if self.row_id is not None:
            out["rowID"] = self.row_id
        if self.chunk_index is not None:
            out["chunkIndex"] = self.chunk_index
            out["chunkCount"] = self.chunk_count
            out["chunkOverlapTokens"] = self.chunk_overlap_tokens
        return out


@dataclass(frozen=True)
class TokenizedData:
    train: tuple[TokenizedExample, ...]
    validation: tuple[TokenizedExample, ...]
    stats: dict
    template: dict


def _as_id_list(value: object) -> list[int]:
    """Normalize whatever a tokenizer hands back (list, BatchEncoding, dict,
    nested batch) into a flat list of ints."""
    ids = getattr(value, "input_ids", None)
    if ids is None and isinstance(value, dict):
        ids = value.get("input_ids")
    if ids is not None:
        value = ids
    items = list(value)  # type: ignore[arg-type]
    if items and isinstance(items[0], (list, tuple)):
        items = list(items[0])
    return [int(item) for item in items]


def _supports_system_role(tokenizer) -> bool:
    """Probe the template with a system message. Gemma's template raises on
    the system role (it has none — system text is prepended to the first user
    turn), and that raise is the honest signal; a template that renders one is
    trusted to honour it."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "probe"},
             {"role": "user", "content": "probe"}],
            add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001 - any template failure means "no system role"
        return False
    return True


def system_fold_required(tokenizer, model_id: str) -> bool:
    """Whether nonempty system text must be folded into the user turn.

    Mirrors FineTuneTrainer.swift:585-593 (model id contains "gemma"), plus a
    template probe so a non-Gemma template without a system role folds too
    instead of refusing every row."""
    if "gemma" in (model_id or "").lower():
        return True
    return not _supports_system_role(tokenizer)


def _instruction_messages(row: Row, *, system_fold: bool,
                          include_assistant: bool) -> list[dict]:
    system = row.fields.get("system", "")
    user = row.fields["user"]
    messages: list[dict] = []
    if system:
        if system_fold:
            user = system + "\n\n" + user
        else:
            messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if include_assistant:
        messages.append({"role": "assistant", "content": row.fields["assistant"]})
    return messages


def _tokenize_instruction_row(row: Row, tokenizer, *, system_fold: bool,
                              max_tokens: int) -> TokenizedExample:
    prompt_ids = _as_id_list(tokenizer.apply_chat_template(
        _instruction_messages(row, system_fold=system_fold,
                              include_assistant=False),
        add_generation_prompt=True, tokenize=True))
    full_ids = _as_id_list(tokenizer.apply_chat_template(
        _instruction_messages(row, system_fold=system_fold,
                              include_assistant=True),
        add_generation_prompt=False, tokenize=True))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        # Without the prefix property the mask boundary is meaningless: the
        # positions we would call "assistant" are not the assistant's.
        raise LoRADataError(
            f"{row.path}:{row.line}: the chat template's prompt render is not "
            "a prefix of its full render — the assistant mask cannot be "
            "placed. Check the template (add_generation_prompt vs the "
            "assistant turn opener).")
    targets = len(full_ids) - len(prompt_ids)
    if targets <= 0:
        raise LoRADataError(
            f"{row.path}:{row.line}: zero assistant target tokens — the row "
            "would contribute nothing to the loss")
    if len(full_ids) > max_tokens:
        raise LoRADataError(
            f"{row.path}:{row.line}: renders to {len(full_ids)} tokens, over "
            f"the declared maxSequenceTokens {max_tokens} — shorten the row or "
            "raise the maximum (instruction rows are never split: a half "
            "answer is not the answer that was authored)")
    labels = [IGNORE_LABEL] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return TokenizedExample(input_ids=full_ids, labels=labels,
                            row_hash=row.row_hash, row_id=row.row_id)


def _document_windows(ids: list[int], *, max_tokens: int,
                      overlap: int) -> list[list[int]]:
    """Deterministic fixed windows with overlap. Every token appears in at
    least one window and window k always starts at ``k * (max - overlap)``, so
    the chunking is a pure function of (ids, max, overlap)."""
    stride = max_tokens - overlap
    windows: list[list[int]] = []
    start = 0
    while True:
        windows.append(ids[start:start + max_tokens])
        if start + max_tokens >= len(ids):
            return windows
        start += stride


def _tokenize_document_row(row: Row, tokenizer, *, spec: LoRADatasetSpec,
                           eos_token_id: int | None) -> list[TokenizedExample]:
    ids = _as_id_list(tokenizer(row.fields["text"], add_special_tokens=True))
    if not ids:
        raise LoRADataError(f"{row.path}:{row.line}: tokenizes to nothing")
    if eos_token_id is not None and ids[-1] != eos_token_id:
        # Every document example terminates: without it the model learns that
        # this text never ends, and the next row's opening becomes its
        # continuation at inference time.
        ids = ids + [eos_token_id]
    if len(ids) <= spec.max_sequence_tokens:
        return [TokenizedExample(input_ids=ids, labels=list(ids),
                                 row_hash=row.row_hash, row_id=row.row_id)]
    if spec.long_document_policy == REFUSE:
        raise LoRADataError(
            f"{row.path}:{row.line}: tokenizes to {len(ids)} tokens, over the "
            f"declared maxSequenceTokens {spec.max_sequence_tokens}, and "
            "longDocumentPolicy is 'refuse'")
    windows = _document_windows(ids, max_tokens=spec.max_sequence_tokens,
                                overlap=spec.chunk_overlap_tokens)
    return [TokenizedExample(input_ids=window, labels=list(window),
                             row_hash=row.row_hash, row_id=row.row_id,
                             chunk_index=index, chunk_count=len(windows),
                             chunk_overlap_tokens=spec.chunk_overlap_tokens)
            for index, window in enumerate(windows)]


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile — deterministic and integer-valued (no
    interpolation to argue about across engines)."""
    rank = max(1, math.ceil(fraction * len(values)))
    return sorted(values)[min(rank, len(values)) - 1]


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "p50": None,
                "p90": None, "total": 0}
    return {"min": min(values), "max": max(values),
            "mean": round(sum(values) / len(values), 3),
            "p50": _percentile(values, 0.5), "p90": _percentile(values, 0.9),
            "total": sum(values)}


def _split_stats(rows: tuple[Row, ...],
                 examples: list[TokenizedExample],
                 *, instruction: bool) -> dict:
    lengths = [len(example.input_ids) for example in examples]
    stats = {
        "accepted": len(rows),
        "examples": len(examples),
        # A refused row RAISES (plan §2.3.5) — these are structurally zero on
        # any successful load, and present so the key set never depends on the
        # data.
        "refused": 0,
        "truncated": 0,
        "chunked": sum(1 for example in examples if example.chunk_index == 0),
        "chunks": sum(1 for example in examples
                      if example.chunk_index is not None),
        "tokenLengths": _distribution(lengths),
    }
    if instruction:
        stats["assistantTargetTokens"] = _distribution(
            [example.target_token_count for example in examples])
    return stats


def tokenize_rows(loaded: LoadedRows, tokenizer, *, model_id: str,
                  spec: LoRADatasetSpec | None = None) -> TokenizedData:
    """Tokenize and label both splits, one row at a time.

    Document rows are NEVER concatenated: each is tokenized alone,
    EOS-terminated, and either kept whole or split into provenance-stamped
    windows. Instruction rows are rendered through the pinned chat template
    with everything before the assistant turn masked to ``-100``.

    ``spec`` defaults to ``loaded.spec`` (they must describe the same dataset;
    passing a different one is how the trainer would re-tokenize the same
    loaded rows under a different length policy).
    """
    spec = spec or loaded.spec
    instruction = spec.training_mode == INSTRUCTION_CHAT
    template: dict = {"chatTemplateHash": None, "systemFold": False,
                      "maskingPolicy": FULL_SEQUENCE}

    if instruction:
        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            raise LoRADataError(
                "instruction_chat training needs the tokenizer's chat "
                "template — refusing to invent one, because the mask boundary "
                "is defined by the template the model was trained with")
        template = {
            "chatTemplateHash": _sha256_text(str(chat_template)),
            "systemFold": system_fold_required(tokenizer, model_id),
            "maskingPolicy": ASSISTANT_ONLY,
        }
        eos_token_id = None
    else:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    split_examples: dict[str, list[TokenizedExample]] = {}
    for name, rows in (("train", loaded.train_rows),
                       ("validation", loaded.validation_rows)):
        examples: list[TokenizedExample] = []
        for row in rows:
            if instruction:
                examples.append(_tokenize_instruction_row(
                    row, tokenizer, system_fold=template["systemFold"],
                    max_tokens=spec.max_sequence_tokens))
            else:
                examples.extend(_tokenize_document_row(
                    row, tokenizer, spec=spec, eos_token_id=eos_token_id))
        split_examples[name] = examples

    stats = {
        "train": _split_stats(loaded.train_rows, split_examples["train"],
                              instruction=instruction),
        "validation": _split_stats(loaded.validation_rows,
                                   split_examples["validation"],
                                   instruction=instruction),
    }
    return TokenizedData(train=tuple(split_examples["train"]),
                         validation=tuple(split_examples["validation"]),
                         stats=stats, template=template)


def pad_batch(examples: list[TokenizedExample] | tuple[TokenizedExample, ...],
              *, pad_token_id: int) -> dict:
    """Right-pad a batch. Padding positions get label ``-100`` and attention
    ``0`` — padding NEVER contributes to the loss (plan §4). Kept here, beside
    the masking rules, so the trainer cannot re-derive it differently."""
    if not examples:
        raise LoRADataError("pad_batch: empty batch")
    width = max(len(example.input_ids) for example in examples)
    return {
        "input_ids": [e.input_ids + [pad_token_id] * (width - len(e.input_ids))
                      for e in examples],
        "labels": [e.labels + [IGNORE_LABEL] * (width - len(e.labels))
                   for e in examples],
        "attention_mask": [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids))
                           for e in examples],
    }


# --- provenance ------------------------------------------------------------


def dataset_manifest_dict(loaded: LoadedRows,
                          tokenized: TokenizedData | None = None) -> dict:
    """The dataset provenance block for the adapter sidecar (plan §2.8).

    ``trainingMode`` and ``template`` are returned alongside the ``dataset``
    keys; the sidecar writer lifts them to its own top level (contract §7).
    ``tokenized`` may be ``None`` for the plan endpoint, which must answer
    file hashes and row counts WITHOUT loading a tokenizer — the token-stat
    keys are then ``None`` rather than absent, so the key set is stable.
    """
    spec = loaded.spec
    return {
        "trainingMode": spec.training_mode,
        "bundleID": spec.bundle_id,
        "manifestPath": spec.manifest_path,
        "manifestHash": spec.manifest_hash,
        "evidenceGrade": loaded.evidence_grade,
        "maxSequenceTokens": spec.max_sequence_tokens,
        "longDocumentPolicy": spec.long_document_policy,
        "chunkOverlapTokens": spec.chunk_overlap_tokens,
        "trainFiles": [file.to_dict() for file in loaded.train_files],
        "validationFiles": [file.to_dict() for file in loaded.validation_files],
        "counts": loaded.counts,
        "tokenStats": tokenized.stats if tokenized is not None else None,
        "template": tokenized.template if tokenized is not None else None,
        # Filled by the caller from the dataset package when it declares
        # reserved/evaluation sets (plan §2.8); the key exists either way.
        "reservedEvaluationHashes": None,
    }


# --- control arm -----------------------------------------------------------

#: How many seeded shuffles to try before repairing the remaining fixed points
#: by hand. Both paths are deterministic; the retry only makes the common case
#: look like a plain shuffle.
_DERANGEMENT_ATTEMPTS = 4


def shuffle_assistant_pairing(rows: list[Row] | tuple[Row, ...],
                              seed: int) -> tuple[list[Row], float]:
    """Deterministically re-pair every prompt with ANOTHER row's assistant
    reply — the S0-analog control arm (plan §0.2): identical schedule,
    identical data volume, the construct's prompt→response mapping destroyed.

    Returns ``(rows, effective_change_fraction)``. The fraction is the share
    of rows whose assistant text ACTUALLY changed: a derangement of positions
    guarantees each row takes a different row's reply, but if two rows carried
    identical replies the text may be unchanged. A control whose fraction is
    low certifies nothing, which is exactly why the trainer stamps it
    (``shuffleEffectiveChangeFraction``) instead of asserting "shuffled".
    """
    rows = list(rows)
    if len(rows) < 2:
        raise LoRADataError(
            "shuffle_assistant_pairing needs at least 2 rows to re-pair")
    for row in rows:
        if "assistant" not in row.fields:
            raise LoRADataError(
                f"{row.path}:{row.line}: assistant-pairing shuffle is an "
                "instruction_chat control — this is a document row")

    rng = random.Random(seed)
    order = list(range(len(rows)))
    for _attempt in range(_DERANGEMENT_ATTEMPTS):
        rng.shuffle(order)
        if all(index != position for position, index in enumerate(order)):
            break
    else:
        # Repair the remaining fixed points in a fixed order: swapping a fixed
        # point with its neighbour can never create a new one (the neighbour
        # receives this position's index, which is not the neighbour's).
        for position in range(len(order)):
            if order[position] == position:
                neighbour = (position + 1) % len(order)
                order[position], order[neighbour] = \
                    order[neighbour], order[position]

    shuffled = [row.with_assistant(rows[source].fields["assistant"])
                for row, source in zip(rows, order)]
    changed = sum(1 for original, new in zip(rows, shuffled)
                  if original.fields["assistant"] != new.fields["assistant"])
    return shuffled, round(changed / len(rows), 6)
