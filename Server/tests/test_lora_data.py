"""Frozen LoRA dataset objects: strict row schemas, split integrity, the
assistant-only loss mask, deterministic document chunking, the control-arm
shuffle, and the ``data check lora`` package readiness verb.

Everything here is CPU-only and tokenizer-free except where the masking
arithmetic itself is under test — those tests use the hand-rolled
:class:`FakeGemmaTokenizer` below (deterministic ids, a Gemma-like template
that REJECTS the system role, exactly like the real one). That is deliberate:
the property being proved is "the mask starts where the prompt render ends",
which is a statement about the two renders, not about any particular
vocabulary. Golden parity against a real tokenizer lives in
``test_golden_tokens.py``.
"""

import hashlib
import json
import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import data_readiness, lora_data
from steerlab_server.experiment.lora_data import (
    DatasetFile, LoRADataError, LoRADatasetSpec)


# --- fake tokenizers -------------------------------------------------------


def _token_id(token: str) -> int:
    """Stable across processes (unlike hash() on str, which is salted)."""
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:6], 16) % 50000 + 100


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeGemmaTokenizer:
    """A Gemma-shaped chat template over a whitespace vocabulary.

    Renders ``<bos> <start_of_turn> <role> …words… <end_of_turn>`` per turn and
    appends ``<start_of_turn> model`` for a generation prompt — so the
    prompt render is a genuine prefix of the full render, as the real template
    is. ``supports_system=False`` (the default) raises on a system message,
    which is how the real Gemma template announces it has no system role.
    """

    chat_template = "{# fake gemma-like template #}{{ messages }}"
    eos_token_id = _token_id("<end_of_turn>")
    pad_token_id = 0

    def __init__(self, supports_system: bool = False):
        self.supports_system = supports_system

    def __call__(self, text, add_special_tokens=True):
        tokens = (["<bos>"] if add_special_tokens else []) + text.split()
        return _Encoding([_token_id(token) for token in tokens])

    def render(self, messages, add_generation_prompt):
        parts = ["<bos>"]
        for message in messages:
            role = message["role"]
            if role == "system" and not self.supports_system:
                raise ValueError("this template has no system role")
            parts.append("<start_of_turn>")
            parts.append("model" if role == "assistant" else role)
            parts.extend(message["content"].split())
            parts.append("<end_of_turn>")
        if add_generation_prompt:
            parts += ["<start_of_turn>", "model"]
        return " ".join(parts)

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True):
        text = self.render(messages, add_generation_prompt)
        return [_token_id(token) for token in text.split()] if tokenize else text


class ZeroTargetTokenizer(FakeGemmaTokenizer):
    """A degenerate template that renders the assistant turn to nothing — the
    row would contribute no supervised token."""

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True):
        kept = [m for m in messages if m["role"] != "assistant"]
        return super().apply_chat_template(kept, add_generation_prompt=True,
                                           tokenize=tokenize)


class NonPrefixTokenizer(FakeGemmaTokenizer):
    """A template whose full render is not an extension of its prompt render
    (a real hazard: templates that re-open the conversation differently once
    an assistant turn exists)."""

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True):
        ids = super().apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt,
            tokenize=tokenize)
        if any(m["role"] == "assistant" for m in messages) and tokenize:
            return [_token_id("<drift>")] + list(ids)
        return ids


class NoTemplateTokenizer(FakeGemmaTokenizer):
    chat_template = None


# --- row / file helpers ----------------------------------------------------


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _document_spec(tmp_path, train_rows, validation_rows, *, pin=True,
                   max_tokens=512, policy="split", overlap=64):
    train = str(tmp_path / "train.jsonl")
    train_hash = _write_jsonl(train, train_rows)
    validation_files = []
    if validation_rows is not None:
        validation = str(tmp_path / "validation.jsonl")
        validation_hash = _write_jsonl(validation, validation_rows)
        validation_files = [DatasetFile(validation,
                                        validation_hash if pin else None,
                                        "validation")]
    return LoRADatasetSpec(
        training_mode="document",
        train_files=[DatasetFile(train, train_hash if pin else None, "train")],
        validation_files=validation_files, max_sequence_tokens=max_tokens,
        long_document_policy=policy, chunk_overlap_tokens=overlap)


def _instruction_spec(tmp_path, train_rows, validation_rows, *, max_tokens=512):
    train = str(tmp_path / "train.jsonl")
    train_hash = _write_jsonl(train, train_rows)
    validation = str(tmp_path / "validation.jsonl")
    validation_hash = _write_jsonl(validation, validation_rows)
    return LoRADatasetSpec(
        training_mode="instruction_chat",
        train_files=[DatasetFile(train, train_hash, "train")],
        validation_files=[DatasetFile(validation, validation_hash, "validation")],
        max_sequence_tokens=max_tokens,
        # Instruction rows are never chunked; keep the overlap out of the way
        # of the spec's own bounds check at small maxima.
        chunk_overlap_tokens=0)


def _row(fields, path="mem.jsonl", line=1):
    return lora_data.Row(path=path, line=line, fields=fields)


# --- strict schemas: document rows -----------------------------------------


def test_document_unknown_key_refuses_with_path_and_line():
    with pytest.raises(LoRADataError) as exc:
        lora_data.parse_rows('{"text": "ok"}\n{"text": "x", "label": "a"}\n',
                             path="adapters/x/train.jsonl",
                             training_mode="document")
    assert "adapters/x/train.jsonl:2" in str(exc.value)
    assert "label" in str(exc.value)


def test_document_empty_and_non_string_text_refuse():
    for payload, needle in (('{"text": "   "}', "empty"),
                            ('{"text": 3}', "JSON string"),
                            ('{"id": "a"}', "no 'text'")):
        with pytest.raises(LoRADataError) as exc:
            lora_data.parse_rows(payload, path="f.jsonl",
                                 training_mode="document")
        assert needle in str(exc.value) and "f.jsonl:1" in str(exc.value)


def test_document_non_object_and_bad_json_refuse():
    with pytest.raises(LoRADataError) as exc:
        lora_data.parse_rows('["text"]', path="f.jsonl", training_mode="document")
    assert "JSON object" in str(exc.value)
    with pytest.raises(LoRADataError) as exc:
        lora_data.parse_rows('{"text": ', path="f.jsonl", training_mode="document")
    assert "invalid JSON" in str(exc.value)


# --- strict schemas: instruction rows --------------------------------------


def test_instruction_aliases_normalize_like_swift():
    rows = lora_data.parse_rows(
        '{"prompt": "  Decide the case.  ", "completion": " It is affirmed. "}',
        path="f.jsonl", training_mode="instruction_chat")
    # prompt→user, completion→assistant, both trimmed (FineTuneTrainer.swift:385-392).
    assert rows[0].fields == {"user": "Decide the case.",
                              "assistant": "It is affirmed."}
    # Canonical spellings only — the persisted form never carries an alias.
    assert "prompt" not in rows[0].fields and "completion" not in rows[0].fields


def test_instruction_both_alias_spellings_refuse():
    for payload, needle in (
            ('{"user": "u", "prompt": "u", "assistant": "a"}', "prompt"),
            ('{"user": "u", "assistant": "a", "completion": "a"}', "completion")):
        with pytest.raises(LoRADataError) as exc:
            lora_data.parse_rows(payload, path="f.jsonl",
                                 training_mode="instruction_chat")
        assert needle in str(exc.value) and "one spelling" in str(exc.value)


def test_instruction_empty_user_or_assistant_refuses():
    for payload, key in (('{"user": " ", "assistant": "a"}', "user"),
                         ('{"user": "u", "assistant": ""}', "assistant"),
                         ('{"user": "u", "assistant": "a", "system": " "}',
                          "system")):
        with pytest.raises(LoRADataError) as exc:
            lora_data.parse_rows(payload, path="f.jsonl",
                                 training_mode="instruction_chat")
        assert f"'{key}' is empty" in str(exc.value)


def test_instruction_missing_field_and_unknown_key_refuse():
    with pytest.raises(LoRADataError) as exc:
        lora_data.parse_rows('{"user": "u"}', path="f.jsonl",
                             training_mode="instruction_chat")
    assert "no 'assistant'" in str(exc.value)
    with pytest.raises(LoRADataError) as exc:
        lora_data.parse_rows('{"user": "u", "assistant": "a", "text": "t"}',
                             path="f.jsonl", training_mode="instruction_chat")
    assert "unknown key(s) text" in str(exc.value)


# --- hashes, roots, split integrity ----------------------------------------


def test_row_hash_and_rows_root_are_the_documented_functions(tmp_path):
    spec = _document_spec(tmp_path, [{"text": "alpha"}, {"text": "beta"}],
                          [{"text": "gamma"}])
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    expected = [
        hashlib.sha256(json.dumps({"text": t}, sort_keys=True,
                                  ensure_ascii=False,
                                  separators=(",", ":")).encode()).hexdigest()
        for t in ("alpha", "beta")]
    assert [row.row_hash for row in loaded.train_files[0].rows] == expected
    assert loaded.train_files[0].rows_root == hashlib.sha256(
        "\n".join(expected).encode()).hexdigest()


def test_rows_root_is_order_sensitive_and_deterministic(tmp_path):
    forward = _document_spec(tmp_path / "a", [{"text": "one"}, {"text": "two"}],
                             [{"text": "held"}])
    reverse = _document_spec(tmp_path / "b", [{"text": "two"}, {"text": "one"}],
                             [{"text": "held"}])
    again = _document_spec(tmp_path / "c", [{"text": "one"}, {"text": "two"}],
                           [{"text": "held"}])
    root = lambda spec: lora_data.load_split_rows(  # noqa: E731
        spec, evidence_grade=True).train_files[0].rows_root
    assert root(forward) == root(again)
    assert root(forward) != root(reverse)


def test_file_hash_mismatch_refuses_naming_both_hashes(tmp_path):
    spec = _document_spec(tmp_path, [{"text": "alpha"}], [{"text": "held"}])
    _write_jsonl(spec.train_files[0].path, [{"text": "edited after pinning"}])
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "sha256 mismatch" in str(exc.value)
    assert spec.train_files[0].expected_sha256 in str(exc.value)


def test_unpinned_file_refuses_only_for_evidence_grade(tmp_path):
    spec = _document_spec(tmp_path, [{"text": "alpha"}], [{"text": "held"}],
                          pin=False)
    assert lora_data.load_split_rows(spec, evidence_grade=False).counts[
        "trainRows"] == 1
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "pinned sha256" in str(exc.value)


def test_missing_validation_refuses_for_evidence_grade(tmp_path):
    spec = _document_spec(tmp_path, [{"text": "alpha"}], None)
    assert lora_data.load_split_rows(spec, evidence_grade=False).counts[
        "validationRows"] == 0
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "explicit validation split" in str(exc.value)


def test_intra_split_duplicate_row_refuses_naming_both_lines(tmp_path):
    spec = _document_spec(tmp_path,
                          [{"text": "same"}, {"text": "other"}, {"text": "same"}],
                          [{"text": "held"}])
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert ":3: duplicate row" in str(exc.value) and ":1)" in str(exc.value)


def test_cross_split_row_hash_overlap_refuses(tmp_path):
    spec = _document_spec(tmp_path, [{"text": "shared passage"}],
                          [{"text": "shared passage"}])
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "also present in the training split" in str(exc.value)


def test_cross_split_same_text_under_a_different_id_still_refuses(tmp_path):
    # Different rowHash (the id is part of it) — the CONTENT check is what
    # catches this, and content is what leaks.
    spec = _document_spec(tmp_path, [{"id": "t-1", "text": "shared passage"}],
                          [{"id": "v-1", "text": "shared passage"}])
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "also present in the training split" in str(exc.value)


def test_instruction_cross_split_overlap_refuses(tmp_path):
    pair = {"user": "Decide.", "assistant": "Affirmed."}
    spec = _instruction_spec(tmp_path, [pair], [dict(pair)])
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "also present in the training split" in str(exc.value)


def test_same_file_in_both_splits_refuses(tmp_path):
    path = str(tmp_path / "train.jsonl")
    digest = _write_jsonl(path, [{"text": "alpha"}])
    spec = LoRADatasetSpec(training_mode="document",
                           train_files=[DatasetFile(path, digest, "train")],
                           validation_files=[DatasetFile(path, digest,
                                                         "validation")],
                           max_sequence_tokens=512)
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=True)
    assert "declared twice" in str(exc.value)


def test_missing_file_and_empty_file_refuse(tmp_path):
    spec = _document_spec(tmp_path, [{"text": "alpha"}], [{"text": "held"}])
    os.remove(spec.train_files[0].path)
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(spec, evidence_grade=False)
    assert "not found" in str(exc.value)

    empty = _document_spec(tmp_path / "e", [{"text": "alpha"}], [{"text": "h"}])
    _write_jsonl(empty.train_files[0].path, [])
    unpinned = LoRADatasetSpec(
        training_mode="document",
        train_files=[DatasetFile(empty.train_files[0].path, None, "train")],
        validation_files=list(empty.validation_files), max_sequence_tokens=512)
    with pytest.raises(LoRADataError) as exc:
        lora_data.load_split_rows(unpinned, evidence_grade=False)
    assert "no rows" in str(exc.value)


def test_spec_rejects_impossible_policies(tmp_path):
    with pytest.raises(LoRADataError):
        LoRADatasetSpec(training_mode="documents", train_files=[],
                        validation_files=[], max_sequence_tokens=8)
    with pytest.raises(LoRADataError):
        LoRADatasetSpec(training_mode="document", train_files=[],
                        validation_files=[], max_sequence_tokens=8,
                        long_document_policy="truncate")
    with pytest.raises(LoRADataError) as exc:
        LoRADatasetSpec(training_mode="document", train_files=[],
                        validation_files=[], max_sequence_tokens=8,
                        chunk_overlap_tokens=8)
    assert "smaller than maxSequenceTokens" in str(exc.value)


# --- document tokenization -------------------------------------------------


def test_document_rows_are_tokenized_independently_and_terminated(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    spec = _document_spec(tmp_path, [{"text": "alpha beta"}, {"text": "gamma"}],
                          [{"text": "held out"}])
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    data = lora_data.tokenize_rows(loaded, tokenizer, model_id="google/gemma-3-27b-it")

    assert len(data.train) == 2                     # one example per row
    first, second = data.train
    assert first.input_ids == [_token_id(t) for t in
                               ("<bos>", "alpha", "beta", "<end_of_turn>")]
    assert second.input_ids == [_token_id(t) for t in
                                ("<bos>", "gamma", "<end_of_turn>")]
    # No example spans two rows: every example's ids are exactly one row's.
    assert all(example.input_ids[0] == _token_id("<bos>")
               for example in data.train)
    assert all(example.input_ids[-1] == tokenizer.eos_token_id
               for example in data.train)
    # Full-sequence labels, and the row it came from is stamped.
    assert first.labels == first.input_ids
    assert {e.row_hash for e in data.train} == {
        row.row_hash for row in loaded.train_rows}
    assert data.template == {"chatTemplateHash": None, "systemFold": False,
                             "maskingPolicy": "fullSequence"}


def test_two_examples_never_share_a_row_and_two_rows_never_share_an_example(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    spec = _document_spec(tmp_path,
                          [{"text": " ".join(f"a{i}" for i in range(20))},
                           {"text": " ".join(f"b{i}" for i in range(20))}],
                          [{"text": "held"}], max_tokens=8, overlap=2)
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    data = lora_data.tokenize_rows(loaded, tokenizer, model_id="gemma")

    a_ids = {_token_id(f"a{i}") for i in range(20)}
    b_ids = {_token_id(f"b{i}") for i in range(20)}
    for example in data.train:
        touched = set(example.input_ids)
        assert not (touched & a_ids and touched & b_ids), \
            "a chunk spans two source rows — document boundaries crossed"


def test_long_document_split_is_deterministic_with_chunk_provenance(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    text = " ".join(f"w{i}" for i in range(30))
    spec = _document_spec(tmp_path, [{"id": "doc-1", "text": text}],
                          [{"text": "held"}], max_tokens=8, overlap=2)
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    data = lora_data.tokenize_rows(loaded, tokenizer, model_id="gemma")
    again = lora_data.tokenize_rows(loaded, tokenizer, model_id="gemma")

    assert [e.input_ids for e in data.train] == [e.input_ids for e in again.train]
    assert len(data.train) > 1
    for index, example in enumerate(data.train):
        assert example.chunk_index == index
        assert example.chunk_count == len(data.train)
        assert example.chunk_overlap_tokens == 2
        assert example.row_id == "doc-1"
        assert example.row_hash == loaded.train_rows[0].row_hash
        assert len(example.input_ids) <= 8
    # Consecutive windows overlap by exactly the declared amount.
    for previous, following in zip(data.train, data.train[1:]):
        assert previous.input_ids[-2:] == following.input_ids[:2]
    # Every source token survives, in order, and the last chunk terminates.
    assert data.train[-1].input_ids[-1] == tokenizer.eos_token_id
    assert data.train[0].provenance()["chunkIndex"] == 0
    assert data.stats["train"]["chunked"] == 1
    assert data.stats["train"]["examples"] == len(data.train)


def test_long_document_refuse_policy_refuses_naming_the_row(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    spec = _document_spec(tmp_path,
                          [{"text": "short"},
                           {"text": " ".join(f"w{i}" for i in range(30))}],
                          [{"text": "held"}], max_tokens=8, policy="refuse",
                          overlap=2)
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    with pytest.raises(LoRADataError) as exc:
        lora_data.tokenize_rows(loaded, tokenizer, model_id="gemma")
    assert ":2:" in str(exc.value) and "longDocumentPolicy is 'refuse'" in str(exc.value)


def test_document_token_length_distribution(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    rows = [{"text": " ".join(["w"] * n)} for n in (1, 2, 3, 8)]
    # distinct texts (w, w w, …) so no duplicate-row refusal
    spec = _document_spec(tmp_path, rows, [{"text": "held"}])
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    stats = lora_data.tokenize_rows(loaded, tokenizer,
                                    model_id="gemma").stats["train"]
    # each example is <bos> + n words + <end_of_turn>
    assert stats["tokenLengths"]["min"] == 3
    assert stats["tokenLengths"]["max"] == 10
    assert stats["tokenLengths"]["total"] == 3 + 4 + 5 + 10
    assert stats["tokenLengths"]["p50"] == 4 and stats["tokenLengths"]["p90"] == 10
    assert stats["truncated"] == 0 and stats["refused"] == 0
    assert stats["accepted"] == 4 and stats["chunked"] == 0


# --- instruction masking ---------------------------------------------------


INSTRUCTION_ROW = {"user": "Decide the appeal.", "assistant": "The order is affirmed."}


def _tokenize_one(tmp_path, row, tokenizer=None, model_id="google/gemma-3-27b-it",
                  max_tokens=512):
    tokenizer = tokenizer or FakeGemmaTokenizer()
    spec = _instruction_spec(tmp_path, [row],
                            [{"user": "Other.", "assistant": "Reversed."}],
                            max_tokens=max_tokens)
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    data = lora_data.tokenize_rows(loaded, tokenizer, model_id=model_id)
    return data, data.train[0]


def test_exact_assistant_mask_fixture(tmp_path):
    """The whole point of the mode: the first and last supervised positions.

    prompt render = <bos> <sot> user Decide the appeal. <eot> <sot> model
                  = 9 tokens (positions 0-8)
    full render   = … <sot> model The order is affirmed. <eot>
                  = 14 tokens (positions 0-13)
    so labels 0-8 are -100 and 9-13 are the assistant's own ids.
    """
    data, example = _tokenize_one(tmp_path, INSTRUCTION_ROW)
    prompt_tokens = ["<bos>", "<start_of_turn>", "user", "Decide", "the",
                     "appeal.", "<end_of_turn>", "<start_of_turn>", "model"]
    assert len(prompt_tokens) == 9
    assert example.input_ids[:9] == [_token_id(t) for t in prompt_tokens]
    assert len(example.input_ids) == 14

    supervised = [i for i, label in enumerate(example.labels)
                  if label != lora_data.IGNORE_LABEL]
    assert supervised[0] == 9 and supervised[-1] == 13
    assert supervised == list(range(9, 14))
    assert example.labels[:9] == [lora_data.IGNORE_LABEL] * 9
    assert example.labels[9:] == example.input_ids[9:]
    # the assistant terminator IS supervised (plan §2.3.4)
    assert example.input_ids[-1] == _token_id("<end_of_turn>")
    assert example.target_token_count == 5
    assert data.template["maskingPolicy"] == "assistantOnly"
    assert data.template["systemFold"] is True
    assert data.template["chatTemplateHash"] == hashlib.sha256(
        FakeGemmaTokenizer.chat_template.encode()).hexdigest()
    assert data.stats["train"]["assistantTargetTokens"]["max"] == 5


def test_changing_only_the_prompt_does_not_move_the_mask(tmp_path):
    """Plan §2.3: prompt tokens never enter the loss, however many there are.
    The supervised SUFFIX is byte-identical; only its offset moves, and it
    moves exactly with the prompt length."""
    short = _tokenize_one(tmp_path / "a", INSTRUCTION_ROW)[1]
    longer = _tokenize_one(
        tmp_path / "b",
        {"user": "Decide the appeal, considering the whole record below.",
         "assistant": INSTRUCTION_ROW["assistant"]})[1]

    def supervised(example):
        return [i for i, label in enumerate(example.labels)
                if label != lora_data.IGNORE_LABEL]

    assert len(supervised(short)) == len(supervised(longer))
    # same supervised ids, in the same order
    assert [short.input_ids[i] for i in supervised(short)] == \
        [longer.input_ids[i] for i in supervised(longer)]
    # and the boundary sits exactly at each render's own prompt length
    assert supervised(short)[0] == len(short.input_ids) - short.target_token_count
    assert supervised(longer)[0] == len(longer.input_ids) - longer.target_token_count
    assert supervised(longer)[0] > supervised(short)[0]


def test_zero_assistant_target_refuses(tmp_path):
    with pytest.raises(LoRADataError) as exc:
        _tokenize_one(tmp_path, INSTRUCTION_ROW, tokenizer=ZeroTargetTokenizer())
    assert "zero assistant target tokens" in str(exc.value)


def test_over_length_instruction_row_refuses_instead_of_truncating(tmp_path):
    with pytest.raises(LoRADataError) as exc:
        _tokenize_one(tmp_path, INSTRUCTION_ROW, max_tokens=8)
    assert "over the declared maxSequenceTokens 8" in str(exc.value)
    assert "train.jsonl:1" in str(exc.value)


def test_template_without_prefix_property_refuses(tmp_path):
    with pytest.raises(LoRADataError) as exc:
        _tokenize_one(tmp_path, INSTRUCTION_ROW, tokenizer=NonPrefixTokenizer())
    assert "not a prefix" in str(exc.value)


def test_instruction_mode_without_a_chat_template_refuses(tmp_path):
    with pytest.raises(LoRADataError) as exc:
        _tokenize_one(tmp_path, INSTRUCTION_ROW, tokenizer=NoTemplateTokenizer())
    assert "chat template" in str(exc.value)


def test_gemma_system_fold_puts_system_text_in_the_user_turn(tmp_path):
    row = {"system": "You are a judge.", "user": "Decide.",
           "assistant": "Affirmed."}
    _data, example = _tokenize_one(tmp_path, row)
    rendered = [_token_id(t) for t in
                ("<bos>", "<start_of_turn>", "user", "You", "are", "a",
                 "judge.", "Decide.", "<end_of_turn>", "<start_of_turn>",
                 "model", "Affirmed.", "<end_of_turn>")]
    assert example.input_ids == rendered      # no system turn was emitted
    assert _token_id("system") not in example.input_ids


def test_system_role_is_kept_when_the_template_supports_it(tmp_path):
    row = {"system": "You are a judge.", "user": "Decide.",
           "assistant": "Affirmed."}
    data, example = _tokenize_one(tmp_path, row,
                                  tokenizer=FakeGemmaTokenizer(supports_system=True),
                                  model_id="Qwen/Qwen3-32B")
    assert data.template["systemFold"] is False
    assert _token_id("system") in example.input_ids


def test_non_gemma_model_with_a_system_rejecting_template_still_folds(tmp_path):
    """The probe, not just the model-id substring: a template that raises on
    the system role folds even when the id says nothing about Gemma."""
    data, example = _tokenize_one(
        tmp_path, {"system": "S", "user": "u", "assistant": "a"},
        tokenizer=FakeGemmaTokenizer(supports_system=False),
        model_id="some-org/mystery-model")
    assert data.template["systemFold"] is True
    assert _token_id("system") not in example.input_ids


def test_padding_never_contributes_to_the_loss(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    spec = _instruction_spec(
        tmp_path,
        [INSTRUCTION_ROW,
         {"user": "Short.", "assistant": "No."}],
        [{"user": "Held.", "assistant": "Out."}])
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    data = lora_data.tokenize_rows(loaded, tokenizer, model_id="gemma")
    batch = lora_data.pad_batch(list(data.train), pad_token_id=0)

    width = max(len(e.input_ids) for e in data.train)
    assert {len(row) for row in batch["input_ids"]} == {width}
    for example, ids, labels, mask in zip(data.train, batch["input_ids"],
                                          batch["labels"], batch["attention_mask"]):
        pad_positions = range(len(example.input_ids), width)
        assert all(labels[i] == lora_data.IGNORE_LABEL for i in pad_positions)
        assert all(mask[i] == 0 for i in pad_positions)
        assert all(ids[i] == 0 for i in pad_positions)
        # the real tokens are untouched
        assert labels[:len(example.labels)] == example.labels
    # supervised token count is unchanged by padding
    assert sum(1 for row in batch["labels"] for label in row
               if label != lora_data.IGNORE_LABEL) == \
        sum(e.target_token_count for e in data.train)


# --- provenance ------------------------------------------------------------


def test_dataset_manifest_dict_shape(tmp_path):
    tokenizer = FakeGemmaTokenizer()
    spec = _document_spec(tmp_path, [{"text": "alpha"}], [{"text": "held"}])
    spec = LoRADatasetSpec(
        training_mode="document", train_files=list(spec.train_files),
        validation_files=list(spec.validation_files), max_sequence_tokens=512,
        bundle_id="stance-lora-family-v1",
        manifest_path="adapters/stance-lora-family-v1-manifest.json",
        manifest_hash="deadbeef")
    loaded = lora_data.load_split_rows(spec, evidence_grade=True)
    tokenized = lora_data.tokenize_rows(loaded, tokenizer, model_id="gemma")

    block = lora_data.dataset_manifest_dict(loaded, tokenized)
    assert block["bundleID"] == "stance-lora-family-v1"
    assert block["manifestHash"] == "deadbeef"
    assert block["trainingMode"] == "document"
    assert block["evidenceGrade"] is True
    assert block["counts"] == {"trainFiles": 1, "validationFiles": 1,
                               "trainRows": 1, "validationRows": 1}
    entry = block["trainFiles"][0]
    assert set(entry) == {"path", "sha256", "rows", "rowsRoot"}
    assert entry["sha256"] == spec.train_files[0].expected_sha256
    assert entry["rowsRoot"] == loaded.train_files[0].rows_root
    assert block["tokenStats"]["train"]["accepted"] == 1
    assert block["template"] == tokenized.template
    assert block["reservedEvaluationHashes"] is None
    # The plan endpoint answers without a tokenizer; the key set is stable.
    plan_block = lora_data.dataset_manifest_dict(loaded)
    assert plan_block.keys() == block.keys()
    assert plan_block["tokenStats"] is None and plan_block["template"] is None


# --- control arm -----------------------------------------------------------


def _instruction_rows(count):
    return [_row({"user": f"Prompt {i}", "assistant": f"Reply {i}"},
                 path="train.jsonl", line=i + 1) for i in range(count)]


def test_shuffle_assistant_pairing_is_a_deterministic_derangement():
    rows = _instruction_rows(8)
    first, fraction = lora_data.shuffle_assistant_pairing(rows, seed=7)
    again, fraction_again = lora_data.shuffle_assistant_pairing(rows, seed=7)

    assert [r.fields for r in first] == [r.fields for r in again]
    assert fraction == fraction_again == 1.0
    # every row took SOMEBODY ELSE's reply, and the user text is untouched
    for original, shuffled in zip(rows, first):
        assert shuffled.fields["assistant"] != original.fields["assistant"]
        assert shuffled.fields["user"] == original.fields["user"]
    # the multiset of replies is preserved (identical data volume, plan §0.2)
    assert sorted(r.fields["assistant"] for r in first) == \
        sorted(r.fields["assistant"] for r in rows)
    # row hashes are recomputed, so the control's rows root cannot collide
    assert {r.row_hash for r in first}.isdisjoint({r.row_hash for r in rows})


def test_shuffle_with_a_different_seed_differs():
    rows = _instruction_rows(8)
    a, _ = lora_data.shuffle_assistant_pairing(rows, seed=1)
    b, _ = lora_data.shuffle_assistant_pairing(rows, seed=2)
    assert [r.fields for r in a] != [r.fields for r in b]


def test_degenerate_shuffle_reports_zero_change_fraction():
    """A "shuffle" of identical replies changes nothing — the fraction is the
    measure that makes that visible on the artifact's face (plan §0.2)."""
    rows = [_row({"user": f"Prompt {i}", "assistant": "Same reply"}, line=i + 1)
            for i in range(6)]
    shuffled, fraction = lora_data.shuffle_assistant_pairing(rows, seed=3)
    assert fraction == 0.0
    assert all(r.fields["assistant"] == "Same reply" for r in shuffled)


def test_partial_change_fraction_counts_rows_that_actually_changed():
    rows = [_row({"user": "a", "assistant": "X"}, line=1),
            _row({"user": "b", "assistant": "X"}, line=2),
            _row({"user": "c", "assistant": "Y"}, line=3),
            _row({"user": "d", "assistant": "Z"}, line=4)]
    shuffled, fraction = lora_data.shuffle_assistant_pairing(rows, seed=11)
    changed = sum(1 for o, s in zip(rows, shuffled)
                  if o.fields["assistant"] != s.fields["assistant"])
    assert fraction == round(changed / 4, 6)
    assert 0.0 < fraction <= 1.0


def test_shuffle_refuses_document_rows_and_singletons():
    with pytest.raises(LoRADataError) as exc:
        lora_data.shuffle_assistant_pairing(
            [_row({"text": "a"}, line=1), _row({"text": "b"}, line=2)], seed=0)
    assert "document row" in str(exc.value)
    with pytest.raises(LoRADataError) as exc:
        lora_data.shuffle_assistant_pairing(_instruction_rows(1), seed=0)
    assert "at least 2 rows" in str(exc.value)


# --- data check lora -------------------------------------------------------


def _package(root, *, mode="document", nested=False, qc_status="pass",
             train_rows=None, validation_rows=None, arm="demo-lora-v1"):
    """A minimal but real-shaped dataset package under ``<root>/adapters``."""
    if mode == "document":
        train_rows = train_rows or [{"text": f"training passage {i}"}
                                    for i in range(4)]
        validation_rows = validation_rows or [{"text": "held-out passage"}]
    else:
        train_rows = train_rows or [{"user": f"Prompt {i}",
                                     "assistant": f"Reply {i}"}
                                    for i in range(4)]
        validation_rows = validation_rows or [{"user": "Held", "assistant": "Out"}]

    adapters = os.path.join(root, "adapters")
    train_rel = f"adapters/{arm}/training/train.jsonl"
    validation_rel = f"adapters/{arm}/validation/validation.jsonl"
    train_hash = _write_jsonl(os.path.join(root, train_rel), train_rows)
    validation_hash = _write_jsonl(os.path.join(root, validation_rel),
                                   validation_rows)
    entry = {
        "adapter": arm,
        "training": {"path": train_rel, "sha256": train_hash,
                     "rows": len(train_rows)},
        "validation": {"path": validation_rel, "sha256": validation_hash,
                       "rows": len(validation_rows)},
    }
    manifest = {
        "dataset": "demo-lora-family-v1",
        "schemaVersion": 1,
        "trainingMode": mode,
        "qcReport": "adapters/demo-lora-family-v1-qc.json",
    }
    # Both manifest spellings the builders emit.
    if nested:
        manifest["families"] = {"demo": {"arms": {"treatment": entry}}}
    else:
        manifest["outputs"] = {"demo": entry}

    os.makedirs(adapters, exist_ok=True)
    qc_path = os.path.join(root, "adapters", "demo-lora-family-v1-qc.json")
    with open(qc_path, "w", encoding="utf-8") as handle:
        json.dump({"status": qc_status, "errors": [] if qc_status == "pass"
                   else ["exact overlap with reserved data"], "warnings": []},
                  handle)
    manifest_path = os.path.join(adapters, "demo-lora-family-v1-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def _edit_manifest(manifest_path, mutate):
    with open(manifest_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    mutate(payload)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _by_name(report):
    return {r.name: r for r in report.requirements}


def test_check_lora_package_passes_for_both_manifest_spellings(tmp_path):
    for nested in (False, True):
        root = tmp_path / ("nested" if nested else "flat")
        manifest = _package(str(root), nested=nested)
        report = data_readiness.check_lora_package(manifest, root=str(root))
        assert report.ready, [r.detail for r in report.blockers]
        names = _by_name(report)
        assert names["demo-lora-v1 split"].status == "present"
        assert names["adapters/demo-lora-v1/training/train.jsonl"].rows == 4
        assert len(names["adapters/demo-lora-v1/training/train.jsonl"].sha256) == 64
        assert report.authoring_spec == data_readiness.LORA_AUTHORING_SPEC
        assert report.to_dict()["manifestPath"] == manifest


def test_check_lora_package_instruction_mode(tmp_path):
    manifest = _package(str(tmp_path), mode="instruction_chat")
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert report.ready
    assert "instruction_chat row(s)" in \
        _by_name(report)["adapters/demo-lora-v1/training/train.jsonl"].detail


def test_check_lora_missing_file_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    os.remove(os.path.join(str(tmp_path),
                           "adapters/demo-lora-v1/validation/validation.jsonl"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)[
        "adapters/demo-lora-v1/validation/validation.jsonl"]
    assert requirement.status == "missing" and not report.ready
    # the split line is reported but not double-counted as a blocker
    assert _by_name(report)["demo-lora-v1 split"].status == "partial"


def test_check_lora_hash_mismatch_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    _write_jsonl(os.path.join(str(tmp_path),
                              "adapters/demo-lora-v1/training/train.jsonl"),
                 [{"text": "edited after the manifest pinned it"}])
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)["adapters/demo-lora-v1/training/train.jsonl"]
    assert requirement.status == "invalid" and "sha256 mismatch" in requirement.detail


def test_check_lora_bad_row_schema_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    train_rel = "adapters/demo-lora-v1/training/train.jsonl"
    digest = _write_jsonl(os.path.join(str(tmp_path), train_rel),
                          [{"text": "fine"}, {"text": "bad", "weight": 2}])
    _edit_manifest(manifest, lambda p: p["outputs"]["demo"]["training"].update(
        sha256=digest, rows=2))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)[train_rel]
    assert requirement.status == "invalid"
    assert "unknown key(s) weight" in requirement.detail and ":2:" in requirement.detail


def test_check_lora_row_count_disagreement_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    _edit_manifest(manifest,
                   lambda p: p["outputs"]["demo"]["training"].update(rows=99))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)["adapters/demo-lora-v1/training/train.jsonl"]
    assert requirement.status == "invalid" and "declares 99 row(s)" in requirement.detail


def test_check_lora_cross_split_leakage_blocks_on_the_split_line(tmp_path):
    manifest = _package(str(tmp_path))
    leaked = {"text": "training passage 0"}      # already in train
    digest = _write_jsonl(
        os.path.join(str(tmp_path),
                     "adapters/demo-lora-v1/validation/validation.jsonl"),
        [leaked])
    _edit_manifest(manifest, lambda p: p["outputs"]["demo"]["validation"].update(
        sha256=digest, rows=1))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    split = _by_name(report)["demo-lora-v1 split"]
    assert split.status == "invalid" and not report.ready
    assert "also present in the training split" in split.detail


def test_check_lora_missing_validation_block_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    _edit_manifest(manifest, lambda p: p["outputs"]["demo"].pop("validation"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)["demo-lora-v1 validation"]
    assert requirement.status == "missing"
    assert "fractional split" in requirement.detail


def test_check_lora_unpinned_file_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    _edit_manifest(manifest,
                   lambda p: p["outputs"]["demo"]["training"].pop("sha256"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)["adapters/demo-lora-v1/training/train.jsonl"]
    assert requirement.status == "invalid" and "pins no sha256" in requirement.detail

    # A block with no 'path' at all is named by the arm, since there is no
    # path to name it with.
    _edit_manifest(manifest,
                   lambda p: p["outputs"]["demo"]["training"].pop("path"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert _by_name(report)["demo-lora-v1 training"].status == "invalid"


def test_check_lora_failed_qc_blocks(tmp_path):
    manifest = _package(str(tmp_path), qc_status="fail")
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    requirement = _by_name(report)["adapters/demo-lora-family-v1-qc.json"]
    assert requirement.status == "invalid"
    assert "not 'pass'" in requirement.detail
    assert "exact overlap with reserved data" in requirement.detail


def test_check_lora_missing_qc_report_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    os.remove(os.path.join(str(tmp_path), "adapters/demo-lora-family-v1-qc.json"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert _by_name(report)["adapters/demo-lora-family-v1-qc.json"].status == "missing"
    _edit_manifest(manifest, lambda p: p.pop("qcReport"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert _by_name(report)["qcReport"].status == "missing"


def test_check_lora_unknown_training_mode_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    _edit_manifest(manifest, lambda p: p.update(trainingMode="legacy_inline"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert len(report.requirements) == 1
    assert report.requirements[0].status == "invalid"
    assert "legacy_inline" in report.requirements[0].detail


def test_check_lora_manifest_without_arms_or_valid_json_blocks(tmp_path):
    manifest = _package(str(tmp_path))
    _edit_manifest(manifest, lambda p: p.pop("outputs"))
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert report.requirements[0].status == "invalid"
    assert "no training arms" in report.requirements[0].detail

    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    report = data_readiness.check_lora_package(manifest, root=str(tmp_path))
    assert "not valid JSON" in report.requirements[0].detail


def test_check_lora_directory_target_resolution(tmp_path):
    _package(str(tmp_path))
    adapters = os.path.join(str(tmp_path), "adapters")
    report = data_readiness.check_lora_package(adapters, root=str(tmp_path))
    assert report.ready

    with open(os.path.join(adapters, "second-manifest.json"), "w") as handle:
        handle.write("{}")
    with pytest.raises(IsADirectoryError) as exc:
        data_readiness.check_lora_package(adapters, root=str(tmp_path))
    assert "name the one to check" in str(exc.value)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        data_readiness.check_lora_package(str(empty))
    with pytest.raises(FileNotFoundError):
        data_readiness.check_lora_package(str(tmp_path / "nope"))


def test_check_lora_resolves_paths_without_an_explicit_root(tmp_path):
    """The workspace-relative paths resolve from the manifest's own location
    when the configured root does not hold the data (a package checked by
    absolute path from elsewhere)."""
    manifest = _package(str(tmp_path))
    report = data_readiness.check_lora_package(manifest)
    assert report.ready, [r.detail for r in report.blockers]


# --- data check lora, through the CLI --------------------------------------


def _run_cli(monkeypatch, root, argv):
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    return cli.main(argv)


def test_cli_lora_ready_package(tmp_path, monkeypatch, capsys):
    _package(str(tmp_path))
    assert _run_cli(monkeypatch, tmp_path, ["data", "check", "lora"]) == 0
    out = capsys.readouterr().out
    assert "ready" in out and "NOT ready" not in out
    assert "demo-lora-v1 split" in out
    assert out.count("sha256 ") >= 3     # manifest + two files + qc report


def test_cli_lora_explicit_target_and_json(tmp_path, monkeypatch, capsys):
    """WP0 step 8: `--json` is the ENVELOPE flag on every agent-path verb
    (audit §2.2). The report this used to print bare is `result.report`, key
    for key."""
    _package(str(tmp_path))
    rc = _run_cli(monkeypatch, tmp_path,
                  ["data", "check", "lora",
                   "adapters/demo-lora-family-v1-manifest.json", "--json"])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready" and envelope["verb"] == "data check"
    report = envelope["result"]["report"]
    assert report["ready"] is True and report["blockerCount"] == 0
    assert report["authoringSpec"] == data_readiness.LORA_AUTHORING_SPEC
    assert report["manifestPath"].endswith("demo-lora-family-v1-manifest.json")


def test_cli_lora_blocker_exits_65(tmp_path, monkeypatch, capsys):
    """WP0 step 8: the scheduled 2 → 65 migration for `data check` blockers
    (audit §7 row 7). The human report is unchanged."""
    manifest = _package(str(tmp_path))
    _edit_manifest(manifest, lambda p: p["outputs"]["demo"]["training"].update(
        sha256="0" * 64))
    rc = _run_cli(monkeypatch, tmp_path, ["data", "check", "lora"])
    assert rc == 65
    assert "NOT ready (1 blocker(s))" in capsys.readouterr().out


def test_cli_lora_missing_package_exits_two(tmp_path, monkeypatch, capsys):
    rc = _run_cli(monkeypatch, tmp_path, ["data", "check", "lora"])
    assert rc == 2
    assert "data check:" in capsys.readouterr().err


def test_cli_usage_still_names_both_templates(tmp_path, monkeypatch, capsys):
    assert _run_cli(monkeypatch, tmp_path, ["data", "check", "other"]) == 64
    err = capsys.readouterr().err
    assert "data check optvec" in err and "data check lora" in err
