"""Stimulus-set hashing must match the Swift contract: SHA-256 over the raw
bytes of positive.jsonl then negative.jsonl. If this drifts, frozen manifests
written on one engine fail to verify on the other.
"""

import hashlib
import os

from steerlab_server.steering.stimulus_set import StimulusSet, load_texts


def _write(path, text):
    with open(path, "wb") as handle:
        handle.write(text)


def test_hash_is_sha256_of_positive_then_negative_bytes(tmp_path):
    pos_bytes = b'{"text": "alpha"}\n{"text": "beta"}\n'
    neg_bytes = b'{"text": "gamma"}\n'
    _write(tmp_path / "positive.jsonl", pos_bytes)
    _write(tmp_path / "negative.jsonl", neg_bytes)

    expected = hashlib.sha256(pos_bytes + neg_bytes).hexdigest()
    stim = StimulusSet.from_directory(str(tmp_path))
    assert stim.hash == expected
    assert stim.positive == ["alpha", "beta"]
    assert stim.negative == ["gamma"]
    assert stim.name == os.path.basename(str(tmp_path))


def test_hash_order_matters(tmp_path):
    _write(tmp_path / "positive.jsonl", b'{"text": "a"}\n')
    _write(tmp_path / "negative.jsonl", b'{"text": "b"}\n')
    stim = StimulusSet.from_directory(str(tmp_path))
    swapped = hashlib.sha256(b'{"text": "b"}\n' + b'{"text": "a"}\n').hexdigest()
    assert stim.hash != swapped  # positive-then-negative order is load-bearing


def test_load_texts_hashes_raw_bytes(tmp_path):
    data = b'{"text": "one"}\n{"text": "two"}\n'
    path = tmp_path / "corpus.jsonl"
    _write(path, data)
    loaded = load_texts(str(path))
    assert loaded.texts == ["one", "two"]
    assert loaded.hash == hashlib.sha256(data).hexdigest()
