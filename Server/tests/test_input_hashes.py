import os
import hashlib
import pytest
from steerlab_server.experiment import input_hashes


def test_reuses_only_inside_review_and_rereads_next_operation(tmp_path, monkeypatch):
    path = tmp_path/'data'; path.write_bytes(b'bytes')
    actual = hashlib.file_digest
    calls = []
    def counted(*args): calls.append(1); return actual(*args)
    monkeypatch.setattr(hashlib, 'file_digest', counted)
    for _ in range(2):
        with input_hashes.session():
            first = input_hashes.file_hash(path)
            with input_hashes.session(): assert input_hashes.file_hash(path) == first
    assert len(calls) == 2


@pytest.mark.parametrize('change', ['write', 'replace', 'symlink'])
def test_changed_or_replaced_bytes_do_not_inherit_hash(tmp_path, change):
    path = tmp_path/'data'; path.write_bytes(b'old')
    stamp = path.stat()
    with pytest.raises(ValueError, match='input'):
        with input_hashes.session():
            input_hashes.file_hash(path)
            if change == 'write': path.write_bytes(b'new')
            else:
                other = tmp_path/'other'; other.write_bytes(b'new')
                if change == 'replace': os.replace(other, path)
                else: path.unlink(); path.symlink_to(other)
            if change != 'symlink': os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            input_hashes.file_hash(path)


def test_exit_rechecks_even_without_a_second_hash_call(tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'old')
    with pytest.raises(ValueError, match='before review finished'):
        with input_hashes.session():
            input_hashes.file_hash(path); path.write_bytes(b'new')
    assert input_hashes.file_hash(path) == hashlib.sha256(b'new').hexdigest()
