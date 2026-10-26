"""The golden rule (``golden_files``): a missing committed golden fails rather
than being written from the code under test, and writing is on request only
(review finding N1, October 23, 2026)."""

import pytest

import golden_files


def _check(path):
    golden_files.check(str(path), "fresh\n", variable=golden_files.CROSS_ENGINE,
                       test_file="tests/test_example.py", label="example.json")


def test_a_missing_golden_fails_and_is_not_written(tmp_path, monkeypatch):
    monkeypatch.delenv(golden_files.CROSS_ENGINE, raising=False)
    path = tmp_path / "example.json"
    with pytest.raises(AssertionError, match="is missing, so nothing was compared"):
        _check(path)
    assert not path.exists()


def test_a_drifted_golden_fails_and_names_how_to_rewrite_it(tmp_path, monkeypatch):
    monkeypatch.delenv(golden_files.CROSS_ENGINE, raising=False)
    path = tmp_path / "example.json"
    path.write_text("committed\n")
    with pytest.raises(AssertionError, match="STEERLAB_WRITE_CROSS_ENGINE_GOLDENS=1"):
        _check(path)
    assert path.read_text() == "committed\n"


def test_goldens_are_written_on_request_only(tmp_path, monkeypatch):
    path = tmp_path / "example.json"
    monkeypatch.setenv(golden_files.CROSS_ENGINE, "1")
    _check(path)
    assert path.read_text() == "fresh\n"
    monkeypatch.delenv(golden_files.CROSS_ENGINE)
    _check(path)  # now committed and matching
