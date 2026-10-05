"""The private-name list: its loader (``scripts/ci/private_names.py``) and the
suite's door to it (``private_name_guard``).

The neutrality guards check that one researcher's site, study, and account
names stay out of what ships. They used to hold those names as string
literals. The list now lives outside the repository, and these tests hold the
replacement to its contract:

* the loader reads terms and ``allow:`` contexts, and treats a missing or
  term-less list as ABSENT, so no guard can pass vacuously against it;
* a guard given a list still catches a planted term — and names it by its
  position in the list, never by the term, because the message lands in a
  public log;
* with no list a guard is skipped with the reason, and with
  ``STEERLAB_REQUIRE_PRIVATE_NAMES=1`` it fails;
* in CI, a repository secret becomes a required list, and a fork with no
  secret is left to skip.

Every term here is made up.
"""

import pathlib

import pytest

import private_name_guard

TERM = "quuxcluster"                    # made up; "private" only in this test


def _list(root: pathlib.Path, text: str = f"# made up\n{TERM}\n") -> pathlib.Path:
    path = root / "private-names.txt"
    path.write_text(text)
    return path


def test_the_loader_reads_terms_and_allowed_contexts(tmp_path):
    module = private_name_guard.loader()
    path = _list(tmp_path, "# comment\n\nQuuxCluster\nquuxcluster\n zorblab \nallow: ZorbLabish\n")
    names = module.load({module.FILE_VARIABLE: str(path)})
    assert names.terms == ("quuxcluster", "zorblab")
    assert names.allowed_contexts == ("zorblabish",)
    # The test guards read strictly: an allowed context does not soften them.
    # A hit is named by its place in the list, never by the term — a failing
    # guard's message lands in a public log.
    assert names.hits("ZorbLabish and QUUXCLUSTER") == [
        "list entry 1 (11 letters)", "list entry 2 (7 letters)"]
    assert names.hits("nothing here") == []
    assert "quuxcluster" not in repr(names)


def test_a_missing_or_term_less_list_is_absent(tmp_path):
    module = private_name_guard.loader()
    missing = tmp_path / "no-such-list.txt"
    assert module.load({module.FILE_VARIABLE: str(missing)}) is None
    empty = _list(tmp_path, "# only a comment\nallow:something\n")
    assert module.load({module.FILE_VARIABLE: str(empty)}) is None
    assert str(missing) in module.absent_reason({module.FILE_VARIABLE: str(missing)})
    assert module.list_path({}).name == "private-names.txt"
    assert not module.is_required({})
    assert module.is_required({module.REQUIRE_VARIABLE: "1"})


def test_ci_turns_the_secret_into_a_required_list_and_a_fork_into_a_skip(
        tmp_path, capsys):
    """The main repository hands the list to a job as a secret; a fork has
    none. With the secret the list is written for the job and REQUIRED; with
    none, nothing is written and nothing is required — the fork's guards skip
    instead of failing."""
    module = private_name_guard.loader()
    job_env = tmp_path / "github-env"
    job_env.write_text("")
    runner = {"GITHUB_ENV": str(job_env), "RUNNER_TEMP": str(tmp_path)}

    assert module.install_from_ci_secret({**runner, module.SECRET_VARIABLE: ""}) == 0
    assert job_env.read_text() == ""
    assert not (tmp_path / "private-names.txt").exists()
    assert "will skip" in capsys.readouterr().out

    secret = f"# made up\n{TERM}\nallow:{TERM}Family"
    assert module.install_from_ci_secret({**runner, module.SECRET_VARIABLE: secret}) == 0
    written = tmp_path / "private-names.txt"
    assert written.read_text() == secret + "\n"
    assert written.stat().st_mode & 0o077 == 0            # owner-only
    assert job_env.read_text() == (
        f"STEERLAB_PRIVATE_NAMES_FILE={written}\nSTEERLAB_REQUIRE_PRIVATE_NAMES=1\n")
    out = capsys.readouterr().out
    assert f"::add-mask::{TERM}" in out                   # consumed by the runner
    assert "1 term(s), 1 allowed context(s)" in out
    # …and what it wrote is what the loader then reads.
    names = module.load({module.FILE_VARIABLE: str(written)})
    assert names.terms == (TERM,)

    # Outside a CI job there is nowhere to put it: refuse, do not guess.
    assert module.install_from_ci_secret({module.SECRET_VARIABLE: secret}) == 2


def test_a_guard_gets_the_list_when_it_is_present(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_PRIVATE_NAMES_FILE", str(_list(tmp_path)))
    names = private_name_guard.private_names()
    assert names.terms == (TERM,)
    private_name_guard.assert_names_nothing_private("a clean page", names, "the page")
    with pytest.raises(pytest.fail.Exception) as caught:
        private_name_guard.assert_names_nothing_private(
            "run it on QuuxCluster", names, "the page")
    assert "the page contains a private name: list entry 1 (11 letters)" in str(caught.value)
    assert TERM not in str(caught.value).lower()
    # What the test itself supplied is not the product's doing.
    private_name_guard.assert_names_nothing_private(
        "wrote /scratch/quuxcluster/run.log", names, "the page",
        ignoring=["/scratch/quuxcluster"])


def test_a_guard_is_skipped_without_the_list_and_fails_when_it_is_required(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_PRIVATE_NAMES_FILE", str(tmp_path / "absent.txt"))
    monkeypatch.delenv("STEERLAB_REQUIRE_PRIVATE_NAMES", raising=False)
    with pytest.raises(pytest.skip.Exception, match="no private-name list"):
        private_name_guard.private_names()
    monkeypatch.setenv("STEERLAB_REQUIRE_PRIVATE_NAMES", "1")
    with pytest.raises(pytest.fail.Exception, match="STEERLAB_REQUIRE_PRIVATE_NAMES is set"):
        private_name_guard.private_names()
