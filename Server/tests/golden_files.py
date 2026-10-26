"""Committed expected files ("goldens") a test compares what it produced with.

Written ON REQUEST only, as the Swift suites do (``ExperimentCLIEnvelopeTests``,
``PortabilityContractTests``): a missing golden FAILS. Before this, a missing
file was written and the test passed, so deleting one by accident regenerated
it from the code under test, and the Swift suite then read the fresh copy
(review finding, October 23, 2026). The structural assertions before the call
always run on the fresh document either way.

To write goldens, set the variable the Swift suites use for the same family
and re-run the test, then review the diff and commit it:
``STEERLAB_WRITE_CLI_ENVELOPE_GOLDENS=1`` for ``tests/fixtures/cli-envelopes/``,
``STEERLAB_WRITE_CROSS_ENGINE_GOLDENS=1`` for ``Tests/Fixtures/cross-engine/``.
"""

import os

CLI_ENVELOPES = "STEERLAB_WRITE_CLI_ENVELOPE_GOLDENS"
CROSS_ENGINE = "STEERLAB_WRITE_CROSS_ENGINE_GOLDENS"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hint(variable, test_file):
    return (f"set {variable}=1 and re-run `Server/.venv.nosync/bin/python -m pytest "
            f"{test_file}` from Server/, then review the diff and commit it")


def check(path, text, *, variable, test_file, label):
    """Compare ``text`` with the committed golden at ``path``; write it only
    when ``variable`` is ``1``."""
    if os.environ.get(variable) == "1":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return
    shown = os.path.relpath(path, REPO)
    assert os.path.exists(path), (
        f"{label}: the committed golden {shown} is missing, so nothing was compared. "
        f"Restore it from git, or, if a new golden is intended, {hint(variable, test_file)}")
    with open(path, encoding="utf-8") as handle:
        assert text == handle.read(), (
            f"{label}: drifted from {shown}; if the change is deliberate, "
            f"{hint(variable, test_file)}")
