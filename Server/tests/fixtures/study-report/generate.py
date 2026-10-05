"""Regenerates the golden results pages in this directory.

Each ``.html`` file is the page ``reports.study_results`` draws for one small
synthetic workspace, built by the writers in ``tests/test_study_results_page.py``
(which reuse the export suite's own): a Python-engine run, a Mac-engine run, a
judged study, a multi-agent study, and a study frozen with force. No model
produced these numbers; every value is chosen by hand so that a reader can
check a page against its source. ``test_study_results_page.py`` compares
against them byte for byte.

Run from ``Server/`` after a deliberate change to the page, then read the diff
of the ``.html`` files before committing it::

    python tests/fixtures/study-report/generate.py
"""
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[2]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / 'tests'))

import test_study_results_page  # noqa: E402

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as scratch:
        test_study_results_page.write_goldens(scratch)
    print('Rewrote', ', '.join(sorted(test_study_results_page.CASES)), 'in', HERE.name)
