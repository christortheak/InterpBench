"""A newly completed assessment carries its readable page; a finished run is never added to."""
import json
from pathlib import Path

import pytest
from test_jlens_fit import fitting  # noqa: F401 - fixture the library builds on
from test_jlens_multi_candidate_assessment import library, plural, single  # noqa: F401
from steerlab_server import client_cli
from steerlab_server.client.reports import jlens_assessment as page
from steerlab_server.experiment import diagnostic_archives as archives
from steerlab_server.experiment import jlens_assessment as assessment


def files(run):
    return {path.relative_to(run).as_posix(): path.read_bytes() for path in sorted(run.rglob('*')) if path.is_file()}


def test_a_completed_assessment_holds_the_page_drawn_from_its_own_report(library):
    root, base, ids, corpora = library
    result = assessment.assess(plural(base, ids[1:], corpora), root=root)
    run = Path(result['runDirectory'])
    stored = (run/'assessment-report.json').read_bytes()
    written = run/page.PAGE_NAME
    assert result['pagePath'] == str(written)
    assert written.read_bytes() == page.render(stored).encode('utf-8')
    # The page is part of the output that export, transfer, and custody inventory.
    relative = run.relative_to(Path(root).resolve()).as_posix()
    assert f'{relative}/{page.PAGE_NAME}' in archives.files_in(root, relative)
    assert sorted(path.name for path in run.iterdir()) == ['COMPLETED', 'assessment-report.html', 'assessment-report.json', 'comparisons']
    text = written.read_text(encoding='utf-8')
    document = json.loads(stored)
    assert len(document['comparisons']) == 4
    for comparison in document['comparisons']:
        assert comparison['comparisonSHA256'] in text and comparison['candidateLensID'] in text and comparison['corpus']['sha256'] in text
    assert all(lens in text for lens in ids)
    assert 'http://' not in text and 'https://' not in text and '<script' not in text


def test_the_single_form_holds_its_page_too(library):
    root, base, ids, corpora = library
    result = assessment.assess(single(base, ids[1], corpora[0]), root=root)
    run = Path(result['runDirectory'])
    assert json.loads((run/'assessment-report.json').read_bytes())['schemaVersion'] == 1
    assert (run/page.PAGE_NAME).read_bytes() == page.render((run/'assessment-report.json').read_bytes()).encode('utf-8')


def test_report_on_a_new_run_answers_with_the_page_the_engine_wrote(library, capsys):
    root, base, ids, corpora = library
    run = Path(assessment.assess(plural(base, ids[1:], corpora), root=root)['runDirectory'])
    before = files(run)
    relative = run.relative_to(Path(root).resolve()).as_posix()
    capsys.readouterr()
    assert client_cli.main(['science', 'report', relative, '--root', str(root), '--json']) == 0
    document = json.loads(capsys.readouterr().out)
    assert document['changed'] is False and document['result']['inRunDirectory'] is True
    assert document['result']['htmlPath'] == str(run/page.PAGE_NAME)
    assert files(run) == before and not (Path(root)/'reports').exists()


def test_report_on_a_run_without_a_page_draws_one_outside_it(library, capsys):
    root, base, ids, corpora = library
    run = Path(assessment.assess(single(base, ids[1], corpora[0]), root=root)['runDirectory'])
    (run/page.PAGE_NAME).unlink()   # a run completed before the engine wrote pages
    before = files(run)
    relative = run.relative_to(Path(root).resolve()).as_posix()
    capsys.readouterr()
    assert client_cli.main(['science', 'report', relative, '--root', str(root), '--json']) == 0
    document = json.loads(capsys.readouterr().out)
    target = Path(root).resolve()/'reports'/run.name/page.PAGE_NAME
    assert document['result']['htmlPath'] == str(target) and document['result']['inRunDirectory'] is False
    assert target.read_bytes() == page.render(before['assessment-report.json']).encode('utf-8')
    assert files(run) == before


def test_a_page_that_cannot_be_drawn_is_logged_and_never_costs_the_run(library, monkeypatch):
    root, base, ids, corpora = library
    def broken(data): raise RuntimeError('fixture page failure')
    monkeypatch.setattr(page, 'render', broken)
    lines = []
    result = assessment.assess(single(base, ids[1], corpora[0]), root=root, log=lines.append)
    run = Path(result['runDirectory'])
    assert 'pagePath' not in result and not (run/page.PAGE_NAME).exists()
    assert (run/'COMPLETED').read_text() == 'jlens-fit-assess\n'
    assert json.loads((run/'assessment-report.json').read_bytes())['comparisons']
    assert any('report page was not written' in line and 'fixture page failure' in line and 'science report' in line for line in lines)
