"""The readable page for a stored J-lens assessment, and the ``science report`` verb.

The page is a pure function of the stored JSON: golden files pin its bytes for
both report shapes, and the rest of this module pins what the page must never
do — cut a list, lose a comparison's identity, reach the network, invent a
number, or write inside a run directory.
"""
import copy
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from steerlab_server import client_cli
from steerlab_server.client.diagnostic_commands import workspace_action
from steerlab_server.client.reports import charts, jlens_assessment as assessment_page, page, science_report
from steerlab_server.experiment import diagnostic_archives as archives

FIXTURES = Path(__file__).parent / 'fixtures' / 'science-report'
HEX64 = re.compile(r'\b[0-9a-f]{64}\b')


def stored(name):
    return (FIXTURES / (name + '.json')).read_bytes()


def rendered(name):
    return assessment_page.render(stored(name))


def section(text, key):
    """One card of the page, by its id."""
    return re.search(r'<section id="%s".*?</section>' % re.escape(key), text, re.S).group(0)


def altered(change):
    """The list-form fixture after ``change`` edits it, re-encoded as the engine encodes reports."""
    report = json.loads(stored('assessment-list'))
    change(report)
    return archives.encoded(report)


class Outline(HTMLParser):
    """What an assistive tool or a printer would find: tables, captions, header scopes, pictures, links."""

    def __init__(self):
        super().__init__()
        self.tags, self.ids, self.links, self.tables, self.pictures, self.cells = [], set(), [], [], [], []
        self._table, self._picture, self._cell = None, None, None

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        self.tags.append((tag, values))
        if 'id' in values: self.ids.add(values['id'])
        if tag == 'a': self.links.append(values.get('href'))
        if tag == 'table':
            self._table = {'caption': False, 'scopes': [], 'unscoped': 0}; self.tables.append(self._table)
        if tag == 'caption' and self._table is not None: self._table['caption'] = True
        if tag == 'th' and self._table is not None:
            if 'scope' in values: self._table['scopes'].append(values['scope'])
            else: self._table['unscoped'] += 1
        if tag == 'svg' and values.get('role') == 'img':
            self._picture = {'labelledby': values.get('aria-labelledby', '').split(), 'named': set()}
            self.pictures.append(self._picture)
        if tag in ('title', 'desc') and self._picture is not None and 'id' in values:
            self._picture['named'].add(values['id'])
        if tag == 'td' and 'num' in (values.get('class') or '').split(): self._cell = []

    def handle_endtag(self, tag):
        if tag == 'table': self._table = None
        if tag == 'svg': self._picture = None
        if tag == 'td' and self._cell is not None:
            self.cells.append(''.join(self._cell)); self._cell = None

    def handle_data(self, data):
        if self._cell is not None: self._cell.append(data)


def outline(text):
    parser = Outline()
    parser.feed(text)
    return parser


@pytest.mark.parametrize('name', ['assessment-list', 'assessment-single'])
def test_each_report_shape_renders_to_its_golden_page(name):
    golden = (FIXTURES / (name + '.html')).read_bytes()
    assert rendered(name).encode('utf-8') == golden, (
        'The page changed for a stored report. If that is intended, run '
        'tests/fixtures/science-report/generate.py from Server/ and review the diff.')


def test_the_fixtures_are_what_their_generator_writes():
    # A fixture edited by hand would no longer match the numbers its generator documents.
    generator = FIXTURES / 'generate.py'
    scope = {'__file__': str(generator), '__name__': 'science_report_fixtures'}
    exec(compile(generator.read_text(), str(generator), 'exec'), scope)
    for name, build in (('assessment-list', scope['list_report']), ('assessment-single', scope['single_report']),
                        ('assessment-legacy', scope['legacy_report'])):
        assert stored(name) == archives.encoded(build())


def test_a_render_is_deterministic_within_and_across_processes():
    first = rendered('assessment-list')
    assert first == rendered('assessment-list')
    digest = hashlib.sha256(first.encode('utf-8')).hexdigest()
    program = ('import hashlib, sys; from steerlab_server.client.reports import jlens_assessment as p; '
               'print(hashlib.sha256(p.render(open(sys.argv[1], "rb").read()).encode("utf-8")).hexdigest())')
    for seed in ('1', '2'):
        # Set and dictionary order depend on the hash seed; the page must not.
        output = subprocess.run([sys.executable, '-c', program, str(FIXTURES / 'assessment-list.json')],
                                cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True,
                                env={**os.environ, 'PYTHONHASHSEED': seed}).stdout.strip()
        assert output == digest


def test_every_comparison_keeps_its_identity_and_nothing_is_cut():
    report = json.loads(stored('assessment-list'))
    text = rendered('assessment-list')
    assert hashlib.sha256(stored('assessment-list')).hexdigest() in section(text, 'identity')
    assert len(report['comparisons']) == 4
    for number, comparison in enumerate(report['comparisons'], 1):
        block = re.search(r'<div id="c%d" class="sub">.*?(?=<div id="c%d" class="sub">|</section>)' % (number, number + 1),
                          text, re.S).group(0)
        for value in (comparison['comparisonSHA256'], comparison['candidateLensID'], comparison['referenceLensID'],
                      comparison['corpus']['path'], comparison['corpus']['sha256'], comparison['heldOutStatus'],
                      'matches the stored entry'):
            assert value in block
        for layer, groups in comparison['layers'].items():
            # Every stored per-layer value is on the page, in this comparison's own table or its text's table.
            assert assessment_page.mean_text(groups['betweenLenses']['meanTopKOverlap']) in block
            assert assessment_page.mean_text(groups['betweenLenses']['meanJSDivergence']) in block
        identity_row = re.search(r'<tr>\s*<th scope="row" class="num"><a href="#c%d">.*?</tr>' % number, text, re.S).group(0)
        assert comparison['comparisonSHA256'] in identity_row and comparison['corpus']['sha256'] in identity_row
    for lens in report['lenses']:
        for value in (lens['lensID'], lens['converted']['sha256'], lens['source']['tensorSHA256']):
            assert value in section(text, 'identity') and value in section(text, 'lenses')
    # Hashes are printed whole, and no list on the page ends in an ellipsis.
    assert all(len(found) == 64 for found in HEX64.findall(text))
    assert '…' not in text and '...' not in text
    assert 'recipe-2' in text  # the one skipped row is named, not counted away


def test_each_text_has_a_chart_and_a_table_for_every_reading():
    report = json.loads(stored('assessment-list'))
    text = rendered('assessment-list')
    for number, corpus in enumerate(report['corpora'], 1):
        block = section(text, f't{number}')
        assert corpus['path'] in block and corpus['sha256'] in block
        for slug in ('overlap', 'divergence'):
            figure = re.search(r'<figure class="chart" id="t%d-%s-chart">.*?</figure>' % (number, slug), block, re.S).group(0)
            legend = re.search(r'<ul class="legend">.*?</ul>', figure, re.S).group(0)
            for label in ('Plain baseline', 'Reference lens-round-1', 'Candidate lens-round-2', 'Candidate lens-mixed'):
                assert label in legend
            assert 'Every value is in the table that follows.' in figure
            table = re.search(r'<table id="t%d-%s">.*?</table>' % (number, slug), block, re.S).group(0)
            assert table.count('<tr>') == 1 + len(report['comparisons'][0]['layers'])
    first = report['comparisons'][0]
    overlap = re.search(r'<table id="t1-overlap">.*?</table>', text, re.S).group(0)
    for layer, groups in first['layers'].items():
        baseline = first['readoutComparison']['layers'][layer]['native']['logitLensToFinal']['meanTopKOverlap']
        for value in (baseline, groups['referenceToFinal']['meanTopKOverlap'], groups['candidateToFinal']['meanTopKOverlap']):
            assert f'>{assessment_page.mean_text(value)}<' in overlap


def test_a_mixed_lens_shows_each_text_it_was_fitted_on():
    report = json.loads(stored('assessment-list'))
    mixed = next(lens for lens in report['lenses'] if lens['fit']['corpora'])
    block = re.search(r'<div id="lens-3" class="sub">.*?</section>', rendered('assessment-list'), re.S).group(0)
    assert 'Mixed: fitted on 2 texts' in block and mixed['lensID'] in block
    for entry in mixed['fit']['corpora']:
        row = re.search(r'<tr>\s*<td><code class="long">%s</code></td>.*?</tr>' % entry['corpusSHA256'], block, re.S).group(0)
        assert f'>{entry["promptsFitted"]}<' in row and f'>{entry["rowsConsidered"]}<' in row
    assert 'heldout/recipes.jsonl' in block and 'not held out for this lens' in block
    # A lens fitted on one text says so, and has no contribution table.
    single = re.search(r'<div id="lens-2" class="sub">.*?(?=<div id="lens-3")', rendered('assessment-list'), re.S).group(0)
    assert 'One text,' in single and '<table' not in single


def test_omitted_and_failed_comparisons_are_listed_under_limits():
    def change(report):
        omitted = report['comparisons'].pop(3)
        assert (omitted['candidateLensID'], omitted['corpus']['path']) == ('lens-mixed', 'heldout/recipes.jsonl')
        failed = report['comparisons'][1]
        failed['rows'] = [{'id': row['id'], 'status': 'skipped-too-short'} for row in failed['rows']]
        for groups in failed['layers'].values():
            for group in groups.values():
                group.update(positions=0, jsDivergenceSum=0.0, topKOverlapSum=0.0, meanJSDivergence=None, meanTopKOverlap=None)
        report['comparisons'].append('not a comparison')
    text = assessment_page.render(altered(change))
    limits = section(text, 'limits')
    assert ('The request named candidate <code>lens-mixed</code> on <code>heldout/recipes.jsonl</code>, '
            'but the report holds no comparison for that pair.') in limits
    assert 'produced no value: no token position was assessed' in limits and 'href="#c2"' in limits
    assert 'Entry 4 in the report’s list of comparisons could not be read and is not shown.' in limits
    assert 'essay-1' in limits and 'essay-2' in limits
    # The edited entry no longer matches the digest the engine stored for it, and the page says so.
    assert 'The stored digest of comparison <a href="#c2">2</a> does not match its stored content.' in limits
    assert 'None found.' not in limits
    assert 'has no stored value on this text, so it cannot be placed' in section(text, 'summary')
    # The untouched report has nothing of the kind to report beyond its one skipped row.
    clean = section(rendered('assessment-list'), 'limits')
    assert 'no comparison for that pair' not in clean and 'produced no value' not in clean
    assert 'recipe-2' in clean and 'comparisons <a href="#c3">3</a>, <a href="#c4">4</a>' in clean
    assert 'None found.' in section(rendered('assessment-single'), 'limits')


def test_a_layer_without_a_value_is_named_and_shown_as_no_value():
    def change(report):
        group = report['comparisons'][0]['layers']['1']['candidateToFinal']
        group.update(positions=0, meanJSDivergence=None, meanTopKOverlap=None)
    text = assessment_page.render(altered(change))
    assert 'has no value at 1 layer: 1.' in section(text, 'limits')
    assert f'<span class="none">{page.NO_VALUE}</span>' in re.search(r'<table id="t1-overlap">.*?</table>', text, re.S).group(0)


def test_the_page_reaches_nothing_outside_itself():
    for name in ('assessment-list', 'assessment-single', 'assessment-legacy'):
        text = rendered(name)
        found = outline(text)
        lowered = text.lower()
        for forbidden in ('http://', 'https://', '<script', '<link', '<img', '<iframe', '<object', '<embed', '@import',
                          'url(', 'src=', 'srcset=', 'javascript:', ' onclick', ' onload'):
            assert forbidden not in lowered, forbidden
        assert found.links and all(link.startswith('#') and link[1:] in found.ids for link in found.links)
        policy = [values['content'] for tag, values in found.tags if tag == 'meta' and values.get('http-equiv')]
        assert policy == ["default-src 'none'; style-src 'unsafe-inline'"]
        assert text.startswith(page.MARKER)


def test_tables_are_real_tables_and_every_chart_has_a_text_alternative():
    found = outline(rendered('assessment-list'))
    assert len(found.tables) >= 10
    for table in found.tables:
        assert table['caption'] and table['unscoped'] == 0 and 'col' in table['scopes']
    assert len(found.pictures) == 4
    for picture in found.pictures:
        assert len(picture['labelledby']) == 2 and set(picture['labelledby']) == picture['named']
    text = rendered('assessment-list')
    assert '@media (prefers-color-scheme:dark)' in text and '@media print' in text
    assert 'stroke-dasharray' in charts.CHART_STYLE.split('@media print,(forced-colors:active)')[1]


def test_no_number_on_the_page_was_computed_by_the_page():
    report = json.loads(stored('assessment-list'))
    allowed = set()

    def collect(value):
        if isinstance(value, dict):
            for item in value.values(): collect(item)
        elif isinstance(value, list):
            for item in value: collect(item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            allowed.update(filter(None, (assessment_page.mean_text(value), assessment_page._general(value),
                                         assessment_page._count(value))))
    collect(report)
    decimal = re.compile(r'^-?\d[\d,]*\.\d+(e-?\d+)?$|^-?\d\.\d+e-?\d+$')
    cells = outline(rendered('assessment-list')).cells
    shown = [cell.strip() for cell in cells if decimal.match(cell.strip())]
    assert len(shown) > 200
    assert set(shown) <= allowed
    # The only other numeric cells are whole counts and "n of N" layer tallies.
    others = {cell.strip() for cell in cells} - set(shown)
    assert all(re.fullmatch(r'[\d,]+( / [\d,]+)*|\d+ of \d+|%s' % re.escape(page.NO_VALUE), cell) for cell in others), others


def test_layer_tallies_only_order_stored_values():
    report = json.loads(stored('assessment-list'))
    comparisons, _ = assessment_page.comparisons_of(report)
    counted = assessment_page.tally(comparisons[0], 'referenceToFinal', assessment_page.OVERLAP, True)
    assert counted == {'closer': ['0', '2'], 'further': [], 'same': ['1'], 'missing': []}
    counted = assessment_page.tally(comparisons[1], 'referenceToFinal', assessment_page.DIVERGENCE, False)
    assert counted == {'closer': ['1'], 'further': ['0', '2'], 'same': [], 'missing': []}
    summary = section(rendered('assessment-list'), 'summary')
    assert 'more closely than the reference at 2 of 3 layers (0, 2), less closely at no layer, and equally at 1 of 3 layers (1).' in summary
    assert 'This text is not held out for this pair of lenses.' in summary
    assert 'no average across layers, no interval, and no significance test' in summary


def test_stored_text_can_never_become_markup():
    hostile = '<script>alert(1)</script>"onmouseover="x'
    def change(report):
        for comparison in report['comparisons']:
            if comparison['candidateLensID'] == 'lens-mixed': comparison['candidateLensID'] = hostile
            comparison['corpus']['path'] = comparison['corpus']['path'].replace('essays', '<b>essays</b>')
        report['config']['candidateLensIDs'][1] = hostile
        report['lenses'][2]['lensID'] = hostile
        report['runtime']['note'] = '</style><img src=x>'
    text = assessment_page.render(altered(change))
    assert '<script' not in text and '<img' not in text and '<b>' not in text and '"onmouseover="' not in text
    assert '&lt;script&gt;alert(1)&lt;/script&gt;&quot;onmouseover=&quot;x' in text
    assert outline(text).tables  # still parses as the same page


def test_the_older_single_form_renders_and_says_what_it_lacks():
    text = rendered('assessment-legacy')
    assert 'Plain baseline: none is recorded in this report.' in section(text, 'summary')
    assert 'This report was written before each comparison carried its own digest.' in text
    limits = section(text, 'limits')
    assert 'records no plain baseline, so none is drawn for it.' in limits
    figure = re.search(r'<figure class="chart" id="t1-overlap-chart">.*?</figure>', text, re.S).group(0)
    assert 'Plain baseline' not in figure and 'Reference lens-round-1' in figure and 'Candidate lens-round-2' in figure
    with pytest.raises(assessment_page.ReportError, match='not a J-lens assessment report'):
        assessment_page.render(b'{"operation": "jlens-fit", "schemaVersion": 1}')
    with pytest.raises(assessment_page.ReportError, match='schema version'):
        assessment_page.render(b'{"operation": "jlens-fit-assess", "schemaVersion": 3}')
    with pytest.raises(assessment_page.ReportError, match='not readable JSON'):
        assessment_page.render(b'not json')


def test_more_candidates_than_colours_are_split_over_charts_and_ticks_are_round():
    def change(report):
        template = report['comparisons'][0]
        report['comparisons'] = []
        report['config']['candidateLensIDs'] = [f'lens-{index}' for index in range(charts.SLOTS + 2)]
        report['config']['corpora'] = [template['corpus']]
        for name in report['config']['candidateLensIDs']:
            entry = copy.deepcopy(template); entry['candidateLensID'] = name
            entry.pop('comparisonSHA256'); entry['comparisonSHA256'] = archives.digest(entry)
            report['comparisons'].append(entry)
    text = assessment_page.render(altered(change))
    assert 'id="t1-overlap-chart-1"' in text and 'id="t1-overlap-chart-2"' in text and 'id="t1-overlap-chart-3"' not in text
    assert 'candidates 1 to 8' in text and 'candidates 9 to 10' in text
    assert [str(tick) for tick in charts.ticks_to(1)] == ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    assert [str(tick) for tick in charts.ticks_to(0.6)] == ['0.0', '0.2', '0.4', '0.6']
    assert [str(tick) for tick in charts.ticks_to(0.0031)] == ['0.000', '0.001', '0.002', '0.003', '0.004']
    assert charts.ticks_to(0) == charts.ticks_to(float('nan')) == charts.ticks_to(None)


# -- the verb -----------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    run = root / 'runs' / 'jlens-assessment-example'
    (run / 'comparisons').mkdir(parents=True)
    (run / 'assessment-report.json').write_bytes(stored('assessment-list'))
    (run / 'comparisons' / 'one.json').write_bytes(archives.encoded(json.loads(stored('assessment-list'))['comparisons'][0]))
    (run / 'COMPLETED').write_text('jlens-fit-assess\n')
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    return root.resolve()


def listing(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def cli(capsys, *arguments):
    capsys.readouterr()
    code = client_cli.main([*arguments, '--json'])
    return code, json.loads(capsys.readouterr().out)


def test_report_renders_outside_the_run_and_leaves_the_run_untouched(workspace, capsys):
    before = listing(workspace / 'runs')
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--root', str(workspace))
    assert code == 0 and document['state'] == 'ready' and document['verb'] == 'science report' and document['changed'] is True
    result = document['result']
    target = workspace / 'reports' / 'jlens-assessment-example' / 'assessment-report.html'
    assert result['htmlPath'] == str(target) and result['rendered'] is True and result['inRunDirectory'] is False
    assert result['sourceReport'] == 'runs/jlens-assessment-example/assessment-report.json'
    assert result['sourceSHA256'] == hashlib.sha256(stored('assessment-list')).hexdigest()
    assert target.read_bytes() == (FIXTURES / 'assessment-list.html').read_bytes()
    assert result['htmlSHA256'] == archives.file_hash(target)
    assert listing(workspace / 'runs') == before
    # The report file itself is an equally good way to name the run, and a second render changes nothing.
    code, again = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example/assessment-report.json', '--root', str(workspace))
    assert code == 0 and again['changed'] is False and again['result']['htmlPath'] == str(target)
    assert 'already up to date' in again['message']
    assert listing(workspace / 'runs') == before
    assert sorted(path.name for path in target.parent.iterdir()) == ['assessment-report.html']


def test_report_writes_where_out_says_but_never_into_a_run(workspace, capsys):
    before = listing(workspace / 'runs')
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', 'shared/for-a-colleague.html',
                         '--root', str(workspace))
    assert code == 0 and document['result']['htmlPath'] == str(workspace / 'shared' / 'for-a-colleague.html')
    assert (workspace / 'shared' / 'for-a-colleague.html').read_bytes() == (FIXTURES / 'assessment-list.html').read_bytes()
    assert not (workspace / 'reports').exists()
    outside = workspace.parent / 'elsewhere' / 'page.html'
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', str(outside), '--root', str(workspace))
    assert code == 0 and outside.read_bytes() == (FIXTURES / 'assessment-list.html').read_bytes()
    for refused in ('runs/jlens-assessment-example/assessment-report.html', 'runs/new-folder/page.html',
                    str(workspace / 'runs' / 'page.html')):
        code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', refused, '--root', str(workspace))
        assert code != 0 and document['error']['code'] == 'reportRefused'
        assert 'never written inside runs/' in document['error']['reason']
        assert 'reports/' in document['error']['repairAction']
    # A completed run kept somewhere other than runs/ is protected by its marker.
    kept = workspace / 'archive' / 'old-run'
    kept.mkdir(parents=True); (kept / 'COMPLETED').write_text('jlens-fit-assess\n')
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', 'archive/old-run/page.html',
                         '--root', str(workspace))
    assert code != 0 and 'completed run folder' in document['error']['reason']
    assert listing(workspace / 'runs') == before and listing(kept) == {'COMPLETED': b'jlens-fit-assess\n'}


def test_report_replaces_only_its_own_pages(workspace, capsys):
    notes = workspace / 'notes.html'
    notes.write_text('<!doctype html>\n<p>My own notes.</p>\n')
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', 'notes.html', '--root', str(workspace))
    assert code != 0 and document['error']['code'] == 'reportRefused' and 'Nothing was changed' in document['error']['reason']
    assert notes.read_text() == '<!doctype html>\n<p>My own notes.</p>\n'
    stale = workspace / 'reports' / 'stale.html'
    stale.parent.mkdir(); stale.write_text(page.MARKER + '<p>An older page.</p>\n')
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', 'reports/stale.html', '--root', str(workspace))
    assert code == 0 and document['changed'] is True
    assert stale.read_bytes() == (FIXTURES / 'assessment-list.html').read_bytes()
    assert not [path for path in stale.parent.iterdir() if path.name.startswith('.')]  # no temporary file left behind


def test_a_run_that_already_holds_its_page_is_answered_with_that_file(workspace, capsys):
    run = workspace / 'runs' / 'jlens-assessment-example'
    stored_page = run / 'assessment-report.html'
    stored_page.write_bytes((FIXTURES / 'assessment-list.html').read_bytes())
    before = listing(workspace)
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--root', str(workspace))
    assert code == 0 and document['changed'] is False
    assert document['result']['htmlPath'] == str(stored_page)
    assert document['result']['rendered'] is False and document['result']['inRunDirectory'] is True
    assert listing(workspace) == before  # nothing written anywhere
    # Asking for a copy elsewhere still draws one, and still leaves the run alone.
    code, document = cli(capsys, 'science', 'report', 'runs/jlens-assessment-example', '--out', 'copy.html', '--root', str(workspace))
    assert code == 0 and document['result']['rendered'] is True and (workspace / 'copy.html').is_file()
    assert listing(run) == {key.removeprefix('runs/jlens-assessment-example/'): value for key, value in before.items()
                            if key.startswith('runs/jlens-assessment-example/')}


@pytest.mark.parametrize('argument, reason, repair', [
    ('runs/missing', 'Nothing was found at runs/missing', 'assessment-report.json'),
    ('runs/jlens-assessment-example/comparisons', 'holds no report that can be drawn', 'assessment-report.json'),
    ('runs/jlens-assessment-example/comparisons/one.json', 'one comparison taken from an assessment', 'run folder'),
    ('runs/jlens-assessment-example/COMPLETED', 'not readable JSON', 'assessment-report.json'),
    ('other.json', 'not a report this command can draw', 'jlens-fit-assess'),
])
def test_report_refusals_say_what_to_give_instead(workspace, capsys, argument, reason, repair):
    (workspace / 'other.json').write_text('{"operation": "jlens-fit", "schemaVersion": 1}')
    before = listing(workspace)
    code, document = cli(capsys, 'science', 'report', argument, '--root', str(workspace))
    assert code != 0 and document['error']['code'] == 'reportRefused'
    assert reason in document['error']['reason'] and repair in document['error']['repairAction']
    assert listing(workspace) == before


def test_report_reads_only_inside_the_workspace(workspace, capsys, tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_bytes(stored('assessment-single'))
    code, document = cli(capsys, 'science', 'report', str(outside), '--root', str(workspace))
    assert code != 0 and 'inside this workspace' in document['error']['reason']
    assert '--root' in document['error']['repairAction'] and '--workspace' in document['error']['repairAction']
    (workspace / 'link.json').symlink_to(outside)
    code, document = cli(capsys, 'science', 'report', 'link.json', '--root', str(workspace))
    assert code != 0 and 'inside this workspace' in document['error']['reason']
    # A loose copy inside the workspace is fine, and is filed by its own hash.
    (workspace / 'received.json').write_bytes(stored('assessment-single'))
    code, document = cli(capsys, 'science', 'report', 'received.json', '--root', str(workspace))
    digest = hashlib.sha256(stored('assessment-single')).hexdigest()
    assert code == 0 and document['result']['htmlPath'] == str(workspace / 'reports' / ('report-' + digest[:12]) / 'assessment-report.html')
    assert Path(document['result']['htmlPath']).read_bytes() == (FIXTURES / 'assessment-single.html').read_bytes()


def test_the_bridge_action_and_the_workbench_route_share_the_owner(workspace):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.diagnostic_transport_routes import build_router
    direct = workspace_action('report', {'workspaceRoot': str(workspace), 'path': 'runs/jlens-assessment-example'})
    assert direct['changed'] is True and direct['rendered'] is True
    again = science_report.render('runs/jlens-assessment-example', workspace)
    assert again['changed'] is False and {key: value for key, value in again.items() if key not in ('changed', 'message')} == {
        key: value for key, value in direct.items() if key not in ('changed', 'message')}
    with pytest.raises(archives.Refusal, match='exactly the declared action fields'):
        workspace_action('report', {'workspaceRoot': str(workspace), 'path': 'runs/jlens-assessment-example', 'extra': 'x'})
    with pytest.raises(archives.Refusal, match='needs a file name'):
        workspace_action('report', {'workspaceRoot': str(workspace), 'path': 'runs/jlens-assessment-example', 'out': ''})
    app = FastAPI(); app.include_router(build_router(SimpleNamespace()))
    with TestClient(app) as client:
        response = client.post('/api/science/workspace/report', json={'workspaceRoot': str(workspace), 'path': 'runs/jlens-assessment-example'})
        assert response.status_code == 200 and response.json()['htmlPath'] == direct['htmlPath'] and response.json()['changed'] is False
        refused = client.post('/api/science/workspace/report', json={'workspaceRoot': str(workspace), 'path': 'runs/missing'})
        assert refused.status_code == 409 and refused.json()['detail']['code'] == 'reportRefused'
        assert 'assessment-report.json' in refused.json()['detail']['repairAction']


def test_the_verb_is_declared_with_its_own_out_flag():
    from steerlab_server.client import science_commands
    spec = next(spec for spec in science_commands.VERB_SPECS if spec.verb == 'report')
    assert spec.value_flags == {'--out'} and not spec.required_flags
    assert 'never written to' in spec.purpose and 'reports/' in spec.purpose
    assert client_cli.main(['science', 'report', '--help']) == 0


def test_drawing_a_page_loads_no_model_or_network_library():
    program = ('import sys; import steerlab_server.client.reports.science_report, steerlab_server.client.reports.jlens_assessment; '
               'heavy = sorted(name for name in sys.modules if name.split(".")[0] in '
               '("torch", "numpy", "transformers", "safetensors", "httpx", "fastapi", "pyarrow", "huggingface_hub")); '
               'print(heavy)')
    output = subprocess.run([sys.executable, '-c', program], cwd=Path(__file__).resolve().parents[1], check=True,
                            capture_output=True, text=True).stdout.strip()
    assert output == '[]'
