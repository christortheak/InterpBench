"""The results page of an ordinary study, and the ``results report`` verb.

The page is drawn from the reading ``results export`` makes, so every fixture
here is a small synthetic workspace built with the export suite's own writers
(``test_results_export``). Golden files pin the page's bytes for a run from
each engine, a judged study, a multi-agent study, and a forced study. The rest
of this module pins what the page must never do: depend on the machine or the
hash seed, reach the network, show a number it computed, leave out a stored
effect, or write inside a run directory.
"""
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.client import diagnostic_commands, results_commands, results_export, results_report
from steerlab_server.client.reports import page, study_results
from steerlab_server.experiment import study_stats
from steerlab_server.experiment.manifest import Manifest

from test_results_export import (  # noqa: F401 - the export suite's writers, one set for both
    ANALYSIS, EVALUATION, ITEMS, RUN, SERVER_DIR, STUDY, _mac_effects_csv, _mac_records, _manifest,
    _panel_records, _python_analysis_by_hand, _python_records, _write_json, _write_lines, _write_mac_analysis,
    _write_python_paired_evaluation, _write_run, _write_study)

FIXTURES = Path(__file__).parent / 'fixtures' / 'study-report'
CODING = f'20261004T030000000-exp-{STUDY}-evaluate-2'
HEX64 = re.compile(r'\b[0-9a-f]{64}\b')


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(cli_envelope, 'now', lambda: 1_000.0)


@pytest.fixture
def root(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    return workspace


# --- fixture workspaces: every value chosen by hand ------------------------------


def _row(condition, endpoint, n, mean, low, high, w, p, adjusted, correction='bh', **extra):
    """One effect row in the Python engine's own row type."""
    return study_stats.EffectRow(
        condition=condition, endpoint=endpoint,
        ci=study_stats.BootstrapCI(n=n, mean=mean, ci_lower=low, ci_upper=high, replicates=10_000, seed=0),
        wilcoxon_w=w, wilcoxon_p=p, adjusted_p=adjusted, correction=correction, modality='injection', **extra)


NAN = float('nan')
STRONG = {'name': 'formal-strong', 'alphaInNormUnits': True, 'bandWidth': 1,
          'slots': [{'concept': 'formality', 'layer': 6, 'alpha': 8.0}]}
BASELINE = {'name': 'baseline', 'slots': [], 'alphaInNormUnits': True}
FORMAL = {'name': 'formal', 'alphaInNormUnits': True, 'bandWidth': 1,
          'slots': [{'concept': 'formality', 'layer': 6, 'alpha': 4.0}]}


def python_engine(root):
    """A Python-engine run, frozen cleanly, with a declared primary outcome, an
    exclusion stamp, a condition with too few pairs for an interval, and one
    cut-off response."""
    manifest = _manifest(status='frozen', frozenAt='2026-10-03T12:00:00Z', freezeHash='f' * 64,
                         primaryOutcome='choiceRate', taskPromptsFile='prompts/tasks/weekend.jsonl',
                         taskPromptsHash='a' * 64, conditions=[BASELINE, FORMAL, STRONG],
                         exclusionRules=[{'rule': 'unparseableEndpoint', 'endpoint': 'parsedChoice'}])
    _write_study(root, manifest)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    records = _python_records(manifest_hash, conditions=('baseline', 'formal', 'formal-strong'),
                              parsedChoice='A', target='A')
    _write_run(root, manifest, records, report={
        'experiment': STUDY, 'experimentHash': manifest_hash,
        'conditions': {
            'baseline': {'generations': 8, 'meanWordCount': 9.5, 'meanDistinct2': 0.5, 'choiceRate': 0.5},
            'formal': {'generations': 8, 'meanWordCount': 13.5, 'meanDistinct2': 0.5, 'choiceRate': 0.75,
                       'agreementWithBaseline': {'n': 4, 'agreement': 0.75}},
            'formal-strong': {'generations': 8, 'meanWordCount': 17.5, 'meanDistinct2': 0.5, 'choiceRate': 1.0,
                              'agreementWithBaseline': {'n': 4, 'agreement': 0.5}}},
        'truncation': {'threshold': 0.25, 'classified': 24, 'lengthStopped': 1, 'lengthStoppedInReasoning': 0,
                       'lengthStoppedFraction': 1 / 24, 'cells': [
                           {'condition': 'baseline', 'promptID': 'item-1', 'classified': 2, 'lengthStopped': 0,
                            'lengthStoppedInReasoning': 0},
                           {'condition': 'formal-strong', 'promptID': 'item-2', 'classified': 2,
                            'lengthStopped': 1, 'lengthStoppedInReasoning': 0}]}})
    _python_analysis_by_hand(root, [
        _row('formal', 'choiceRate', 4, 0.25, 0.0, 0.5, 0.0, 0.0625, 0.125),
        _row('formal-strong', 'choiceRate', 2, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5),
        _row('formal', 'wordCount', 4, 4.0, 3.5, 4.5, 0.0, 0.0625, 0.0625),
        _row('formal-strong', 'wordCount', 4, 8.0, 7.25, 8.75, 0.0, 0.0625, 0.0625),
        _row('formal', 'distinct2', 4, 0.0, 0.0, 0.0, NAN, NAN, NAN),
        _row('formal-strong', 'distinct2', 4, -0.125, -0.25, 0.0, 1.5, 0.375, NAN, correction=''),
        _row('formal', 'wordCount', 2, 4.0, 4.0, 4.0, 0.0, 0.5, NAN, correction='', stratify_by='promptID',
             stratum='item-1', unit='sample', estimand='withinItemSamples', inference='diagnostic'),
    ])
    _write_json(os.path.join(str(root), 'runs', ANALYSIS, 'exclusions.json'), {
        'excludedRecords': 1,
        'rules': [{'rule': 'unparseableEndpoint', 'description': 'A record whose declared answer could not be read.'}],
        'excludedByRule': {'baseline': {'unparseableEndpoint': 1}, 'formal': {}, 'formal-strong': {}},
        'consideredN': {'baseline': 8, 'formal': 8, 'formal-strong': 8},
        'survivingN': {'baseline': 7, 'formal': 8, 'formal-strong': 8}})


def mac_engine(root):
    """A Mac-engine run, not frozen, whose headline is marker density by default order."""
    manifest = _manifest(samplesPerItem=1, concepts=[{'name': 'formality', 'stimulusSetHash': '7' * 64}])
    _write_study(root, manifest)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    _write_run(root, manifest, _mac_records(manifest_hash), engine='mac', report={
        'experiment': STUDY, 'experimentHash': manifest_hash, 'promptCount': 4, 'conditionCount': 2,
        'conditions': {
            'baseline': {'generations': 4, 'meanWordCount': 9.25, 'meanDistinct2': 0.5,
                         'meanMarkerDensity': {'formality': 0.25}},
            'formal': {'generations': 4, 'meanWordCount': 13.25, 'meanDistinct2': 0.5,
                       'meanMarkerDensity': {'formality': 0.25}}}})
    _write_mac_analysis(root, manifest, [
        {'condition': 'formal', 'metric': 'formalityMarkerDensity', 'n': 4, 'meanDiff': 0.0, 'ciLower': -0.05,
         'ciUpper': 0.05, 'wilcoxonW': None, 'wilcoxonP': None, 'adjustedP': None, 'correction': 'bh'},
        {'condition': 'formal', 'metric': 'wordCount', 'n': 4, 'meanDiff': 4.0, 'ciLower': 4.0, 'ciUpper': 4.0,
         'wilcoxonW': 0.0, 'wilcoxonP': 0.125, 'adjustedP': 0.125, 'correction': 'bh'},
    ])


def judged(root):
    """A Python run with an analysis, paired judging (the headline), and response coding."""
    manifest = _manifest(judges=[{'name': 'strict', 'kind': 'local'}, {'name': 'lenient', 'kind': 'openrouter'}],
                         judgeRubricFile='prompts/rubrics/register.md')
    _write_study(root, manifest)
    _write_run(root, manifest, _python_records(Manifest.from_dict(manifest).content_hash()))
    _python_analysis_by_hand(root, [_row('formal', 'wordCount', 4, 4.0, 3.5, 4.5, 0.0, 0.125, 0.125)])
    _write_python_paired_evaluation(root)
    rows = []
    for judge in ('coder-a', 'coder-b'):
        for item in ITEMS:
            rows.append({'experiment': STUDY, 'condition': 'formal', 'promptID': item, 'sampleIndex': 0,
                         'seed': 1000, 'wordCount': 12, 'codes': {'polite': True, 'register': 'formal'},
                         'briefReason': 'Polite throughout.', 'judge': judge, 'judgeKind': 'local',
                         'judgeModel': 'judge/model-a'})
    directory = os.path.join(str(root), 'runs', CODING)
    _write_lines(os.path.join(directory, 'codings.jsonl'), rows)
    _write_json(os.path.join(directory, 'coding-report.json'), {
        'mode': 'perResponseCoding', 'experiment': STUDY, 'sourceRun': RUN, 'judges': ['coder-a', 'coder-b'],
        'judgeDetails': [{'name': name, 'kind': 'local', 'requestedModel': 'judge/model-a'}
                         for name in ('coder-a', 'coder-b')],
        'judgeRubricFile': 'prompts/rubrics/politeness.md', 'judgeRubricHash': 'd' * 64,
        'fields': [{'name': 'polite', 'type': 'boolean', 'optional': False},
                   {'name': 'register', 'type': 'categorical', 'optional': False, 'values': ['formal', 'casual']}],
        'codings': 8, 'conditions': {'formal': {'codedResponses': 4, 'codings': 8, 'meanWordCount': 12.0, 'fields': {
            'polite': {'n': 8, 'nulls': 0, 'trueCount': 6, 'trueShare': 0.75},
            'register': {'n': 8, 'nulls': 0, 'counts': {'casual': 2, 'formal': 6}}}}},
        'fieldAgreement': [{'field': 'polite', 'judgeA': 'coder-a', 'judgeB': 'coder-b', 'n': 4,
                            'percentAgreement': 0.75, 'kappa': 0.5}]})


def multi_agent(root):
    """A multi-agent study analyzed per transcript: two play-throughs each, too few for an interval."""
    manifest = _manifest(studyKind='multiAgent', multiAgentScenarioPath='prompts/panels/committee.json',
                         multiAgentScenarioHash='e' * 64)
    _write_study(root, manifest)
    _write_run(root, manifest, _panel_records(), report={
        'experiment': STUDY, 'conditions': {'baseline': {'generations': 6, 'meanWordCount': 13.0,
                                                         'meanDistinct2': 1.0},
                                            'formal': {'generations': 6, 'meanWordCount': 13.0,
                                                       'meanDistinct2': 1.0}},
        'unitOfAnalysis': 'transcript', 'transcriptsPerCondition': 2})
    _write_mac_analysis(root, manifest, [
        {'condition': 'formal', 'metric': 'wordCount', 'n': 2, 'meanDiff': 0.0, 'ciLower': 0.0, 'ciUpper': 0.0,
         'wilcoxonW': None, 'wilcoxonP': None, 'adjustedP': None, 'correction': None, 'unit': 'transcript'}])


def forced(root):
    """A study frozen with force, with a control arm, a capability battery, an
    exempt agent, and a measurement change the analysis recorded."""
    control = {'name': 'random', 'alphaInNormUnits': True, 'bandWidth': 1, 'controlType': 'randomMatchedNorm',
               'slots': [{'concept': 'formality', 'layer': 6, 'alpha': 4.0}]}
    manifest = _manifest(status='frozen', frozenAt='2026-10-03T12:00:00Z', freezeHash='f' * 64,
                         freezeForced=True, forcedGatesSkipped=['validateEvidence', 'gitClean'],
                         capabilityBatteryNotApplied=[{'condition': 'guided', 'reason': 'interventionPolicy'}],
                         conditions=[BASELINE, FORMAL, control])
    _write_study(root, manifest)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    battery = {'itemCount': 20, 'batteryHash': 'c' * 64}
    _write_run(root, manifest, _python_records(manifest_hash, conditions=('baseline', 'formal', 'random')), report={
        'experiment': STUDY, 'experimentHash': manifest_hash,
        'conditions': {'baseline': {'generations': 8, 'meanWordCount': 9.5, 'meanDistinct2': 0.5,
                                    'capabilityBattery': {'accuracy': 0.9, **battery}},
                       'formal': {'generations': 8, 'meanWordCount': 13.5, 'meanDistinct2': 0.5,
                                  'capabilityBattery': {'accuracy': 0.85, **battery}},
                       'random': {'generations': 8, 'meanWordCount': 9.5, 'meanDistinct2': 0.5,
                                  'capabilityBattery': {'accuracy': 0.9, **battery}}},
        'capabilityBatteryNotApplied': [{'condition': 'guided', 'reason': 'interventionPolicy'}]})
    _python_analysis_by_hand(root, [
        _row('formal', 'wordCount', 4, 4.0, 3.5, 4.5, 0.0, 0.0625, 0.125),
        _row('random', 'wordCount', 4, 0.0, -0.5, 0.5, 4.5, 0.875, 0.875)])
    _write_json(os.path.join(str(root), 'runs', ANALYSIS, 'measurement-drift.json'),
                {'measurementDrift': 'maxTokens 64 -> 96'})


CASES = {'python-engine': python_engine, 'mac-engine': mac_engine, 'judged': judged,
         'multi-agent': multi_agent, 'forced': forced}


def built(root, name):
    CASES[name](root)
    return results_export.read_results(str(root), STUDY)


def rendered(root, name):
    return study_results.render(built(root, name))


def section(text, key):
    return re.search(r'<section id="%s".*?</section>' % re.escape(key), text, re.S).group(0)


def write_goldens(base):
    """Rewrite the golden pages. ``tests/fixtures/study-report/generate.py`` calls this."""
    for name in CASES:
        workspace = Path(base) / name
        workspace.mkdir(parents=True)
        (FIXTURES / f'{name}.html').write_text(rendered(workspace, name), encoding='utf-8')


# --- the page --------------------------------------------------------------------


@pytest.mark.parametrize('name', sorted(CASES))
def test_each_fixture_study_renders_to_its_golden_page(root, name):
    golden = (FIXTURES / f'{name}.html').read_text(encoding='utf-8')
    assert rendered(root, name) == golden, (
        'The page changed for a stored study. If that is intended, run '
        'tests/fixtures/study-report/generate.py from Server/ and review the diff.')


def test_a_render_is_deterministic_within_and_across_processes(root):
    stored = built(root, 'python-engine')
    first = study_results.render(stored)
    assert first == study_results.render(results_export.read_results(str(root), STUDY))
    digest = hashlib.sha256(first.encode('utf-8')).hexdigest()
    program = ('import hashlib, sys; from steerlab_server.client import results_export; '
               'from steerlab_server.client.reports import study_results as p; '
               'print(hashlib.sha256(p.render(results_export.read_results(sys.argv[1], sys.argv[2]))'
               '.encode("utf-8")).hexdigest())')
    for seed in ('1', '2'):
        output = subprocess.run([sys.executable, '-c', program, str(root), STUDY], cwd=SERVER_DIR, check=True,
                                capture_output=True, text=True,
                                env={**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONPATH': SERVER_DIR}).stdout.strip()
        assert output == digest


def test_the_page_names_no_machine_path_date_or_version(root):
    stored = built(root, 'judged')
    text = study_results.render(stored)
    for machine in (str(root), os.path.realpath(root), os.path.expanduser('~'), tempfile.gettempdir()):
        assert machine not in text
    # The software reading the run is not part of the page (the run's own stamp is, as stored).
    assert study_results.render({**stored, 'version': '99.99.99'}) == text
    assert 'steerlab ' + stored['version'] not in text
    # The only dates are the ones the run stored.
    assert not re.search(r'20\d\d-\d\d-\d\d(?!T00:00:00Z)', text)


class Outline(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.ids, self.links, self.tables, self.pictures, self.cells = [], set(), [], [], [], []
        self._table, self._picture, self._cell = None, None, None

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        self.tags.append((tag, values))
        if 'id' in values:
            assert values['id'] not in self.ids, f'duplicate id {values["id"]}'
            self.ids.add(values['id'])
        if tag == 'a':
            self.links.append(values.get('href'))
        if tag == 'table':
            self._table = {'caption': False, 'scopes': [], 'unscoped': 0}
            self.tables.append(self._table)
        if tag == 'caption' and self._table is not None:
            self._table['caption'] = True
        if tag == 'th' and self._table is not None:
            if 'scope' in values:
                self._table['scopes'].append(values['scope'])
            else:
                self._table['unscoped'] += 1
        if tag == 'svg' and values.get('role') == 'img':
            self._picture = {'labelledby': values.get('aria-labelledby', '').split(), 'named': set()}
            self.pictures.append(self._picture)
        if tag in ('title', 'desc') and self._picture is not None and 'id' in values:
            self._picture['named'].add(values['id'])
        if tag == 'td' and 'num' in (values.get('class') or '').split():
            self._cell = []

    def handle_endtag(self, tag):
        if tag == 'table':
            self._table = None
        if tag == 'svg':
            self._picture = None
        if tag == 'td' and self._cell is not None:
            self.cells.append(''.join(self._cell))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def outline(text):
    parser = Outline()
    parser.feed(text)
    return parser


@pytest.mark.parametrize('name', sorted(CASES))
def test_the_page_reaches_nothing_outside_itself(root, name):
    text = rendered(root, name)
    found = outline(text)
    lowered = text.lower()
    for forbidden in ('http://', 'https://', '<script', '<link', '<img', '<iframe', '<object', '<embed', '@import',
                      'url(', 'src=', 'srcset=', 'javascript:', ' onclick', ' onload'):
        assert forbidden not in lowered, forbidden
    assert found.links and all(link.startswith('#') and link[1:] in found.ids for link in found.links)
    policy = [values['content'] for tag, values in found.tags if tag == 'meta' and values.get('http-equiv')]
    assert policy == ["default-src 'none'; style-src 'unsafe-inline'"]
    assert text.startswith(page.MARKER)
    for table in found.tables:
        assert table['caption'] and table['unscoped'] == 0
    for picture in found.pictures:
        assert len(picture['labelledby']) == 2 and set(picture['labelledby']) == picture['named']


def test_every_stored_effect_is_on_the_page_with_its_numbers(root):
    stored = built(root, 'python-engine')
    text = study_results.render(stored)
    table = re.search(r'<table id="effects-table">.*?</table>', text, re.S).group(0)
    rows = stored['context']['effectRows']
    pooled = [row for row in rows if row['stratify_by'] == 'pooled']
    assert table.count('<tr>') == 2 + len(pooled)    # two header rows (the interval group), then every row
    for row in pooled:
        line = re.search(r'<tr>\s*<td><code>%s</code></td>\s*<td>%s</td>.*?</tr>'
                         % (re.escape(row['outcome']), re.escape(row['condition'])), table, re.S).group(0)
        for value, formatter in ((row['estimate'], study_results.signed_text), (row['ci_lower'], study_results.number_text),
                                 (row['ci_upper'], study_results.number_text), (row['p_value'], study_results.p_text),
                                 (row['adjusted_p_value'], study_results.p_text)):
            shown = formatter(value)
            assert (f'>{shown}<' if shown else f'>{page.NO_VALUE}<') in line, (row, value)
    strata = re.search(r'<table id="strata-table">.*?</table>', text, re.S).group(0)
    assert strata.count('<tr>') == 2 + len(rows) - len(pooled)
    assert 'one item’s samples only' in strata and 'a description, not a test' in strata
    # Every file the reading recorded, by its whole hash.
    for entry in stored['sources']:
        assert entry['sha256'] in section(text, 'freeze') and entry['path'] in section(text, 'freeze')
    assert all(len(found) == 64 for found in HEX64.findall(text))
    # And everything the run did not store is listed.
    limits = section(text, 'limits')
    assert stored['notAvailable'] and all(entry['what'] in limits for entry in stored['notAvailable'])


@pytest.mark.parametrize('name, at_least', [('python-engine', 20), ('judged', 6), ('forced', 6)])
def test_no_number_on_the_page_was_computed_by_the_page(root, name, at_least):
    stored = built(root, name)
    text = study_results.render(stored)
    allowed = set()

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, str) or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            for formatter in (study_results.number_text, study_results.signed_text, study_results.p_text,
                              study_results.count_text):
                shown = formatter(value)
                if shown:
                    allowed.add(shown)
    context = stored['context']
    collect([context['effectRows'], context['report'], context['pairedReport'], context['codingReport']])
    decimal = re.compile(r'^[+-]?\d+\.\d+$')
    cells = [cell.strip() for cell in outline(text).cells]
    shown = [cell for cell in cells if decimal.match(cell)]
    assert len(shown) >= at_least
    assert set(shown) <= allowed, set(shown) - allowed


def test_the_headline_says_which_rule_chose_it(root):
    declared = rendered(root, 'python-engine')
    headline = section(declared, 'headline')
    assert ('<strong>The target-choice rate (<code>choiceRate</code>)</strong>, declared by the researcher.'
            in headline)
    # The condition with two pairs gets no interval in the sentence or the chart, and the table says why.
    assert 'across 2 paired items. That is too few pairs for an interval or a test (at least 3 are needed)' in headline
    assert 'too few pairs for an interval: 2' in headline
    assert '>Fewer than 3 pairs: no interval or test<' in headline
    assert 'still shows the interval and test the analysis stored, but they describe those items only' in headline
    # Two comparisons of one outcome were corrected together; a single one is named as such.
    assert 'Corrected p = 0.125 (Benjamini-Hochberg (false discovery rate), over 2 comparisons of this outcome).' in headline
    assert 'The interval includes zero, so this run is consistent with no difference.' in headline
    effects = section(declared, 'effects')
    assert 'id="effect-chart-1"' in effects and 'headline-chart' not in effects

    other = root.parent / 'mac'
    other.mkdir()
    default = section(rendered(other, 'mac-engine'), 'headline')
    assert "<strong>&#x27;formality&#x27; marker density (<code>formalityMarkerDensity</code>)</strong>, " \
           'chosen by default order.' in default
    assert 'No test is stored, for example because every difference was zero.' in default


def test_a_declared_outcome_the_run_lacks_is_said_first(root):
    manifest = _manifest(primaryOutcome='choiceLogOdds')
    _write_run(root, manifest, _python_records('0' * 64))
    _python_analysis_by_hand(root, [_row('formal', 'wordCount', 4, 4.0, 3.5, 4.5, 0.0, 0.125, 0.125)])
    text = study_results.render(results_export.read_results(str(root), STUDY))
    first = re.search(r'<div class="note" id="first">.*?</div>', text, re.S).group(0)
    assert ('The study declared <code>choiceLogOdds</code> as its primary outcome, and this run does not have '
            'it.') in first
    assert ("chosen by default order; the declared primary outcome &#x27;choiceLogOdds&#x27; is not in this run."
            in section(text, 'headline'))
    assert 'p = 0.125 from the one Wilcoxon signed-rank test of this outcome (a single comparison' in text


def test_a_judged_study_leads_with_the_judges_counts_and_their_agreement(root):
    text = rendered(root, 'judged')
    headline = section(text, 'headline')
    assert '<strong>The judged outcome (<code>judged</code>)</strong>, chosen by default order.' in headline
    assert 'It computed no interval and no test for it, so this page shows none.' in headline
    tally = re.search(r'<table id="headline-tally">.*?</table>', headline, re.S).group(0)
    # Per judge, as the report stores them: the first judge's 4 verdicts, the second's 3.
    assert re.search(r'<code>strict</code></td>\s*<td>formal</td>\s*<td class="num">4</td>\s*'
                     r'<td class="num">2</td>\s*<td class="num">1</td>\s*<td class="num">1</td>', tally)
    assert re.search(r'<code>lenient</code></td>\s*<td>formal</td>\s*<td class="num">3</td>', tally)
    assert 'Rubric field <code>polite</code> (boolean)' in headline and '0.75 (6 true)' in headline
    assert 'casual 2; formal 6' in headline
    judges = section(text, 'judges')
    assert re.search(r'<code>strict</code> and <code>lenient</code></td>\s*<td class="num">3</td>\s*'
                     r'<td class="num">1</td>\s*<td class="num">1</td>', judges)
    assert '8, of which 1 marked noncompliant' in judges
    assert re.search(r'<code>polite</code></td>\s*<td><code>coder-a</code> and <code>coder-b</code></td>\s*'
                     r'<td class="num">4</td>\s*<td class="num">0.75</td>\s*<td class="num">0.5</td>', judges)
    assert 'b' * 64 in judges and 'd' * 64 in judges
    # The analysis's outcome is then the first of the other outcomes.
    assert 'id="effect-chart-1"' in section(text, 'effects')


def test_a_multi_agent_study_is_described_by_its_panel_and_transcripts(root):
    text = rendered(root, 'multi-agent')
    design = section(text, 'design')
    assert '<code>prompts/panels/committee.json</code>, SHA-256 <code class="long">' + 'e' * 64 in design
    assert 'Chair (seat <code>seat-chair</code>), Member (seat <code>seat-member</code>)' in design
    assert '4, one for each condition and play-through' in design
    headline = section(text, 'headline')
    assert 'across 2 paired transcripts. That is too few pairs' in headline
    assert re.search(r'<td>transcript</td>', section(text, 'headline'))


def test_a_forced_study_says_so_first_and_names_every_skipped_check(root):
    text = rendered(root, 'forced')
    first = re.search(r'<div class="note" id="first">.*?</div>', text, re.S).group(0)
    assert text.index('id="first"') < text.index('id="headline"')
    assert 'This study was frozen with force: some freeze checks were skipped.' in first
    assert 'The capability battery was not applied to guided, because its agent uses an intervention policy' in first
    assert 'The study’s measurement settings changed after the run was made. The analysis recorded: maxTokens 64 -&gt; 96' in first
    freeze = section(text, 'freeze')
    assert ('the check that the steering directions were validated (<code>validateEvidence</code>); the check that '
            'the pinned inputs are committed (<code>gitClean</code>)') in freeze
    assert 'treats the study as not citable' in freeze
    controls = section(text, 'controls')
    assert '<strong>random</strong>: adds a random direction of the same size' in controls
    assert '(recorded as <code>randomMatchedNorm</code>)' in controls
    assert re.search(r'<td>formal</td>\s*<td class="num">0.85</td>\s*<td class="num">20</td>', controls)


def test_a_bare_run_says_what_it_lacks(root):
    directory = os.path.join(str(root), 'runs', RUN)
    _write_lines(os.path.join(directory, 'generations.jsonl'), [
        {'condition': 'baseline', 'promptID': 'item-1', 'prompt': 'Hello?', 'output': 'Hello.'}])
    _write_json(os.path.join(directory, 'report.json'), {})
    text = study_results.render(results_export.read_results(str(root), STUDY))
    assert 'This run has no outcome to lead with' in section(text, 'headline')
    assert 'No analysis of this run was found, so there are no effect estimates.' in section(text, 'effects')
    limits = section(text, 'limits')
    for what in ('the model', 'the engine', 'the freeze status', 'effects.csv'):
        assert f'<li>{what}: ' in limits, what


def test_stored_text_can_never_become_markup(root):
    hostile = '<script>alert(1)</script>"onmouseover="x'
    manifest = _manifest(experimentDescription=hostile, conditions=[BASELINE, {**FORMAL, 'name': 'formal'}])
    records = _python_records('0' * 64)
    for record in records:
        if record['condition'] == 'formal':
            record['condition'] = 'formal<b>'
    _write_run(root, manifest, records)
    text = study_results.render(results_export.read_results(str(root), STUDY))
    assert '<script' not in text and '<b>' not in text and '"onmouseover="' not in text
    assert '&lt;script&gt;alert(1)&lt;/script&gt;&quot;onmouseover=&quot;x' in text


def test_numbers_are_formatted_from_stored_values_only():
    assert study_results.number_text('4.5') == '4.5' and study_results.number_text('0.123456') == '0.1235'
    assert study_results.signed_text('4.5') == '+4.5' and study_results.signed_text('-0.0') == '0'
    assert study_results.signed_text('-0.125') == '-0.125'
    assert study_results.p_text('0.00001') == '< 0.0001' and study_results.p_text('') is None
    assert study_results.number_text('nan') is None and study_results.number_text(True) is None
    assert study_results.count_text('4') == '4' and study_results.count_text(4.0) == '4'


def test_the_forest_scale_is_round_and_includes_zero():
    from steerlab_server.client.reports import charts
    assert [str(tick) for tick in charts.ticks_between(0.0, 4.75)] == ['0', '1', '2', '3', '4', '5']
    assert [str(tick) for tick in charts.ticks_between(-0.25, 0.0)] == ['-0.25', '-0.20', '-0.15', '-0.10', '-0.05',
                                                                     '0.00']
    # A scale with no width is widened to one step either side; an unusable one falls back to -1 to 1.
    assert [float(tick) for tick in charts.ticks_between(0.0, 0.0)] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert charts.ticks_between(float('nan'), 1) == charts.ticks_between(2, 1) == [-1, 0, 1]
    empty = charts.forest_chart(key='k', title='T', rows=[charts.Estimate('a')], value_text=str, axis_name='x')
    assert 'No value is stored for this chart' in empty


# --- the verb --------------------------------------------------------------------


def listing(path):
    path = Path(path)
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in sorted(path.rglob('*')) if item.is_file()}


def cli(capsys, *arguments):
    capsys.readouterr()
    code = client_cli.main([*arguments, '--json'])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_results_report_writes_under_reports_and_leaves_runs_untouched(root, capsys):
    python_engine(root)
    before = listing(root / 'runs')
    code, document, err = cli(capsys, '--root', str(root), 'results', 'report', STUDY)
    assert code == 0, err
    assert (document['state'], document['verb'], document['changed']) == ('ready', 'results report', True)
    result = document['result']
    target = Path(os.path.realpath(root)) / 'reports' / STUDY / RUN / 'report.html'
    assert result['htmlPath'] == str(target)
    assert target.read_text(encoding='utf-8') == (FIXTURES / 'python-engine.html').read_text(encoding='utf-8')
    assert result['htmlSHA256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert (result['run'], result['analysis']) == ('runs/' + RUN, 'runs/' + ANALYSIS)
    assert result['headline']['outcome'] == 'choiceRate' and result['headline']['rule'] == 'declared'
    assert 'Results page for tone-study' in err and 'Open the file in a web browser' in err
    assert listing(root / 'runs') == before
    # Again: nothing changes, and the answer says so.
    code, again, _ = cli(capsys, '--root', str(root), 'results', 'report', STUDY)
    assert code == 0 and again['changed'] is False and 'already up to date' in again['message']
    assert sorted(path.name for path in target.parent.iterdir()) == ['report.html']


def test_the_export_carries_the_same_page(root):
    python_engine(root)
    exported = results_export.export_results(str(root), STUDY)
    written = results_report.write_report(str(root), STUDY)
    page_in_export = Path(exported['exportDirectory'], 'report.html').read_bytes()
    assert page_in_export == Path(written['htmlPath']).read_bytes()
    manifest = json.loads(Path(exported['exportDirectory'], 'manifest.json').read_text(encoding='utf-8'))
    [entry] = [entry for entry in manifest['files'] if entry['file'] == 'report.html']
    assert entry['sha256'] == hashlib.sha256(page_in_export).hexdigest() == written['htmlSHA256']
    assert 'report.html' in Path(exported['exportDirectory'], 'codebook.md').read_text(encoding='utf-8')


def test_out_names_the_page_file(root, tmp_path, capsys, monkeypatch):
    mac_engine(root)
    monkeypatch.chdir(tmp_path)
    code, document, _ = cli(capsys, '--root', str(root), 'results', 'report', STUDY, '--out', 'for-a-colleague.html')
    assert code == 0
    assert os.path.realpath(document['result']['htmlPath']) == os.path.realpath(tmp_path / 'for-a-colleague.html')
    assert (tmp_path / 'for-a-colleague.html').read_text(encoding='utf-8').startswith(page.MARKER)
    assert not (root / 'reports').exists()


@pytest.mark.parametrize('where', ['runs/page.html', f'runs/{RUN}/report.html', 'runs/new-folder/page.html'])
def test_a_page_is_never_written_into_runs(root, capsys, where):
    python_engine(root)
    before = listing(root)
    code, document, _ = cli(capsys, '--root', str(root), 'results', 'report', STUDY, '--out', str(root / where))
    assert (code, document['state'], document['error']['code']) == (65, 'refused', 'reportDestinationRefused')
    assert 'never written inside runs/' in document['error']['reason']
    assert 'reports/' in document['error']['repairAction']
    assert listing(root) == before


def test_a_completed_run_elsewhere_and_a_file_that_is_not_a_page_are_protected(root):
    python_engine(root)
    kept = root / 'archive' / 'old-run'
    kept.mkdir(parents=True)
    (kept / 'COMPLETED').write_text('run\n')
    notes = root / 'notes.html'
    notes.write_text('<!doctype html>\n<p>My own notes.</p>\n')
    for out, reason in ((kept / 'page.html', 'completed run folder'), (notes, 'not a report page')):
        with pytest.raises(results_export.ResultsExportRefusal) as refusal:
            results_report.write_report(str(root), STUDY, out=str(out))
        assert refusal.value.code == 'reportDestinationRefused' and reason in refusal.value.reason
    assert notes.read_text() == '<!doctype html>\n<p>My own notes.</p>\n'
    assert listing(kept) == {'COMPLETED': b'run\n'}
    # An earlier page of its own is replaced.
    stale = root / 'stale.html'
    stale.write_text(page.MARKER + '<p>An older page.</p>\n')
    assert results_report.write_report(str(root), STUDY, out=str(stale))['changed'] is True
    assert stale.read_text(encoding='utf-8') == (FIXTURES / 'python-engine.html').read_text(encoding='utf-8')


def test_command_line_refusals_are_typed_in_this_clients_words(root, capsys):
    code, document, _ = cli(capsys, '--root', str(root), 'results', 'report', STUDY)
    assert (code, document['state'], document['error']['code']) == (66, 'notFound', 'notFound')
    _write_study(root, _manifest())
    code, document, _ = cli(capsys, '--root', str(root), 'results', 'report', STUDY)
    assert (code, document['error']['code']) == (65, results_export.NO_COMPLETED_RUN_CODE)
    assert document['error']['repairAction'].startswith(f'steerlab run {STUDY} --runner <url>')
    code, document, _ = cli(capsys, '--root', str(root), 'results', 'report')
    assert (code, document['error']['code']) == (64, 'usage')
    assert document['error']['reason'] == 'Name one study to make a results page for.'
    assert not (root / 'reports').exists()


def test_the_verb_is_declared_with_its_own_out_flag(capsys):
    spec = client_cli.spec_for('results', 'report')
    assert spec.value_flags == {'--run', '--out'}
    assert client_cli.synopsis(spec) == 'steerlab results report <study> [--out <file>] [--run <run-dir>]'
    assert 'reports/' in spec.purpose and 'recalculates nothing' in spec.purpose
    assert client_cli.main(['results', '--help']) == 0
    assert 'steerlab results report <study>' in capsys.readouterr().out


# --- the bridge the Mac command line and the app use --------------------------------


def test_the_bridge_action_writes_the_same_page(root):
    python_engine(root)
    result = diagnostic_commands.workspace_action(
        results_commands.REPORT_BRIDGE_ACTION, {'workspaceRoot': str(root), 'study': STUDY, 'client': 'app'})
    assert result['written'] is True and result['changed'] is True
    assert Path(result['htmlPath']).read_text(encoding='utf-8') == (
        FIXTURES / 'python-engine.html').read_text(encoding='utf-8')


def test_the_bridge_returns_a_refusal_as_data_in_the_asking_clients_words(root):
    _write_study(root, _manifest())
    mac = diagnostic_commands.workspace_action(
        results_commands.REPORT_BRIDGE_ACTION, {'workspaceRoot': str(root), 'study': STUDY})
    assert (mac['written'], mac['changed'], mac['refusal']['code'], mac['refusal']['state']) == (
        False, False, results_export.NO_COMPLETED_RUN_CODE, 'refused')
    assert mac['refusal']['repairAction'].startswith(f'steerlab-cli experiment run {STUDY}, then ')
    app = diagnostic_commands.workspace_action(
        results_commands.REPORT_BRIDGE_ACTION, {'workspaceRoot': str(root), 'study': STUDY, 'client': 'app'})
    assert app['refusal']['repairAction'].startswith('Run the study, then ')
    with pytest.raises(ValueError):
        diagnostic_commands.workspace_action(
            results_commands.REPORT_BRIDGE_ACTION, {'workspaceRoot': str(root), 'study': STUDY, 'force': 'yes'})


def test_the_bridge_process_writes_the_page_end_to_end(root):
    """The exact process the Mac side starts: JSON on stdin, one JSON answer on stdout."""
    from steerlab_server.client.runtime_identity import source_sha256
    python_engine(root)
    before = listing(root / 'runs')
    identity = source_sha256()
    request = {'action': results_commands.REPORT_BRIDGE_ACTION, 'clientSHA256': identity,
               'payload': {'workspaceRoot': str(root), 'study': STUDY, 'client': 'steerlab-cli'}}
    process = subprocess.run(
        [sys.executable, '-B', '-s', '-m', 'steerlab_server.client.diagnostic_workspace'],
        input=json.dumps(request), capture_output=True, text=True, check=False,
        cwd=str(root), env={**os.environ, 'PYTHONPATH': SERVER_DIR})
    assert process.returncode == 0, process.stderr
    answer = json.loads(process.stdout)
    assert answer['ok'] is True and answer['result']['written'] is True
    assert Path(answer['result']['htmlPath']).read_text(encoding='utf-8') == (
        FIXTURES / 'python-engine.html').read_text(encoding='utf-8')
    assert listing(root / 'runs') == before
