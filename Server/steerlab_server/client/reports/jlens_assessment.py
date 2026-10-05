"""The J-lens assessment page: the stored comparisons, laid out for a reader.

Input is the bytes of an ``assessment-report.json`` written by
``experiment.jlens_assessment`` (``schemaVersion`` 1, the single form, or 2,
the list form with one entry per candidate and corpus). Output is one HTML
page. The same bytes always give the same page.

Nothing here computes a statistic. The page prints stored per-layer values and
says, layer by layer, which stored value is larger. It never averages across
layers, never subtracts one reading from another, and never attaches an
interval or a test the engine did not compute. Where a value is missing the
page says so, and its last section lists everything that was left out.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from . import charts
from .page import (Column, block, bullets, code, contents, document, el, facts, join, link, note, paragraph,
                   table)

OPERATION = 'jlens-fit-assess'
REPORT_NAME, PAGE_NAME = 'assessment-report.json', 'assessment-report.html'

OVERLAP, DIVERGENCE = 'meanTopKOverlap', 'meanJSDivergence'
#: Stored key, name on the page, and whether a larger value means a closer match.
MEASURES = ((OVERLAP, 'Top-k overlap', True), (DIVERGENCE, 'Jensen–Shannon divergence', False))
IN_SENTENCE = {OVERLAP: 'top-k overlap', DIVERGENCE: 'Jensen–Shannon divergence'}
FLOAT32_GROUPS = (
    ('betweenLenses', 'Candidate and reference'), ('referenceToFinal', 'Reference and final'),
    ('candidateToFinal', 'Candidate and final'), ('logitLensToFinal', 'Baseline and final'),
    ('referenceToNativeReadout', 'Reference: float32 and native'),
    ('candidateToNativeReadout', 'Candidate: float32 and native'),
    ('logitLensToNativeReadout', 'Baseline: float32 and native'),
    ('finalToNativeReadout', 'Final: float32 and native'))
MATRIX_FIELDS = (('relativeFrobenius', 'Relative difference'), ('cosine', 'Cosine'),
                 ('maxAbs', 'Largest single difference'), ('referenceNorm', 'Reference size'),
                 ('candidateNorm', 'Candidate size'), ('differenceNorm', 'Difference size'))
HELD_OUT = {
    'sameCorpusAsFit': 'Not held out. At least one of the two lenses was fitted on this same text, so agreement '
                       'here can look better than it would on new text.',
    'researcherDeclared; overlapNotEstablished':
        'Declared held out by the researcher. The engine checked only that this file is not the lenses’ fitting '
        'text. It did not look for passages the two share.',
}
#: Stored in the JSON and deliberately not drawn; the Limits section names each.
NOT_SHOWN = ('the token positions assessed in each row', 'the sums behind each stored mean',
             'memory and storage budgets', 'the tensor precisions observed at each layer',
             'lens record fields other than those under Lenses')


class ReportError(ValueError):
    """The bytes are not an assessment report this page can draw."""


def _dict(value): return value if isinstance(value, dict) else {}
def _list(value): return value if isinstance(value, list) else []
def _text(value): return value if isinstance(value, str) and value else None


def _number(value):
    """A stored finite number, or ``None``. ``true`` and ``false`` are not numbers."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def mean_text(value):
    """A stored mean to four decimal places, as every table and chart on the page prints it."""
    number = _number(value)
    if number is None: return None
    return f'{number:.2e}' if number != 0 and abs(number) < 0.00005 else f'{number:.4f}'


def _count(value):
    number = _number(value)
    return None if number is None else f'{number:,}' if isinstance(number, int) else f'{number:,.0f}'


def _general(value):
    number = _number(value)
    return None if number is None else f'{number:.6g}'


def _layer_order(keys):
    return sorted(keys, key=lambda key: (0, int(key), '') if key.isdigit() else (1, 0, key))


def _layer_runs(layers):
    """``['0', '1', '2', '5']`` as ``0–2, 5``; a run is consecutive whole numbers only."""
    out, run = [], []
    for key in [*layers, None]:
        if run and key is not None and key.isdigit() and run[-1].isdigit() and int(key) == int(run[-1]) + 1:
            run.append(key)
            continue
        if run: out.append(run[0] if len(run) == 1 else f'{run[0]}–{run[-1]}')
        run = [] if key is None else [key]
    return ', '.join(out)


def _plural(count, noun):
    return f'{count:,} {noun}' if count == 1 else f'{count:,} {noun}s'


def _lenses(count):
    return f'{count:,} candidate lens' if count == 1 else f'{count:,} candidate lenses'


def _where(layers, total):
    """How many of the compared layers, and which: the plain-language unit of this page."""
    if not layers: return 'at no layer'
    if len(layers) == total: return f'at all {total:,} layers' if total != 1 else 'at the one layer compared'
    return f'at {len(layers):,} of {total:,} layers ({_layer_runs(layers)})'


def _lens(value):
    return code(value) if _text(value) else 'a lens with no recorded ID'


def _file(corpus):
    return code(corpus['path']) if _text(corpus.get('path')) else 'a text with no recorded path'


def _hash(value):
    return code(value) if _text(value) else None


def _held_out(status):
    if not _text(status): return None
    return join(HELD_OUT.get(status, 'A recorded status this page cannot explain.'), ' Recorded as ', code(status), '.')


def pair_key(corpus, candidate):
    """What makes a comparison this one and no other: the text's hash and path, and the candidate."""
    return (_text(corpus.get('sha256')), _text(corpus.get('path')), _text(candidate))


@dataclass(frozen=True)
class Reading:
    """One line of a chart and one column of its table."""
    label: str       # plain words, for a legend and a chart's description
    heading: object  # the same as markup, for a table header
    phrase: object   # the same again, as it reads inside a sentence
    style: object
    values: tuple


@dataclass
class Comparison:
    """One candidate against the reference on one text, exactly as stored."""
    number: int
    raw: dict

    @property
    def key(self): return f'c{self.number}'
    @property
    def ref(self): return link(self.key, str(self.number))
    @property
    def candidate(self): return self.raw.get('candidateLensID')
    @property
    def reference(self): return self.raw.get('referenceLensID')
    @property
    def corpus(self): return _dict(self.raw.get('corpus'))
    @property
    def pair(self): return pair_key(self.corpus, self.candidate)
    @property
    def layers(self): return _dict(self.raw.get('layers'))
    @property
    def readout(self): return _dict(_dict(self.raw.get('readoutComparison')).get('layers'))
    @property
    def precision(self): return _dict(_dict(self.raw.get('readoutComparison')).get('precision'))
    @property
    def rows(self): return [row for row in _list(self.raw.get('rows')) if isinstance(row, dict)]
    @property
    def assessed(self): return [row for row in self.rows if row.get('status') == 'assessed']
    @property
    def skipped(self): return [row for row in self.rows if row.get('status') != 'assessed']

    def group(self, layer, name):
        """One stored group of a layer. The plain baseline lives beside the native readout."""
        if name == 'baseline':
            return _dict(_dict(_dict(self.readout.get(layer)).get('native')).get('logitLensToFinal'))
        return _dict(_dict(self.layers.get(layer)).get(name))

    def value(self, layer, name, measure):
        return _number(self.group(layer, name).get(measure))

    def _stores(self, name):
        return any(isinstance(_dict(entry).get(name), dict) for entry in self.readout.values())

    @property
    def has_baseline(self):
        return any('logitLensToFinal' in _dict(_dict(entry).get('native')) for entry in self.readout.values())
    @property
    def has_float32(self): return self._stores('float32')
    @property
    def has_matrices(self): return self._stores('matrixComparison')

    def digest_state(self):
        """Whether the stored digest still describes the stored entry; ``None`` when none is stored."""
        stored = self.raw.get('comparisonSHA256')
        if not isinstance(stored, str): return None
        try:
            body = json.dumps({key: value for key, value in self.raw.items() if key != 'comparisonSHA256'},
                              sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        except ValueError:
            return False
        return hashlib.sha256(body).hexdigest() == stored


def load(data):
    try:
        report = json.loads(data)
    except ValueError as exc:
        raise ReportError('This file is not readable JSON, so no page was made.') from exc
    if not isinstance(report, dict) or report.get('operation') != OPERATION:
        raise ReportError('This file is not a J-lens assessment report (its operation is not jlens-fit-assess).')
    if report.get('schemaVersion') not in (1, 2):
        raise ReportError('This assessment report has a schema version this page does not know. '
                          'Update SteerLab, then make the page again.')
    return report


def comparisons_of(report):
    """Every stored comparison, in report order, and the positions of entries that could not be read.

    A single-form report written before comparisons were listed keeps its one
    comparison at the top level; it is read from there, with no digest.
    """
    config = _dict(report.get('config'))
    listed = report.get('comparisons')
    if isinstance(listed, list):
        entries = listed
    elif report.get('schemaVersion') == 1 and isinstance(report.get('layers'), dict):
        entries = [{'referenceLensID': config.get('referenceLensID'), 'candidateLensID': config.get('candidateLensID'),
                    'corpus': config.get('corpus'),
                    **{key: report[key] for key in ('layers', 'rows', 'heldOutStatus', 'readoutComparison', 'resources')
                       if key in report}}]
    else:
        entries = []
    found, unreadable = [], []
    for position, entry in enumerate(entries, 1):
        if isinstance(entry, dict) and isinstance(entry.get('layers'), dict):
            found.append(Comparison(len(found) + 1, entry))
        else:
            unreadable.append(position)
    return found, unreadable


def requested_pairs(report):
    """The text and candidate pairs the request named, text first: the order the engine works in."""
    config = _dict(report.get('config'))
    candidates = config['candidateLensIDs'] if isinstance(config.get('candidateLensIDs'), list) \
        else [config['candidateLensID']] if 'candidateLensID' in config else []
    corpora = config['corpora'] if isinstance(config.get('corpora'), list) \
        else [config['corpus']] if 'corpus' in config else []
    return [(_dict(corpus), candidate) for corpus in corpora for candidate in candidates]


def tally(comparison, against, measure, larger_is_closer):
    """Layers where the candidate's stored value is closer, further, or the same. No arithmetic on the values."""
    result = {'closer': [], 'further': [], 'same': [], 'missing': []}
    for layer in _layer_order(comparison.layers):
        mine = comparison.value(layer, 'candidateToFinal', measure)
        other = comparison.value(layer, against, measure)
        if mine is None or other is None: standing = 'missing'
        elif mine == other: standing = 'same'
        else: standing = 'closer' if (mine > other) == larger_is_closer else 'further'
        result[standing].append(layer)
    return result


class Page:
    """Everything the page says, gathered once so the last section can list what was left out."""

    def __init__(self, data):
        self.report = load(data)
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.config = _dict(self.report.get('config'))
        self.comparisons, self.unreadable = comparisons_of(self.report)
        self.lenses = [lens for lens in _list(self.report.get('lenses')) if isinstance(lens, dict)]
        self.reference = self.config.get('referenceLensID')
        self.candidates = []
        for candidate in [*[name for _, name in requested_pairs(self.report)],
                          *[comparison.candidate for comparison in self.comparisons]]:
            if candidate not in self.candidates: self.candidates.append(candidate)
        #: Comparisons grouped by text, in the order the report lists them.
        self.texts = []
        for comparison in self.comparisons:
            group = next((group for group in self.texts if group[0].pair[:2] == comparison.pair[:2]), None)
            if group is None: self.texts.append([comparison])
            else: group.append(comparison)
        self.has_baseline = any(comparison.has_baseline for comparison in self.comparisons)
        self.limits = []

    def role(self, lens_id):
        return 'Reference' if lens_id == self.reference else 'Candidate' if lens_id in self.candidates else 'Listed only'

    def top_k(self):
        stored = sorted({group['effectiveTopK'] for comparison in self.comparisons
                         for layer in comparison.layers.values() for group in _dict(layer).values()
                         if isinstance(group, dict) and _number(group.get('effectiveTopK')) is not None})
        asked = _number(self.config.get('topK'))
        if stored and stored != [asked]:
            return join(_count(asked) or 'not recorded', ' requested; the engine recorded an effective k of ',
                        ', '.join(_count(value) for value in stored))
        return _count(asked)

    def rows_sentence(self, comparison):
        if not comparison.rows: return 'The report lists no rows for this comparison.'
        sentence = f'{len(comparison.assessed):,} of {_plural(len(comparison.rows), "row")} assessed'
        return sentence + (f'; {len(comparison.skipped):,} skipped.' if comparison.skipped else '.')

    def readings(self, group, measure):
        """One text's layers, and its readings in page order: baseline, reference, then each candidate.

        The baseline and reference are stored again with every candidate. The
        copies stored with the first candidate are the ones shown.
        """
        first = group[0]
        layers = _layer_order({layer for comparison in group for layer in comparison.layers})
        def values(source, name): return tuple(source.value(layer, name, measure) for layer in layers)
        found = []
        if first.has_baseline:
            found.append(Reading('Plain baseline', 'Plain baseline', 'the plain baseline', charts.BASELINE,
                                 values(first, 'baseline')))
        found.append(Reading(f'Reference {_text(first.reference) or "lens"}', join('Reference ', _lens(first.reference)),
                             join('reference ', _lens(first.reference)), charts.REFERENCE, values(first, 'referenceToFinal')))
        # A candidate keeps one colour on every chart of the page.
        found += [Reading(f'Candidate {_text(comparison.candidate) or "lens"}', join('Candidate ', _lens(comparison.candidate)),
                          join('candidate ', _lens(comparison.candidate)),
                          self.candidates.index(comparison.candidate) % charts.SLOTS, values(comparison, 'candidateToFinal'))
                  for comparison in group]
        return layers, found

    def closest(self, group, measure, larger_is_closer):
        """Which reading holds the closest stored value at each layer, as phrases for one sentence."""
        layers, readings = self.readings(group, measure)
        won, shared = [[] for _ in readings], []
        for index, layer in enumerate(layers):
            stored = [(reading.values[index], position) for position, reading in enumerate(readings)
                      if reading.values[index] is not None]
            if not stored: continue
            best = (max if larger_is_closer else min)(value for value, _ in stored)
            holders = [position for value, position in stored if value == best]
            (won[holders[0]] if len(holders) == 1 else shared).append(layer)
        phrases = [join(reading.phrase, f' at {_plural(len(layers_won), "layer")} ({_layer_runs(layers_won)})')
                   for reading, layers_won in zip(readings, won) if layers_won]
        if shared:
            phrases.append(f'two or more readings share the closest value at {_plural(len(shared), "layer")} '
                           f'({_layer_runs(shared)})')
        return phrases

    # -- sections -------------------------------------------------------------

    def summary(self):
        model = self.config.get('modelID')
        compared = [
            join('Model ', _hash(model) or 'not recorded', ' at revision ', _hash(self.config.get('revision')) or 'not recorded', '.'),
            join('Reference lens: ', _lens(self.reference), '.'),
            join(f'Candidate {"lens" if len(self.candidates) == 1 else "lenses"} ({len(self.candidates):,}): ',
                 join(*[_lens(candidate) for candidate in self.candidates], sep=', '), '.'),
            join(f'{"Text" if len(self.texts) == 1 else "Texts"} ({len(self.texts):,}): ',
                 join(*[join(_file(group[0].corpus), ' (', self.rows_sentence(group[0]).rstrip('.'), ')')
                        for group in self.texts], sep=', ') if self.texts else 'none recorded', '.'),
            'Plain baseline: each layer read directly, with no lens, at the same token positions.'
            if self.has_baseline else 'Plain baseline: none is recorded in this report.',
        ]
        parts = [
            paragraph('A J-lens translates a model’s internal state at one layer into the tokens (word pieces) '
                      'the model is leaning toward. This report asks how well each candidate lens reads the '
                      'model’s own final prediction, layer by layer, next to a reference lens and next to '
                      'reading the layer with no lens at all.'),
            el('h3', 'What was compared'), bullets(compared),
            el('h3', 'Result, text by text'),
            paragraph('“More closely” below means a higher top-k overlap with the model’s final prediction, '
                      'the main measure here. Both measures are explained at the end of this summary.'),
        ]
        reading = [
            el('h3', 'How to read the numbers'),
            paragraph(el('strong', 'Top-k overlap'), ' is the main measure. Take the k tokens the model finally '
                      'ranks highest at a position, and the k tokens a reading ranks highest at the same '
                      'position. The overlap is the share they have in common, from 0 (none) to 1 (the same k '
                      'tokens). Higher means the reading is closer to the final prediction. k is ',
                      self.top_k() or 'not recorded', '.'),
            paragraph(el('strong', 'Jensen–Shannon divergence'), ' compares the two full sets of token '
                      'probabilities, in nats. 0 means identical, and lower means closer. The engine averaged '
                      'both measures over the assessed token positions of a text.'),
        ]
        if not self.comparisons:
            parts.append(paragraph('This report holds no comparison that can be read, so there is no result to show.'))
        rows = []
        for group in self.texts:
            parts.append(el('h4', 'On ', _file(group[0].corpus)))
            sentences = []
            for comparison in group:
                reference = tally(comparison, 'referenceToFinal', OVERLAP, True)
                baseline = tally(comparison, 'baseline', OVERLAP, True)
                compared_layers = len(comparison.layers) - len(reference['missing'])
                if not compared_layers:
                    sentences.append(join('Candidate ', _lens(comparison.candidate), ' has no stored value on this '
                                          'text, so it cannot be placed. See ', link('limits', 'Limits'), '.'))
                else:
                    sentence = ['Candidate ', _lens(comparison.candidate), ' matched the model’s final prediction '
                                'more closely than the reference ', _where(reference['closer'], compared_layers),
                                ', less closely ', _where(reference['further'], compared_layers)]
                    if reference['same']:
                        sentence += [', and equally ', _where(reference['same'], compared_layers)]
                    sentence.append('.')
                    with_baseline = len(comparison.layers) - len(baseline['missing'])
                    if with_baseline:
                        sentence += [' It matched more closely than the plain baseline ',
                                     _where(baseline['closer'], with_baseline), '.']
                    if comparison.raw.get('heldOutStatus') == 'sameCorpusAsFit':
                        sentence.append(' This text is not held out for this pair of lenses.')
                    sentences.append(join(*sentence))
                cells = [link(comparison.key, _file(comparison.corpus)), _lens(comparison.candidate)]
                for against in ('referenceToFinal', 'baseline'):
                    for measure, _, larger in MEASURES:
                        counted = tally(comparison, against, measure, larger)
                        total = len(comparison.layers) - len(counted['missing'])
                        cells.append(f'{len(counted["closer"]):,} of {total:,}' if total else None)
                rows.append(cells)
            parts.append(bullets(sentences))
            leaders = self.closest(group, OVERLAP, True)
            if leaders:
                parts.append(paragraph('Closest reading at each layer, by top-k overlap: ', join(*leaders, sep='; '), '.'))
        if rows:
            parts.append(table(
                'Layers where each candidate is the closer reading',
                [Column('Text'), Column('Candidate lens'),
                 *[Column(name, numeric=True, group=heading)
                   for heading in ('Closer than the reference', 'Closer than the plain baseline')
                   for _, name, _ in MEASURES]],
                rows, key='summary-table', row_header=False,
                footnote='Each count is the number of layers where the candidate’s stored value is the closer one, '
                         'out of the layers where both values are stored.'))
        parts.append(note('These are layer-by-layer comparisons of stored values. The engine computed no average '
                          'across layers, no interval, and no significance test, so this page shows none. A small '
                          'difference at one layer may not be a meaningful one.', label='Read with care'))
        if self.report.get('qualification') == 'notPerformed':
            parts.append(paragraph('This assessment does not qualify a lens. Whether a lens is good enough for a '
                                   'study remains the researcher’s decision.'))
        return block('Summary', *parts, *reading, key='summary')

    def identity(self):
        lens_rows = [[link(f'lens-{number}', _lens(lens.get('lensID'))), self.role(lens.get('lensID')),
                      _hash(_dict(lens.get('converted')).get('sha256')), _hash(_dict(lens.get('source')).get('tensorSHA256')),
                      _hash(lens.get('fitReportSHA256'))] for number, lens in enumerate(self.lenses, 1)]
        recorded = [lens.get('lensID') for lens in self.lenses]
        for name in [self.reference, *self.candidates]:
            if name not in recorded:
                lens_rows.append([_lens(name), self.role(name), None, None, None])
                self.limits.append(join('The report holds no lens record for ', _lens(name),
                                        ', so its hashes and fitting text cannot be shown.'))
        comparison_rows = []
        for comparison in self.comparisons:
            state = comparison.digest_state()
            if state is False:
                self.limits.append(join('The stored digest of comparison ', comparison.ref, ' does not match its stored '
                                        'content. The entry may have been changed after the engine wrote it.'))
            comparison_rows.append([
                comparison.ref, _file(comparison.corpus), _hash(comparison.corpus.get('sha256')), _lens(comparison.candidate),
                _hash(comparison.raw.get('comparisonSHA256')),
                None if state is None else 'matches the stored entry' if state else 'does not match the stored entry',
                _count(len(comparison.assessed)), _count(len(comparison.skipped))])
        baseline = ('The plain baseline, also called the logit lens: the layer’s own state is read through the '
                    'model’s final normalization and output layer with no lens applied, at the same token '
                    'positions as the lenses. It is stored for every layer of every comparison.'
                    if self.has_baseline else
                    'No baseline is recorded. This report was written before the plain baseline was added.')
        precisions = [comparison.precision for comparison in self.comparisons if comparison.precision]
        shared = precisions[0] if precisions and all(entry == precisions[0] for entry in precisions) else {}
        precision = [('Readout precision requested', code(_text(self.config.get('readoutDtype')) or 'native'))]
        if shared:
            precision += [
                ('Output head precision', _hash(shared.get('nativeHeadDtype'))),
                ('Final normalization precision',
                 join(*[code(item) for item in _list(shared.get('nativeNormParameterDtypes'))], sep=', ') or None),
                ('Precision of the lens step', _hash(shared.get('transportTensorDtype'))),
                ('Float32 readout', _text(shared.get('float32Readout')) or 'not used'),
                ('Extra readout parameter bytes', _count(shared.get('additionalReadoutParameterBytes'))),
                ('What every reading is compared with', _text(shared.get('target'))),
                ('Recorded limits of this precision', _text(shared.get('limitations'))),
            ]
        else:
            precision.append(('Recorded precision', 'It differs between comparisons; each comparison shows its own.'
                              if precisions else None))
        layers = self.config.get('sourceLayers')
        settings = [
            ('Rows read per text, at most', _count(self.config.get('maxPrompts'))),
            ('Tokens read per row, at most', _count(self.config.get('maxSeqLen'))),
            ('Leading tokens skipped in each row', _count(self.config.get('skipFirst'))),
            ('Token positions assessed per row, at most', _count(self.config.get('maxPositionsPerRow'))),
            ('k for top-k overlap', self.top_k()),
            ('Layers requested', 'every layer the lenses share' if layers is None
             else _layer_runs(_layer_order([str(layer) for layer in _list(layers)])) or None),
            ('Model precision', _hash(self.config.get('dtype'))),
            ('Device', _hash(self.config.get('device'))),
        ]
        runtime = _dict(self.report.get('runtime'))
        return block(
            'Identity',
            facts([('Report file SHA-256', code(self.sha256)), ('Report schema version', self.report.get('schemaVersion')),
                   ('Operation', code(OPERATION)), ('Model', _hash(self.config.get('modelID'))),
                   ('Model revision', _hash(self.config.get('revision')))]),
            table('Lenses', [Column('Lens ID'), Column('Role'), Column('Converted tensor SHA-256'),
                             Column('Source tensor SHA-256'), Column('Fit report SHA-256')],
                  lens_rows, key='identity-lenses',
                  footnote='The converted tensor is the file the assessment read. The fitting text and row counts '
                           'of each lens are under Lenses, further down.') if lens_rows else None,
            table('Comparisons', [Column('No.', numeric=True), Column('Text'), Column('Text SHA-256'),
                                  Column('Candidate lens'), Column('Comparison SHA-256'), Column('Digest check'),
                                  Column('Rows assessed', numeric=True), Column('Rows skipped', numeric=True)],
                  comparison_rows, key='identity-comparisons',
                  footnote='Every comparison uses the one reference lens named above. The comparison SHA-256 is the '
                           'engine’s digest of that entry. This page recomputes it only to check that the entry is '
                           'unchanged.') if comparison_rows else None,
            el('h3', 'Baseline'), paragraph(baseline),
            el('h3', 'Readout precision'), facts(precision),
            el('h3', 'Settings'), facts(settings),
            *([el('h3', 'Runtime recorded by the engine'),
               table('Runtime', [Column('Field'), Column('Recorded value')],
                     [[code(key), code(value if isinstance(value, str) else json.dumps(value, sort_keys=True))]
                      for key, value in sorted(runtime.items())], key='identity-runtime')] if runtime else []),
            key='identity')

    def text_section(self, number, group):
        first, key = group[0], f't{number}'
        parts = [facts([('File', _file(first.corpus)), ('SHA-256', _hash(first.corpus.get('sha256'))),
                        ('Rows', self.rows_sentence(first)),
                        ('Comparisons', join(*[link(comparison.key, join('candidate ', _lens(comparison.candidate)))
                                               for comparison in group], sep=', '))])]
        for comparison in group[1:]:
            if [row.get('id') for row in comparison.assessed] != [row.get('id') for row in first.assessed]:
                self.limits.append(join('On ', _file(first.corpus), ', comparison ', comparison.ref,
                                        ' lists different assessed rows from comparison ', first.ref, '.'))
        # Candidates on one text share its rows, so the same skipped rows are reported once.
        skipped = {}
        for comparison in group:
            if comparison.skipped:
                rows = tuple((str(row.get('id')), str(row.get('status'))) for row in comparison.skipped)
                skipped.setdefault((rows, len(comparison.rows)), []).append(comparison)
        for (rows, total), members in skipped.items():
            self.limits.append(join(
                'On ', _file(first.corpus), f', {len(rows):,} of {_plural(total, "row")} ',
                'was' if len(rows) == 1 else 'were', ' skipped (recorded as ',
                join(*[code(kind) for kind in sorted({kind for _, kind in rows})], sep=', '), ') in ',
                'comparison ' if len(members) == 1 else 'comparisons ', join(*[member.ref for member in members], sep=', '),
                '. Row IDs: ', join(*[code(row) for row, _ in rows], sep=', '), '.'))
        for measure, name, larger in MEASURES:
            layers, readings = self.readings(group, measure)
            slug = 'overlap' if measure == OVERLAP else 'divergence'
            direction = 'higher is closer' if larger else 'lower is closer'
            for copy, label in (('referenceToFinal', 'reference'), ('baseline', 'plain baseline')):
                differing = [comparison for comparison in group[1:]
                             if any(comparison.value(layer, copy, measure) != first.value(layer, copy, measure) for layer in layers)]
                if differing:
                    self.limits.append(join(
                        'On ', _file(first.corpus), ', the ', label, ' values (', IN_SENTENCE[measure], ') stored with ',
                        join(*[join('comparison ', comparison.ref) for comparison in differing], sep=', '),
                        ' differ from those stored with comparison ', first.ref, '. The chart and table show the copy '
                        'stored with comparison ', str(first.number), '. This page does not judge the size of the difference.'))
            fixed = [reading for reading in readings if reading.style in (charts.BASELINE, charts.REFERENCE)]
            coloured = [reading for reading in readings if reading not in fixed]
            panels = [coloured[start:start + charts.SLOTS] for start in range(0, len(coloured), charts.SLOTS)] or [[]]
            for panel_number, panel in enumerate(panels, 1):
                start = (panel_number - 1) * charts.SLOTS
                parts.append(charts.line_chart(
                    key=f'{key}-{slug}-chart' + (f'-{panel_number}' if len(panels) > 1 else ''),
                    title=f'{name} with the final prediction, by layer ({direction})' + (
                        f', candidates {start + 1} to {start + len(panel)}' if len(panels) > 1 else ''),
                    series=[charts.Series(reading.label, tuple((int(layer), value) for layer, value in zip(layers, reading.values)
                                                               if layer.isdigit()), reading.style)
                            for reading in [*fixed, *panel]],
                    x_name='Layer', value_text=mean_text, y_max=1 if measure == OVERLAP else None,
                    summary='Every value is in the table that follows.'))
            rows = []
            for index, layer in enumerate(layers):
                stored = [reading.values[index] for reading in readings if reading.values[index] is not None]
                best = (max if larger else min)(stored) if len(stored) > 1 else None
                positions = sorted({count for group_name in ('referenceToFinal', 'candidateToFinal', 'baseline')
                                    if (count := _number(first.group(layer, group_name).get('positions'))) is not None})
                rows.append([layer, ' / '.join(_count(count) for count in positions) or None,
                             *[None if reading.values[index] is None else
                               el('strong', mean_text(reading.values[index])) if reading.values[index] == best
                               else mean_text(reading.values[index]) for reading in readings]])
            parts.append(table(
                join(name, ' with the final prediction, by layer, on ', _file(first.corpus)),
                [Column('Layer', numeric=True), Column('Token positions', numeric=True),
                 *[Column(reading.heading, numeric=True) for reading in readings]],
                rows, key=f'{key}-{slug}',
                footnote=join('The closest value in each row is in bold (', direction, '). The reference',
                              ' and baseline' if first.has_baseline else '', ' columns are the copies stored with comparison ',
                              first.ref, '. Values are rounded to four decimal places; the JSON holds them in full.')))
        parts += [self.comparison_section(comparison) for comparison in group]
        return block(join('Text ', str(number), ': ', _file(first.corpus)), *parts, key=key)

    def comparison_section(self, comparison):
        layers = _layer_order(comparison.layers)
        named = join('Comparison ', comparison.ref, ' (candidate ', _lens(comparison.candidate), ' on ', _file(comparison.corpus), ')')
        empty = [layer for layer in layers if comparison.value(layer, 'candidateToFinal', OVERLAP) is None]
        if not layers:
            self.limits.append(join(named, ' lists no layers.'))
        elif len(empty) == len(layers):
            self.limits.append(join(named, ' produced no value: no token position was assessed. ', self.rows_sentence(comparison)))
        elif empty:
            self.limits.append(join(named, f' has no value at {_plural(len(empty), "layer")}: {_layer_runs(empty)}.'))
        skipped = join(*[code(str(row.get('id'))) for row in comparison.skipped], sep=', ')
        if not comparison.has_baseline:
            self.limits.append(join(named, ' records no plain baseline, so none is drawn for it.'))
        state = comparison.digest_state()
        details = [
            ('Candidate lens', _lens(comparison.candidate)), ('Reference lens', _lens(comparison.reference)),
            ('Text', join(_file(comparison.corpus), ', SHA-256 ', _hash(comparison.corpus.get('sha256')) or 'not recorded')),
            ('Comparison SHA-256',
             'Not recorded. This report was written before each comparison carried its own digest.' if state is None else
             join(code(comparison.raw['comparisonSHA256']),
                  ' (matches the stored entry)' if state else ' (does not match the stored entry)')),
            ('Held out?', _held_out(comparison.raw.get('heldOutStatus'))),
            ('Rows', join(self.rows_sentence(comparison), *([' Skipped row IDs: ', skipped, '.'] if comparison.skipped else []))),
            ('Readout precision', join(code(_text(comparison.precision.get('requested')) or 'native'), ' requested; float32 readout ',
                                       'recorded' if comparison.has_float32 else 'not recorded')
             if comparison.precision else None),
        ]
        between = 'The two lenses, read against each other'
        columns = [Column('Layer', numeric=True), Column('Token positions', numeric=True),
                   Column('Top-k overlap', numeric=True, group=between),
                   Column('Jensen–Shannon divergence', numeric=True, group=between)]
        if comparison.has_matrices:
            columns += [Column(label, numeric=True, group='The two lens matrices') for _, label in MATRIX_FIELDS]
        rows = []
        for layer in layers:
            stored = comparison.group(layer, 'betweenLenses')
            row = [layer, _count(stored.get('positions')), mean_text(stored.get(OVERLAP)), mean_text(stored.get(DIVERGENCE))]
            if comparison.has_matrices:
                matrices = _dict(_dict(comparison.readout.get(layer)).get('matrixComparison'))
                row += [_general(matrices.get(field)) for field, _ in MATRIX_FIELDS]
            rows.append(row)
        parts = [facts(details)]
        if rows:
            parts.append(table(
                join('Candidate ', _lens(comparison.candidate), ' and the reference, compared directly, on ', _file(comparison.corpus)),
                columns, rows, key=comparison.key + '-direct',
                footnote='Overlap and divergence here compare the two lenses’ readings with each other, not with the '
                         'final prediction. ' + (
                             'The matrix columns compare the two lenses’ own numbers at a layer: sizes are Frobenius '
                             'norms, and the relative difference is the difference size divided by the reference '
                             'size, as the engine stored it.' if comparison.has_matrices else
                             'This report stores no direct comparison of the two lens matrices.')))
        if comparison.has_float32:
            for measure, _, _ in MEASURES:
                parts.append(table(
                    join('Float32 readout check, ', IN_SENTENCE[measure], ': candidate ', _lens(comparison.candidate), ' on ', _file(comparison.corpus)),
                    [Column('Layer', numeric=True), *[Column(label, numeric=True) for _, label in FLOAT32_GROUPS]],
                    [[layer, *[mean_text(_dict(_dict(_dict(comparison.readout.get(layer)).get('float32')).get(group)).get(measure))
                               for group, _ in FLOAT32_GROUPS]] for layer in layers],
                    key=f'{comparison.key}-float32-{"overlap" if measure == OVERLAP else "divergence"}',
                    footnote='A check on numerical precision, not on the lens. The first four columns repeat the '
                             'comparisons with every reading taken in float32. The last four compare each float32 '
                             'reading with the model’s native reading of the same state.'))
        return block(join('Comparison ', str(comparison.number), ': candidate ', _lens(comparison.candidate)),
                     *parts, key=comparison.key, level=3)

    def lens_section(self):
        assessed = {comparison.pair[0]: comparison.corpus for comparison in self.comparisons if comparison.pair[0]}
        parts = [paragraph('What each lens was fitted on. A lens fitted on several texts is a mixed lens. Each of '
                           'its texts is listed with its row counts, so it cannot be mistaken for a lens fitted on one.')]
        for number, lens in enumerate(self.lenses, 1):
            fit, name = _dict(lens.get('fit')), lens.get('lensID')
            contributions = [entry for entry in _list(fit.get('corpora')) if isinstance(entry, dict)]
            layers = _layer_order([str(layer) for layer in _list(lens.get('sourceLayers'))])
            qualifications = [entry for entry in _list(lens.get('qualifications')) if isinstance(entry, dict)]
            single = _text(fit.get('corpus'))
            details = [
                ('Role in this report', self.role(name)),
                ('Fitted on model', join(_hash(fit.get('modelID')) or 'not recorded', ' at revision ',
                                         (_hash(fit.get('revision')) if fit.get('revisionKnown', True) else None) or 'not recorded')),
                ('Fitting text', join(f'Mixed: fitted on {len(contributions):,} texts, listed below. Combined digest ',
                                      _hash(single) or 'not recorded', '.') if contributions else
                 join('One text, ', code(single), '.') if single else None),
                ('Rows fitted', _count(fit.get('promptsFitted')) or _count(lens.get('nPrompts'))),
                ('Tokens read per row, at most', _count(fit.get('maxSeqLen'))),
                ('Layers', f'{len(layers):,} ({_layer_runs(layers)})' if layers else None),
                ('Converted tensor SHA-256', _hash(_dict(lens.get('converted')).get('sha256'))),
                ('Source tensor SHA-256', _hash(_dict(lens.get('source')).get('tensorSHA256'))),
                ('Fit report SHA-256', _hash(lens.get('fitReportSHA256'))),
                ('Evidence tier', join(code(lens['tier']), f' ({lens["tierSource"]})' if _text(lens.get('tierSource')) else '')
                 if _text(lens.get('tier')) else None),
                ('Qualifications on record', 'none' if not qualifications else join(*[
                    join(code(str(entry.get('qualificationID'))), ' (', 'passed' if entry.get('passed') is True else 'not passed', ')')
                    for entry in qualifications], sep=', ')),
            ]
            body = [facts(details)]
            if contributions:
                body.append(table(
                    join('Texts that lens ', _lens(name), ' was fitted on'),
                    [Column('Text SHA-256'), Column('Rows fitted', numeric=True), Column('Rows considered', numeric=True),
                     Column('Also assessed in this report?')],
                    [[_hash(entry.get('corpusSHA256')), _count(entry.get('promptsFitted')), _count(entry.get('rowsConsidered')),
                      join('Yes: ', _file(assessed[entry['corpusSHA256']]), ', so that text is not held out for this lens')
                      if _text(entry.get('corpusSHA256')) in assessed else 'No'] for entry in contributions],
                    key=f'lens-{number}-texts', row_header=False,
                    footnote='Rows considered includes rows the fit skipped. The report records each fitting text '
                             'by its hash only, not by file name.'))
            elif single and single.removeprefix('sha256:') in assessed:
                body.append(paragraph('This lens was fitted on ', _file(assessed[single.removeprefix('sha256:')]),
                                      ', which is also assessed in this report, so that text is not held out for it.'))
            parts.append(block(join(self.role(name), ' lens ', _lens(name)), *body, key=f'lens-{number}', level=3))
        if not self.lenses:
            parts.append(paragraph('The report holds no lens records.'))
        return block('Lenses', *parts, key='lenses')

    def limit_section(self):
        requested = requested_pairs(self.report)
        found = {comparison.pair for comparison in self.comparisons}
        asked = {pair_key(corpus, candidate) for corpus, candidate in requested}
        problems = [
            *[join('The request named candidate ', _lens(candidate), ' on ', _file(corpus),
                   ', but the report holds no comparison for that pair.')
              for corpus, candidate in requested if pair_key(corpus, candidate) not in found],
            *[f'Entry {position:,} in the report’s list of comparisons could not be read and is not shown.'
              for position in self.unreadable],
            *[join('Comparison ', comparison.ref, ' (candidate ', _lens(comparison.candidate), ' on ', _file(comparison.corpus),
                   ') is in the report but was not named in its recorded request.')
              for comparison in self.comparisons if requested and comparison.pair not in asked],
            *self.limits]
        return block(
            'Limits',
            el('h3', 'What these measures show'),
            paragraph('Top-k overlap and Jensen–Shannon divergence say how closely a reading at one layer matches '
                      'the model’s own final prediction, on the token positions that were assessed. They do not '
                      'show that a lens is causally faithful, that it will read other kinds of text as well, or '
                      'that a difference between two lenses is larger than chance. A close match on these texts '
                      'is a reason to look further, not a qualification.'),
            facts([('Recorded by the engine: how values were averaged', _text(self.report.get('aggregation'))),
                   ('Recorded by the engine: limitations', _text(self.report.get('limitations'))),
                   ('Recorded by the engine: qualification', _hash(self.report.get('qualification')))]),
            el('h3', 'Omissions and failed comparisons'),
            bullets(problems) if problems else
            paragraph('None found. Every requested comparison is present, every listed row was assessed, and '
                      'every layer has a stored value.'),
            paragraph('The request read at most ', _count(self.config.get('maxPrompts')) or 'an unrecorded number of',
                      ' rows from each text and assessed at most ',
                      _count(self.config.get('maxPositionsPerRow')) or 'an unrecorded number of',
                      ' token positions in each row. The report does not record how many rows each text holds, so '
                      'it cannot say whether rows beyond that cap were left unread.'),
            el('h3', 'In the report file but not on this page'),
            paragraph('The JSON also holds ', ', '.join(NOT_SHOWN[:-1]), ', and ', NOT_SHOWN[-1],
                      '. Nothing else was left out, and no list on this page is shortened.'),
            key='limits')

    def render(self):
        texts = [self.text_section(number, group) for number, group in enumerate(self.texts, 1)]
        summary, identity, lenses = self.summary(), self.identity(), self.lens_section()
        limits = self.limit_section()   # last: the sections above record what they could not show
        entries = [('summary', 'Summary'), ('identity', 'Identity'),
                   *[(f't{number}', join('Text ', str(number), ': ', _file(group[0].corpus)))
                     for number, group in enumerate(self.texts, 1)],
                   ('lenses', 'Lenses'), ('limits', 'Limits')]
        model = _text(self.config.get('modelID'))
        return document(
            title='J-lens assessment' + (f': {model}' if model else ''),
            lead=f'{_lenses(len(self.candidates))} compared with a reference lens'
                 f'{" and a plain baseline" if self.has_baseline else ""}, on {_plural(len(self.texts), "text")}, '
                 f'in {_plural(len(self.comparisons), "comparison")}.',
            body=join(contents(entries), summary, identity, *texts, lenses, limits),
            styles=(charts.CHART_STYLE,),
            footer=paragraph('Made by SteerLab from the stored report with SHA-256 ', code(self.sha256), '. The page '
                             'is a single file. It loads nothing from the network, and it shows only what that '
                             'report stores.'))


def render(data):
    """The page for one stored assessment report: the same bytes in, the same text out."""
    return Page(bytes(data)).render()
