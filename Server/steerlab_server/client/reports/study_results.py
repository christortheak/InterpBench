"""The results page of an ordinary study: one completed run, laid out for a reader.

Input is the reading :func:`results_export.read_results` makes of a run, its
newest analysis, and its newest completed evaluation. That reading is the
export's own, so this page and the files ``results export`` writes cannot
disagree about what was stored. Output is one HTML page, and the same reading
always gives the same page: it carries no date, no machine path, and no
software version, so a colleague who makes it again from the same run gets the
same bytes.

Nothing here computes a statistic. Estimates, intervals, p-values, counts,
agreement, and summaries are printed as the engine stored them, rounded for
reading. The only arithmetic is ordering and counting what is on the page:
which outcome leads (the shared headline rule), how many responses each
condition holds, and how many comparisons one correction covered. Where a
value was not stored, the page says so, and its last section lists everything
the run did not record.
"""
import math

from .. import results_export as source
from ...experiment import headline_outcome
from . import charts
from .page import Column, block, bullets, code, contents, document, el, facts, join, link, note, paragraph, table

PAGE_NAME = source.REPORT_PAGE

#: Fewer paired units than this carry no interval and no test. One or two
#: pairs still give a bootstrap "interval", but it is two or three possible
#: values dressed as a range. Swift twin: ``EffectNarrative.minimumPairsForInterval``.
MINIMUM_PAIRS = 3

_UNITS = {'item': ('paired item', 'paired items'), 'transcript': ('paired transcript', 'paired transcripts'),
          'sample': ('paired sample', 'paired samples'), 'response': ('paired response', 'paired responses'),
          'unknown': ('pair', 'pairs')}
#: What the table adds after a unit the analysis did not record itself.
_UNIT_NOTES = {'engine_default': ' (default)', 'inferred_from_records': ' (from the records)',
               'not_established': ''}
_CORRECTIONS = {'bh': 'Benjamini-Hochberg (false discovery rate)', 'holm': 'Holm'}
_SHORT_CORRECTIONS = {'bh': 'Benjamini-Hochberg', 'holm': 'Holm'}
_CONTROLS = {
    'randomMatchedNorm': 'adds a random direction of the same size as the concept direction, so an effect that '
                         'appears here too is not specific to the concept',
    'randomDirectionAblation': 'removes a random direction instead of the concept direction, so an effect that '
                               'appears here too is not specific to the concept',
}


# --- numbers, as stored ---------------------------------------------------------


def stored_number(value):
    """A stored value as a finite number, or ``None``. Effect rows hold text, reports hold JSON numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def number_text(value):
    """A stored number to four significant digits, as every table and chart on the page prints it."""
    number = stored_number(value)
    if number is None:
        return None
    if isinstance(number, int):
        return str(number)
    text = f'{number:.4g}'
    return '0' if text == '-0' else text


def signed_text(value):
    """An estimate: a difference from the baseline, so its sign is always shown."""
    number = stored_number(value)
    if number is None:
        return None
    text = number_text(number)
    return text if text.startswith('-') or text == '0' else '+' + text


def p_text(value):
    number = stored_number(value)
    if number is None:
        return None
    return '< 0.0001' if number < 0.0001 else number_text(number)


def count_text(value):
    number = stored_number(value)
    if number is None:
        return None
    return str(int(number)) if float(number).is_integer() else number_text(number)


def _plural(count, one, many=None):
    return f'{count} {one}' if count == 1 else f'{count} {many or one + "s"}'


def _inline(sentence):
    """A sentence shared with ``methods.md``, where `backticks` mark identifiers, as page text.

    The whole sentence is escaped first, so a stored name can never become
    markup. Unbalanced backticks are left as they are.
    """
    pieces = str(sentence).split('`')
    if len(pieces) % 2 == 0:
        return join(sentence)
    return join(*[code(piece) if index % 2 else piece for index, piece in enumerate(pieces)])


def _name(value):
    return code(value) if isinstance(value, str) and value else 'not recorded'


def _capital(phrase):
    return phrase[:1].upper() + phrase[1:]


def _outcome(name, *, capital=False):
    phrase = headline_outcome.plain_phrase(name)
    if not phrase:
        return code(name)
    return join(_capital(phrase) if capital else phrase, ' (', code(name), ')')


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


# --- the page --------------------------------------------------------------------


class Page:
    def __init__(self, stored):
        self.stored = stored
        self.study = stored['study']
        self.run = stored['runName']
        context = stored['context']
        self.context = context
        self.snapshot = _dict(context.get('snapshot'))
        self.config = _dict(context.get('config'))
        self.report = _dict(context.get('report'))
        self.loaded = context['loaded']
        self.responses = self.loaded['responses']
        self.rows = list(context.get('effectRows') or [])
        self.pooled = [row for row in self.rows if row['stratify_by'] in ('', 'pooled')]
        self.strata = [row for row in self.rows if row['stratify_by'] not in ('', 'pooled')]
        self.evaluations = [(kind, context[kind + 'Name'], _dict(context[kind + 'Report']))
                            for kind in ('paired', 'coding') if context.get(kind + 'Name')]
        self.headline = headline_of(stored)
        self.limits = []
        self.per_condition = {}
        for record in self.responses:
            name = str(record.get('condition', ''))
            self.per_condition[name] = self.per_condition.get(name, 0) + 1
        self.conditions = sorted(self.per_condition, key=lambda name: name != 'baseline')

    # -- shared pieces

    def unit(self, row):
        return row['unit_resolved']

    def family(self, row):
        """How many comparisons the row's correction covered: the rows of the same outcome that carry an adjusted p."""
        return sum(1 for other in self.pooled if other['outcome'] == row['outcome'] and other['adjusted_p_value'])

    @staticmethod
    def too_few(row):
        """Fewer than the minimum independent pairs. Paired responses are not
        independent of each other, so a response row counts its items."""
        n = row['paired_items'] if row['unit_resolved'] == 'response' else stored_number(row['n_pairs'])
        return n is not None and 1 <= n < MINIMUM_PAIRS

    def sentence(self, row):
        """One plain sentence for one stored effect row, in the shared rule's words."""
        n = stored_number(row['n_pairs'])
        one, many = _UNITS.get(self.unit(row), ('paired unit', 'paired units'))
        across = f' across {_plural(int(n), one, many)}' if n is not None and n >= 1 else ''
        parts = ['Under ', el('strong', row['condition'] or 'an unnamed condition'), ', ',
                 headline_outcome.plain_phrase(row['outcome']) or row['outcome'], ' differed from the baseline by ',
                 signed_text(row['estimate']) or 'an unrecorded amount', across]
        if row['unit_resolved'] == 'response':
            items = row['paired_items']
            parts.append(f' from {_plural(items, "item", "items")}. These are responses, not items: the analysis '
                         'paired each response with the baseline response to the same item and seed, so its interval '
                         'treats responses to the same item as independent and is not a finding about items')
        elif row['unit_resolved'] == 'unknown':
            parts.append('. The unit of these pairs is not established')
        if self.too_few(row):
            fewer = 'items' if row['unit_resolved'] == 'response' else 'pairs'
            parts.append(f'. That is too few {fewer} for an interval or a test (at least {MINIMUM_PAIRS} are needed), '
                         'so this describes these items only.')
            return join(*parts)
        low, high = stored_number(row['ci_lower']), stored_number(row['ci_upper'])
        if low is None or high is None:
            parts.append('. No interval is stored.')
        else:
            parts.append(f' (interval {number_text(low)} to {number_text(high)}). ')
            parts.append('The interval includes zero, so this run is consistent with no difference.'
                         if low <= 0 <= high else 'The interval does not include zero.')
        adjusted, raw, family = row['adjusted_p_value'], row['p_value'], self.family(row)
        if adjusted and family > 1:
            parts.append(f' Corrected p = {p_text(adjusted)} '
                         f'({_CORRECTIONS.get(row["correction"], row["correction"] or "correction not named")}, '
                         f'over {family} comparisons of this outcome).')
        elif adjusted or raw:
            parts.append(f' p = {p_text(adjusted or raw)} from the one Wilcoxon signed-rank test of this outcome '
                         '(a single comparison, so no correction applies).' if adjusted else
                         f' Uncorrected p = {p_text(raw)}; no correction for multiple comparisons is stored.')
        else:
            parts.append(' No test is stored, for example because every difference was zero.')
        return join(*parts)

    def forest(self, outcome, key):
        rows = [row for row in self.pooled if row['outcome'] == outcome]
        estimates = [charts.Estimate(
            label=row['condition'] or 'unnamed condition', value=stored_number(row['estimate']),
            lower=stored_number(row['ci_lower']), upper=stored_number(row['ci_upper']),
            note=f'too few pairs for an interval: {count_text(row["n_pairs"])}' if self.too_few(row) else None)
            for row in rows]
        phrase = headline_outcome.plain_phrase(outcome) or outcome
        return charts.forest_chart(
            key=key, title=f'{_capital(phrase)} ({outcome}): each condition minus the baseline', rows=estimates,
            value_text=number_text, estimate_text=signed_text, axis_name=f'Difference from the baseline in {phrase}',
            summary='Every value is in the table under Effects.')

    # -- sections

    def first(self):
        """What a reader must not miss, before anything else."""
        items = []
        if self.snapshot.get('freezeForced'):
            items.append(join('This study was frozen with force: some freeze checks were skipped. See ',
                              link('freeze', 'Freeze and identity'), '.'))
        elif self.snapshot.get('status') not in (None, 'frozen', 'complete'):
            items.append(join('This study was not frozen when the run was made, so its settings were not fixed '
                              'before behavior was measured. See ', link('freeze', 'Freeze and identity'), '.'))
        for entry in _list(self.snapshot.get('capabilityBatteryNotApplied')):
            if isinstance(entry, dict):
                items.append(source._battery_sentence(entry))
        if self.headline.declared_absent:
            items.append(join('The study declared ', code(self.headline.declared_outcome), ' as its primary outcome, '
                              'and this run does not have it. The headline below was chosen by default order.'))
        flags = self.context.get('analysisFlags') or {}
        if flags.get('epochUnverified'):
            items.append('The analysis accepted a run that carries no stamp of the study’s settings, so it could not '
                         'confirm the run matches them.')
        if flags.get('measurementDrift'):
            items.append(join('The study’s measurement settings changed after the run was made. The analysis '
                              'recorded: ', str(flags['measurementDrift'])))
        for _, _, evaluation in self.evaluations:
            if evaluation.get('epochUnverified'):
                items.append('The evaluation accepted a run that carries no stamp of the study’s settings, so it '
                             'could not confirm the run matches them.')
            if evaluation.get('measurementDrift'):
                items.append(join('The study’s judging settings changed after the run was made. The evaluation '
                                  'recorded: ', str(evaluation['measurementDrift'])))
        if not items:
            return None
        return el('div', el('p', el('strong', 'Read this first')), bullets(items), class_='note', id='first')

    def headline_section(self):
        selection = self.headline
        parts = []
        if selection.outcome is None:
            parts.append(paragraph('This run has no outcome to lead with: no analysis and no completed evaluation '
                                   'of it were found. Analyze the run, or evaluate it with the study’s judges, then '
                                   'make this page again.'))
            if selection.declared_outcome:
                parts.append(paragraph('The study declared ', code(selection.declared_outcome),
                                       ' as its primary outcome.'))
            return block('Headline outcome', *parts, key='headline')
        parts.append(el('p', el('strong', _outcome(selection.outcome, capital=True)), ', ', selection.chosen_by, '.',
                        class_='headline'))
        parts.append(paragraph(
            'The headline is the outcome the researcher declared before the run, when the run has it. Otherwise it '
            'is the first outcome the run has in a fixed order: a judged outcome, then a declared choice or numeric '
            'outcome, a reader score, reasoning style, marker density, and surface measures such as length last. '
            'Every other outcome follows under ', link('effects', 'Effects'), '.'))
        if selection.source == headline_outcome.SOURCE_EVALUATION:
            parts += self.judged_headline()
            return block('Headline outcome', *parts, key='headline')
        rows = [row for row in self.pooled if row['outcome'] == selection.outcome]
        parts.append(bullets([self.sentence(row) for row in rows]))
        parts.append(self.forest(selection.outcome, 'headline-chart'))
        parts.append(self.effects_table(rows, key='headline-table', caption=join(
            'The headline outcome, ', code(selection.outcome), ', as the analysis stored it')))
        return block('Headline outcome', *parts, key='headline')

    def judged_headline(self):
        parts = []
        for kind, name, evaluation in self.evaluations:
            if kind == 'paired':
                parts.append(paragraph('Paired judging, from ', code('runs/' + name), ': each judge saw a condition’s '
                                       'response and the baseline’s response to the same item, in varied order and '
                                       'without being told which was which, and preferred one or called a tie.'))
                parts.append(self.tally_table(evaluation, key='headline-tally'))
            else:
                parts.append(paragraph('Response coding, from ', code('runs/' + name), ': each judge coded one '
                                       'response at a time against the rubric’s fields, without being told its '
                                       'condition.'))
                parts += self.coding_tables(evaluation, key='headline-coding')
        parts.append(note('The engine stores counts and shares for a judged outcome. It computed no interval and no '
                          'test for it, so this page shows none.', label='Read with care'))
        self.limits.append('The judged outcome has no stored interval or test; only the counts and shares the '
                           'evaluation stored are shown.')
        return parts

    def tally_table(self, evaluation, *, key):
        """Per-condition preferences, as the evaluation report stored them: per judge where it keeps them."""
        blocks = [entry for entry in _list(evaluation.get('judges'))
                  if isinstance(entry, dict) and isinstance(entry.get('conditions'), dict)]
        rows = []
        if blocks:
            for entry in blocks:
                for condition, tally in entry['conditions'].items():
                    tally = _dict(tally)
                    rows.append([code(str(entry.get('name'))), condition, count_text(tally.get('n', tally.get('pairs'))),
                                 count_text(tally.get('variantWins', tally.get('conditionWins'))),
                                 count_text(tally.get('baselineWins')), count_text(tally.get('ties')),
                                 number_text(tally.get('meanConfidence'))])
        else:
            for condition, tally in _dict(evaluation.get('conditions')).items():
                tally = _dict(tally)
                rows.append(['the panel', condition, count_text(tally.get('pairs', tally.get('n'))),
                             count_text(tally.get('conditionWins', tally.get('variantWins'))),
                             count_text(tally.get('baselineWins')), count_text(tally.get('ties')),
                             number_text(tally.get('meanConfidence'))])
        if not rows:
            return paragraph('The evaluation report stores no per-condition counts.')
        return table('Which response the judges preferred, for each condition',
                     [Column('Judge'), Column('Condition'), Column('Verdicts', numeric=True),
                      Column('Condition preferred', numeric=True), Column('Baseline preferred', numeric=True),
                      Column('Ties', numeric=True), Column('Mean confidence', numeric=True)],
                     rows, key=key, row_header=False,
                     footnote='Counts as the evaluation report stored them. Noncompliant answers, which carry no '
                              'verdict, are not in these counts; the Judges section gives their number.')

    def coding_tables(self, evaluation, *, key):
        fields = [field for field in _list(evaluation.get('fields')) if isinstance(field, dict) and field.get('name')]
        conditions = _dict(evaluation.get('conditions'))
        if not conditions:
            return [paragraph('The coding report stores no per-condition summary.')]
        parts = []
        for number, field in enumerate(fields, 1):
            rows = []
            for condition, summary in conditions.items():
                entry = _dict(_dict(_dict(summary).get('fields')).get(field['name']))
                if 'trueShare' in entry:
                    value = join(number_text(entry.get('trueShare')) or 'no value', ' (',
                                 count_text(entry.get('trueCount')) or 'no count', ' true)')
                elif 'mean' in entry:
                    value = number_text(entry.get('mean'))
                elif isinstance(entry.get('counts'), dict):
                    value = join(*[f'{label} {count_text(count)}' for label, count in entry['counts'].items()],
                                 sep='; ') or 'none'
                else:
                    value = None
                rows.append([condition, count_text(entry.get('n')), count_text(entry.get('nulls')), value])
            kind = field.get('type')
            parts.append(table(join('Rubric field ', code(field['name']), f' ({kind})' if kind else '', ', for each condition'),
                               [Column('Condition'), Column('Coded', numeric=True), Column('Left empty', numeric=True),
                                Column('True share' if kind == 'boolean' else 'Mean' if kind in ('integer', 'number')
                                       else 'Counts')],
                               rows, key=f'{key}-{number}',
                               footnote='As the coding report stored it, across every judge. Noncompliant codings '
                                        'are outside these numbers.'))
        return parts or [paragraph('The coding report declares no rubric fields.')]

    def design(self):
        snapshot, observed = self.snapshot, self.loaded['observed']
        models = observed['modelID'] or ([snapshot['modelID']] if snapshot.get('modelID') else [])
        revisions = observed['modelRevision'] or ([snapshot['modelRevision']] if snapshot.get('modelRevision') else [])
        substrate = self.config.get('substrate') or (observed['engine'][0] if observed['engine'] else None)
        items = []
        for record in self.responses:
            if record.get('promptID') not in items:
                items.append(record.get('promptID'))
        panel = any(source._is_turn(record) for record in self.responses)
        prompts_file = snapshot.get('taskPromptsFile') or (observed['taskPromptsFile'][0]
                                                           if observed['taskPromptsFile'] else None)
        prompts_hash = snapshot.get('taskPromptsHash') or (observed['taskPromptsHash'][0]
                                                           if observed['taskPromptsHash'] else None)
        pairs = [('Description, as the researcher wrote it', snapshot.get('experimentDescription') or None),
                 ('Task, as the researcher wrote it', snapshot.get('taskDescription') or None),
                 ('Kind of study', _name(snapshot.get('studyType') or snapshot.get('studyKind'))
                  if snapshot.get('studyType') or snapshot.get('studyKind') else None),
                 ('Model', join(*[code(model) for model in models], sep=', ') if models else None),
                 ('Model revision', join(*[code(revision) for revision in revisions], sep=', ') if revisions else None),
                 ('Engine', join(source._ENGINES.get(substrate, 'an engine this version does not know'),
                                 ' (recorded as ', code(substrate), ')') if substrate else None)]
        if panel:
            pairs.append(('Panel script', join(_name(snapshot.get('multiAgentScenarioPath')),
                                               *([', SHA-256 ', code(snapshot['multiAgentScenarioHash'])]
                                                 if snapshot.get('multiAgentScenarioHash') else []))))
            pairs.append(('Turns in the script', str(len(items))))
            pairs.append(('Conversations', join(str(self.context.get('conversations') or 0),
                                                ', one for each condition and play-through')))
            speakers = []
            for record in self.responses:
                seat = (record.get('speakerAgentID'), record.get('speakerName'))
                if source._is_turn(record) and seat not in speakers:
                    speakers.append(seat)
            pairs.append(('Seats that spoke', join(*[join(name or 'unnamed', ' (seat ', code(seat), ')' if seat else '')
                                                     if seat else (name or 'unnamed') for seat, name in speakers],
                                                   sep=', ') or None))
        else:
            pairs.append(('Task items', join(str(len(items)), *([', from ', code(prompts_file)] if prompts_file else []),
                                             *([', SHA-256 ', code(prompts_hash)] if prompts_hash else []))))
        counts = ', '.join(f'{name} {self.per_condition[name]}' for name in self.conditions)
        pairs.append(('Responses', f'{len(self.responses)} in total' + (f' ({counts})' if counts else '')))
        sampling = []
        temperatures = observed['temperature'] or ([snapshot['temperature']] if 'temperature' in snapshot else [])
        if temperatures:
            sampling.append('temperature ' + ', '.join(source._number(value) for value in temperatures))
        for field, label in (('topP', 'top-p'), ('topK', 'top-k')):
            if observed[field]:
                sampling.append(label + ' ' + ', '.join(source._number(value) for value in observed[field]))
        if 'maxTokens' in snapshot:
            sampling.append(f'at most {snapshot["maxTokens"]} new tokens for each response')
        samples = self.config.get('samplesPerItem') or snapshot.get('samplesPerItem')
        if samples:
            sampling.append(f'{_plural(samples, "sample")} for each item')
        policy = self.config.get('seedPolicy') or snapshot.get('seedPolicy') or (
            observed['seedPolicy'][0] if observed['seedPolicy'] else None)
        if policy:
            sampling.append(f'seed policy {policy}')
        if snapshot.get('reasoningEffort'):
            sampling.append(f'reasoning effort {snapshot["reasoningEffort"]}')
        mode = snapshot.get('promptMode') or (observed['promptMode'][0] if observed['promptMode'] else None)
        if mode:
            sampling.append(f'prompt mode {mode}')
        pairs.append(('Sampling', '; '.join(sampling) or None))
        if snapshot.get('systemPrompt'):
            pairs.append(('System prompt for every condition', snapshot['systemPrompt']))
        elif snapshot:
            pairs.append(('System prompt', 'none declared for the study'))
        parts = [facts(pairs)]
        rows = [[el('strong', name), _inline(source._describe_condition(name, snapshot)),
                 str(self.per_condition.get(name, 0))] for name in self.conditions]
        declared = [entry.get('name') for field in ('conditions', 'variantConditions', 'saeLatentConditions')
                    for entry in _list(snapshot.get(field)) if isinstance(entry, dict)]
        failed = {str(entry.get('condition')): entry.get('error') for entry in self.loaded['failedConditions']}
        for name in declared:
            if name and name not in self.per_condition:
                rows.append([el('strong', name), join(
                    'Declared for the study. ', 'The run recorded an error and no responses: ' + str(failed[name])
                    if name in failed else 'The run holds no responses for it.'), '0'])
        for name, error in failed.items():
            if name not in declared and name not in self.per_condition:
                rows.append([el('strong', name), join('The run recorded an error and no responses: ', str(error)), '0'])
        if rows:
            parts.append(table('What each condition applied', [Column('Condition'), Column('What it applied'),
                                                               Column('Responses', numeric=True)],
                               rows, key='design-conditions',
                               footnote='Every condition is compared with the baseline, the arm with no intervention.'))
        else:
            parts.append(paragraph('The run holds no responses, so no condition can be described.'))
        return block('What was asked and compared', *parts, key='design')

    def effects_table(self, rows, *, key, caption):
        modality = any(row['modality'] for row in rows)
        columns = [Column('Outcome'), Column('Condition'), Column('Estimate', numeric=True),
                   Column('From', numeric=True, group='Interval'), Column('To', numeric=True, group='Interval'),
                   Column('Pairs', numeric=True), Column('Unit'), Column('W', numeric=True),
                   Column('p', numeric=True), Column('Corrected p', numeric=True), Column('Correction'),
                   *([Column('Intervention')] if modality else []), Column('Note')]
        drawn = []
        for row in rows:
            drawn.append([code(row['outcome']), row['condition'], signed_text(row['estimate']),
                          number_text(row['ci_lower']), number_text(row['ci_upper']), count_text(row['n_pairs']),
                          self.unit(row) + _UNIT_NOTES.get(row['unit_source'], ''),
                          number_text(row['test_statistic']), p_text(row['p_value']), p_text(row['adjusted_p_value']),
                          _SHORT_CORRECTIONS.get(row['correction'], row['correction']) or None,
                          *([row['modality'] or None] if modality else []),
                          f'Fewer than {MINIMUM_PAIRS} pairs: no interval or test' if self.too_few(row) else ''])
        few = any(self.too_few(row) for row in rows)
        return table(caption, columns, drawn, key=key, row_header=False,
                     footnote='Copied from the analysis and rounded to four significant digits; the stored file holds '
                              'every digit. Estimate is the condition minus the baseline. W is the Wilcoxon signed-rank '
                              'statistic. Benjamini-Hochberg controls the false discovery rate. “No value” means the '
                              'analysis stored none, for example a test that is undefined because every difference '
                              'was zero.' + (f' A row with fewer than {MINIMUM_PAIRS} pairs still shows the interval '
                                             'and test the analysis stored, but they describe those items only and '
                                             'are not a test.' if few else ''))

    def unit_explanations(self):
        """One "Unit of analysis" line per way the rows' units are known: the
        export's methods summary draws on the same resolution."""
        groups = source.unit_groups(self.context)
        lines = []
        for how, units in groups:
            label = (join(' for the rows marked ', join(*[code(unit) for unit in units], sep=', '))
                     if len(groups) > 1 else '')
            if how == 'recorded':
                said = join(join(*[code(unit) for unit in units], sep=', '), ', as the analysis recorded.')
            elif how == 'engine_default':
                said = ('the item, with an item’s samples averaged within each condition. This is the engines’ '
                        'documented default; the analysis did not stamp the unit itself. The run’s records agree: no '
                        'such row counts more pairs than the items paired in the run.')
            elif how == 'inferred_from_records':
                said = 'the response, not the item. ' + source.RESPONSE_UNIT_EXPLANATION
            else:
                said = ('not established. The analysis did not stamp it, and the run’s records have no items paired '
                        'with the baseline for these conditions.')
            lines.append(join(el('strong', 'Unit of analysis'), label, ': ', said))
        return lines

    def effects(self):
        context = self.context
        if context.get('effectsSource') is None:
            return block('Effects', paragraph(
                'No analysis of this run was found, so there are no effect estimates. Analyze the run, then make '
                'this page again.'), key='effects')
        corrections = context.get('corrections') or []
        explained = [
            join(el('strong', 'Estimate'), ': the mean of the paired differences, condition minus baseline. A pair is '
                 'the same unit measured under the condition and under the baseline.'),
            *self.unit_explanations(),
            join(el('strong', 'Interval'), ': a bootstrap confidence interval for the estimate. The engines compute a '
                 '95% percentile interval by default, but the analysis files do not record the level, the number of '
                 'resamples, or the seed, so this page does not state them as fact.'),
            join(el('strong', 'Test'), ': ', 'the Wilcoxon signed-rank test on the paired differences.'
                 if context['effectsSource'].get('p_value') or context['effectsSource'].get('test_statistic')
                 else 'none stored.'),
            join(el('strong', 'Correction for multiple comparisons'), ': ',
                 ', '.join(_CORRECTIONS.get(name, name) for name in corrections) +
                 ', applied to each outcome separately, across conditions.' if corrections else 'none recorded.'),
        ]
        parts = [paragraph('The analysis, from ', code(context['analysisLabel']), ', compared each condition with the '
                           'baseline on each outcome.'), bullets(explained)]
        outcomes = [name for name in context.get('outcomes') or [] if name != self.headline.outcome]
        if self.headline.outcome in (context.get('outcomes') or []):
            parts.append(paragraph('The headline outcome’s chart is under ', link('headline', 'Headline outcome'),
                                   '. One chart for each other outcome follows.' if outcomes else '.'))
        for number, outcome in enumerate(outcomes, 1):
            parts.append(self.forest(outcome, f'effect-chart-{number}'))
        if self.pooled:
            parts.append(self.effects_table(self.pooled, key='effects-table',
                                            caption='Every effect the analysis stored, over all items'))
            described = [join(code(name), ': ', source._describe_outcome(name), '.')
                         for name in context.get('outcomes') or []]
            parts.append(el('h3', 'What each outcome measures'))
            parts.append(bullets(described))
        else:
            parts.append(paragraph('The analysis stored no effect over all items.'))
        if self.strata:
            parts.append(el('h3', 'Within subgroups of the items'))
            parts.append(table(
                'Every effect the analysis stored within a subgroup',
                [Column('Outcome'), Column('Condition'), Column('Subgroups by'), Column('Subgroup'),
                 Column('Estimate', numeric=True), Column('From', numeric=True, group='Interval'),
                 Column('To', numeric=True, group='Interval'), Column('Pairs', numeric=True), Column('Unit'),
                 Column('p', numeric=True), Column('Corrected p', numeric=True), Column('What it estimates'),
                 Column('What its p supports')],
                [[code(row['outcome']), row['condition'], row['stratify_by'], row['stratum'],
                  signed_text(row['estimate']), number_text(row['ci_lower']), number_text(row['ci_upper']),
                  count_text(row['n_pairs']), self.unit(row), p_text(row['p_value']), p_text(row['adjusted_p_value']),
                  {'itemLevel': 'an effect across items',
                   'withinItemSamples': 'one item’s samples only'}.get(
                      row['estimand'], row['estimand']) or None,
                  {'corrected': 'a corrected test', 'diagnostic': 'a description, not a test'}.get(
                      row['inference'], row['inference']) or None]
                 for row in self.strata],
                key='strata-table', row_header=False,
                footnote='Copied from the analysis and rounded to four significant digits. A row marked as a '
                         'description describes the samples of one item and is not a test.'))
        return block('Effects', *parts, key='effects')

    def condition_summary(self):
        conditions = _dict(self.report.get('conditions'))
        fields = (('generations', 'Responses', count_text), ('meanWordCount', 'Mean words', number_text),
                  ('meanDistinct2', 'Mean lexical variety', number_text), ('choiceRate', 'Target-choice rate', number_text),
                  ('choiceReadouts', 'Answer-option readings', count_text), ('ordinalMean', 'Mean scale position', number_text),
                  ('ordinalSD', 'Scale position SD', number_text))
        present = [field for field in fields if any(field[0] in _dict(block) for block in conditions.values())]
        agreement = any(isinstance(_dict(block).get('agreementWithBaseline'), dict) for block in conditions.values())
        markers = []
        for block_ in conditions.values():
            for concept in _dict(_dict(block_).get('meanMarkerDensity')):
                if concept not in markers:
                    markers.append(concept)
        parts = [paragraph('What the run itself stored for each condition, before any analysis. These are '
                           'descriptions of each arm, not comparisons with the baseline.')]
        if conditions and (present or agreement or markers):
            columns = [Column('Condition'), *[Column(label, numeric=True) for _, label, _ in present],
                       *([Column('Agreement with baseline', numeric=True)] if agreement else []),
                       *[Column(concept, numeric=True, group='Mean marker density') for concept in markers]]
            rows = []
            for name, block_ in conditions.items():
                block_ = _dict(block_)
                together = _dict(block_.get('agreementWithBaseline'))
                rows.append([name, *[format_(block_.get(key)) for key, _, format_ in present],
                             *([join(number_text(together.get('agreement')) or 'no value', ' over ',
                                     count_text(together.get('n')) or 'an unrecorded number of', ' items')
                                if together else None] if agreement else []),
                             *[number_text(_dict(block_.get('meanMarkerDensity')).get(concept)) for concept in markers]])
            explained = {
                'meanDistinct2': 'Lexical variety is the share of two-word sequences that are distinct.',
                'choiceRate': 'The target-choice rate is the share of readable responses that chose the item’s '
                              'declared target.',
                'ordinalSD': 'SD is the standard deviation the engine stored.'}
            notes = [explained[key] for key, _, _ in present if key in explained]
            if agreement:
                notes.append('Agreement with baseline is the share of items where the condition chose the same answer '
                             'option as the baseline.')
            if markers:
                notes.append('Marker density is the share of a response’s words that are marker words for the concept.')
            parts.append(table('What the run stored for each condition', columns, rows, key='condition-summary',
                               footnote=' '.join(['Rounded to four significant digits.', *notes])))
        elif conditions:
            parts.append(paragraph('The run’s report stores no per-condition summary this page can show.'))
        else:
            parts.append(paragraph('The run’s report stores no per-condition summary.'))
            self.limits.append('The run’s report.json holds no per-condition summary.')
        errors = [(name, _dict(block_).get('error')) for name, block_ in conditions.items() if _dict(block_).get('error')]
        if errors:
            parts.append(bullets([join(code(name), ' recorded an error and produced no responses: ', str(error))
                                  for name, error in errors]))
        truncation = _dict(self.report.get('truncation'))
        if truncation.get('classified') is not None:
            cut = [cell for cell in _list(truncation.get('cells'))
                   if isinstance(cell, dict) and stored_number(cell.get('lengthStopped'))]
            parts.append(el('h3', 'Responses cut off at a token limit'))
            parts.append(paragraph(f'{count_text(truncation.get("lengthStopped")) or "An unrecorded number"} of '
                                   f'{count_text(truncation.get("classified"))} responses stopped at the token limit '
                                   'instead of finishing, as the engine counted them.'
                                   + (f' The study allows at most {number_text(truncation["threshold"])} of any '
                                      'condition and item.' if stored_number(truncation.get('threshold')) is not None
                                      else '')))
            if cut:
                parts.append(table('Conditions and items with a response cut off',
                                   [Column('Condition'), Column('Item'), Column('Cut off', numeric=True),
                                    Column('Responses', numeric=True)],
                                   [[cell.get('condition'), code(str(cell.get('promptID'))),
                                     count_text(cell.get('lengthStopped')), count_text(cell.get('classified'))]
                                    for cell in cut], key='truncation-cells', row_header=False,
                                   footnote='Only the cells with at least one cut-off response are listed; every '
                                            'other cell finished all its responses.'))
        return block('Each condition', *parts, key='conditions')

    def controls(self):
        parts = []
        arms = [entry for entry in _list(self.snapshot.get('conditions'))
                if isinstance(entry, dict) and entry.get('controlType')]
        if arms:
            parts.append(bullets([join(el('strong', arm.get('name')), ': ',
                                       _CONTROLS.get(arm['controlType'], 'a control this version cannot describe'),
                                       ' (recorded as ', code(arm['controlType']), '). Its effects are under ',
                                       link('effects', 'Effects'), '.') for arm in arms]))
        else:
            parts.append(paragraph('This study declares no control arm, such as a random direction of the same size, '
                                   'to compare its steering with.'))
        conditions = _dict(self.report.get('conditions'))
        scored = [(name, _dict(block_.get('capabilityBattery'))) for name, block_ in conditions.items()
                  if isinstance(block_, dict) and isinstance(block_.get('capabilityBattery'), dict)]
        exempt = _list(self.report.get('capabilityBatteryNotApplied')) or _list(
            self.snapshot.get('capabilityBatteryNotApplied'))
        parts.append(el('h3', 'Capability battery'))
        if scored or exempt:
            parts.append(paragraph('A set of unrelated questions, scored under each condition, to check that the '
                                   'intervention did not damage other abilities.'))
        if scored:
            parts.append(table('Capability battery accuracy, for each condition',
                               [Column('Condition'), Column('Accuracy', numeric=True), Column('Items', numeric=True),
                                Column('Battery SHA-256')],
                               [[name, number_text(entry.get('accuracy')), count_text(entry.get('itemCount')),
                                 code(entry['batteryHash']) if entry.get('batteryHash') else None]
                                for name, entry in scored], key='battery', row_header=False,
                               footnote='As the engine scored it.'))
        for entry in exempt:
            if isinstance(entry, dict):
                parts.append(paragraph(source._battery_sentence(entry)))
        if not scored and not exempt:
            parts.append(paragraph('No capability battery was run for this study.'))
        return block('Controls', *parts, key='controls')

    def judges(self):
        context = self.context
        if not self.evaluations:
            parts = [paragraph('No completed evaluation by judges was found for this run.')]
            declared = [entry.get('name') for entry in _list(self.snapshot.get('judges')) if isinstance(entry, dict)]
            if declared:
                parts.append(paragraph('The study’s settings declare these judges: ',
                                       join(*[code(name) for name in declared], sep=', '),
                                       '. Evaluate the run, then make this page again.'))
            return block('Judges', *parts, key='judges')
        parts = []
        for kind, name, evaluation in self.evaluations:
            rows = context['judgmentRows' if kind == 'paired' else 'codingRows']
            details = source._judge_details(evaluation)
            label = 'Paired judging' if kind == 'paired' else 'Response coding'
            body = [paragraph('From ', code('runs/' + name), '.')]
            judges = []
            for judge in evaluation.get('judges') or []:
                judge_name = judge.get('name') if isinstance(judge, dict) else judge
                if not isinstance(judge_name, str):
                    continue
                entry = details.get(judge_name, {})
                row = next((item for item in rows if item.get('judge') == judge_name), {})
                judges.append([code(judge_name), entry.get('kind') or row.get('judgeKind'),
                               _name(entry.get('requestedModel') or row.get('judgeModel'))
                               if entry.get('requestedModel') or row.get('judgeModel') else None,
                               _name(entry.get('revision') or row.get('judgeRevision'))
                               if entry.get('revision') or row.get('judgeRevision') else None,
                               entry.get('provider') or row.get('judgeProvider')])
            if judges:
                body.append(table(join(label, ': the judges'), [Column('Judge'), Column('Kind'), Column('Model'),
                                                                Column('Revision'), Column('Provider')],
                                  judges, key=f'{kind}-judges', row_header=False,
                                  footnote='A judge’s name is a label in the study’s panel, not a model. A local judge '
                                           'is a model the engine ran itself.'))
            elif evaluation.get('judgeModel'):
                body.append(paragraph('Judge model: ', code(str(evaluation['judgeModel'])), '.'))
            rubric = evaluation.get('judgeRubricFile') or evaluation.get('rubricFile')
            rubric_hash = evaluation.get('judgeRubricHash') or evaluation.get('rubricHash')
            noncompliant = sum(1 for row in rows if row.get('noncompliant'))
            stored = evaluation.get('noncompliantJudgments' if kind == 'paired' else 'noncompliantCodings')
            body.append(facts([
                ('Rubric', join(code(rubric), *([', SHA-256 ', code(rubric_hash)] if rubric_hash else []))
                 if rubric else None),
                ('Rows', join(str(len(rows)), ', of which ', str(noncompliant),
                              ' marked noncompliant: the judge answered without a usable result. Such rows carry no '
                              'verdict and are left out of every count and agreement statistic.')),
                ('Noncompliant, as the report recorded', count_text(stored) if stored is not None else
                 'none recorded' if noncompliant == 0 else None),
            ]))
            subsample = evaluation.get('sampling')
            if isinstance(subsample, dict):
                body.append(note('The judges coded a seeded subsample, not every response: ',
                                 count_text(subsample.get('sampledRecords')) or 'an unrecorded number', ' of ',
                                 count_text(subsample.get('sourceRecords')) or 'an unrecorded number', ' responses, ',
                                 count_text(subsample.get('samplePerCondition')) or 'an unrecorded number',
                                 ' for each condition, drawn with seed ', code(str(subsample.get('sampleSeed'))), '.',
                                 label='Subsample'))
            agreement = [entry for entry in _list(evaluation.get('agreement') or evaluation.get('judgeAgreement'))
                         if isinstance(entry, dict)]
            if agreement:
                body.append(table(join(label, ': agreement between judges'),
                                  [Column('Judges'), Column('Shared verdicts', numeric=True),
                                   Column('Proportion in agreement', numeric=True), Column('Cohen’s kappa', numeric=True)],
                                  [[join(*[code(str(judge)) for judge in (entry.get('judges') or
                                                                          [entry.get('judgeA'), entry.get('judgeB')])],
                                         sep=' and '),
                                    count_text(entry.get('n', entry.get('items'))),
                                    number_text(entry.get('percentAgreement')), number_text(entry.get('kappa'))]
                                   for entry in agreement], key=f'{kind}-agreement', row_header=False,
                                  footnote='As the engine computed it, over the verdicts both judges gave. Kappa '
                                           'corrects agreement for what chance alone would give; no interval is '
                                           'stored for it.'))
            fields = [entry for entry in _list(evaluation.get('fieldAgreement')) if isinstance(entry, dict)]
            if fields:
                body.append(table(join(label, ': agreement between judges on each rubric field'),
                                  [Column('Field'), Column('Judges'), Column('Shared responses', numeric=True),
                                   Column('Proportion in agreement', numeric=True), Column('Cohen’s kappa', numeric=True),
                                   Column('Mean absolute difference', numeric=True)],
                                  [[code(str(entry.get('field'))),
                                    join(code(str(entry.get('judgeA'))), ' and ', code(str(entry.get('judgeB')))),
                                    count_text(entry.get('n')), number_text(entry.get('percentAgreement')),
                                    number_text(entry.get('kappa')), number_text(entry.get('meanAbsoluteDifference'))]
                                   for entry in fields], key=f'{kind}-field-agreement', row_header=False,
                                  footnote='As the engine computed it.'))
            elif evaluation.get('fieldAgreementAbsentReason'):
                body.append(paragraph('Agreement between judges: not available. ',
                                      str(evaluation['fieldAgreementAbsentReason'])))
            elif not agreement:
                body.append(paragraph('The evaluation stores no agreement between judges.'))
            human = [entry for entry in _list(evaluation.get('humanAgreement')) if isinstance(entry, dict)]
            if human:
                body.append(table(join(label, ': agreement with the researcher’s own verdicts'),
                                  [Column('Judge'), Column('Shared verdicts', numeric=True),
                                   Column('Proportion in agreement', numeric=True), Column('Cohen’s kappa', numeric=True)],
                                  [[code(str(entry.get('judge'))), count_text(entry.get('n', entry.get('items'))),
                                    number_text(entry.get('percentAgreement')), number_text(entry.get('kappa'))]
                                   for entry in human], key=f'{kind}-human', row_header=False,
                                  footnote='As the engine computed it, against the human validation file the study pins.'))
            sessions = evaluation.get('judgingSessions')
            if isinstance(sessions, dict):
                body.append(paragraph('This evaluation was completed over more than one sitting: ',
                                      count_text(sessions.get('reusedJudgments')) or 'an unrecorded number of',
                                      ' verdicts were reused from ', code(str(sessions.get('resumedFrom'))), ' and ',
                                      count_text(sessions.get('freshJudgments')) or 'an unrecorded number', ' are new.'))
            if self.headline.source != headline_outcome.SOURCE_EVALUATION:
                body.append(self.tally_table(evaluation, key=f'{kind}-tally') if kind == 'paired'
                            else join(*self.coding_tables(evaluation, key=f'{kind}-coding')))
            else:
                body.append(paragraph('The per-condition results are under ', link('headline', 'Headline outcome'), '.'))
            parts.append(block(label, *body, key=f'judges-{kind}', level=3))
        return block('Judges', *parts, key='judges')

    def exclusions(self):
        stamps = [(label, stamp) for label, stamp in self.context.get('exclusions') or [] if isinstance(stamp, dict)]
        rules = _list(self.snapshot.get('exclusionRules'))
        parts = []
        if stamps:
            for label, stamp in stamps:
                removed = stamp.get('excludedRecords')
                parts.append(paragraph(f'{label} applied the study’s declared exclusion rules and removed '
                                       f'{count_text(removed) or "an unrecorded number of"} '
                                       f'{"record" if stored_number(removed) == 1 else "records"}.'))
                described = [join(code(str(rule.get('rule'))), ': ', str(rule.get('description') or
                                                                       'no description recorded'))
                             for rule in _list(stamp.get('rules')) if isinstance(rule, dict)]
                if described:
                    parts.append(bullets(described))
                by_condition = _dict(stamp.get('excludedByRule'))
                if by_condition:
                    parts.append(table(join(label, ': records for each condition'),
                                       [Column('Condition'), Column('Considered', numeric=True),
                                        Column('Removed, by rule'), Column('Kept', numeric=True)],
                                       [[condition, count_text(_dict(stamp.get('consideredN')).get(condition)),
                                         ', '.join(f'{rule} {count_text(count)}' for rule, count in _dict(by_rule).items())
                                         or 'none', count_text(_dict(stamp.get('survivingN')).get(condition))]
                                        for condition, by_rule in by_condition.items()],
                                       key=f'exclusions-{len(parts)}', row_header=False,
                                       footnote='As the engine counted them.'))
                if stamp.get('note'):
                    parts.append(paragraph('The engine’s note: ', str(stamp['note'])))
            parts.append(paragraph('Excluded records stay in the run. The engine records how many each rule removed, '
                                   'not which records they were.'))
        elif rules:
            parts.append(paragraph('The study declares exclusion rules (',
                                   join(*[code(str(rule.get('rule'))) for rule in rules if isinstance(rule, dict)],
                                        sep=', '),
                                   '), and how many records they removed is not recorded by any analysis or '
                                   'evaluation of this run.'))
        elif self.snapshot:
            parts.append(paragraph('The study declares no exclusion rules, so no records were excluded.'))
        else:
            parts.append(paragraph('Whether the study declares exclusion rules is not recorded: the run has no '
                                   'settings snapshot. No analysis or evaluation of this run recorded an exclusion.'))
        return block('Exclusions', *parts, key='exclusions')

    def freeze(self):
        snapshot, config, observed = self.snapshot, self.config, self.loaded['observed']
        status = snapshot.get('status')
        if snapshot.get('freezeForced'):
            skipped = _list(snapshot.get('forcedGatesSkipped'))
            state = join(el('strong', 'Frozen with force.'), ' Freezing fixes a study’s settings before any behavior '
                         'is measured, and normally every check must pass. These checks were skipped: ',
                         join(*[join(source._GATES.get(gate, 'a check this version cannot name'), ' (', code(gate), ')')
                                for gate in skipped], sep='; ') if skipped else 'not recorded',
                         '. SteerLab records a forced freeze permanently and treats the study as not citable. Say so '
                         'wherever these results are reported.')
        elif status in ('frozen', 'complete'):
            state = join('Frozen before this run, with every freeze check passed',
                         f' (frozen at {snapshot["frozenAt"]})' if snapshot.get('frozenAt') else '', '.')
        elif status:
            state = join(el('strong', 'Not frozen when this run was made'), ' (status ', code(status), '). Its settings '
                         'were not fixed before behavior was measured.')
        else:
            state = None
        pairs = [('Freeze state', state)]
        for entry in _list(snapshot.get('capabilityBatteryNotApplied')):
            if isinstance(entry, dict):
                pairs.append(('Capability battery exemption', source._battery_sentence(entry)))
        pairs += [('Settings hash stamped on the run', _name(self.context.get('experimentHash'))
                   if self.context.get('experimentHash') else None),
                  ('Freeze hash', code(snapshot['freezeHash']) if snapshot.get('freezeHash') else None),
                  ('Workspace commit at freeze', code(snapshot['gitCommit']) if snapshot.get('gitCommit') else None),
                  ('Software that froze the study', code(snapshot['appVersion']) if snapshot.get('appVersion') else None),
                  ('Run started', config.get('createdAt')),
                  ('Software that made the run', code(config['appVersion']) if config.get('appVersion') else None)]
        backend = []
        if observed['device']:
            backend.append(join('device ', join(*[code(device) for device in observed['device']], sep=', ')))
        precision = config.get('dtype') or (observed['dtype'][0] if observed['dtype'] else None)
        if precision:
            backend.append(join('numeric precision ', code(precision)))
        if config.get('platform'):
            backend.append(join('platform ', code(config['platform'])))
        pairs.append(('Back end', join(*backend, sep=', ') or None))
        environment = _dict(config.get('pythonEnvironment'))
        packages = _dict(environment.get('packages'))
        libraries = ([f'Python {environment["python"]}'] if isinstance(environment.get('python'), str) else []) + [
            f'{name} {packages[name]}' for name in ('torch', 'transformers', 'numpy', 'scipy')
            if isinstance(packages.get(name), str)]
        if libraries:
            pairs.append(('Libraries the run used', ', '.join(libraries)))
        files = [[code(entry['path']), entry['role'], code(entry['sha256']), str(entry['bytes'])]
                 for entry in self.stored['sources']]
        return block(
            'Freeze and identity', facts(pairs),
            table('Every file this page was made from', [Column('File'), Column('Read as'), Column('SHA-256'),
                                                        Column('Bytes', numeric=True)],
                  files, key='sources', row_header=False,
                  footnote='Paths are relative to the workspace. A file’s SHA-256 identifies its exact bytes, so a '
                           'reader with the workspace can confirm this page was made from these files.'),
            key='freeze')

    def limit_section(self):
        recorded = [join(entry['what'], ': ', entry['why'], '.') for entry in self.stored['notAvailable']]
        recorded += self.limits
        always = [
            'Response text is not on this page. results export writes every response, judgment, and transcript as '
            'tables and text files.',
            'Numbers are rounded to four significant digits for reading. The files listed under Freeze and identity '
            'hold every digit.',
            'This page reads the run’s newest analysis and its newest completed evaluation of each kind. Older ones '
            'are not shown.']
        return block(
            'Limits',
            el('h3', 'What the run did not store'),
            bullets(recorded) if recorded else paragraph('Nothing this page looks for was missing from the run.'),
            el('h3', 'What this page leaves out'),
            bullets(always),
            el('h3', 'What these results show'),
            paragraph('An effect here is a difference between conditions, in this model, on these items, under these '
                      'settings. An interval that does not include zero is a reason to look further, not proof that '
                      'the difference holds elsewhere. Report the study’s freeze state, its exclusions, and its '
                      'judges’ agreement with any result taken from this page.'),
            key='limits')

    def render(self):
        headline, design, effects = self.headline_section(), self.design(), self.effects()
        conditions, controls, judges = self.condition_summary(), self.controls(), self.judges()
        exclusions, freeze = self.exclusions(), self.freeze()
        limits = self.limit_section()   # last: the sections above record what they could not show
        first = self.first()
        entries = [('headline', 'Headline outcome'), ('design', 'What was asked and compared'), ('effects', 'Effects'),
                   ('conditions', 'Each condition'), ('controls', 'Controls'), ('judges', 'Judges'),
                   ('exclusions', 'Exclusions'), ('freeze', 'Freeze and identity'), ('limits', 'Limits')]
        models = self.loaded['observed']['modelID'] or (
            [self.snapshot['modelID']] if self.snapshot.get('modelID') else [])
        lead = (f'{_plural(len(self.responses), "response")} from {", ".join(models) or "an unrecorded model"} under '
                f'{_plural(len(self.conditions), "condition")}, in the run runs/{self.run}.')
        sources = [f'runs/{self.run}'] + ([self.context['analysisLabel']] if self.context.get('analysisLabel') else []) \
            + [f'runs/{name}' for _, name, _ in self.evaluations]
        return document(
            title=f'Study results: {self.study}', lead=lead,
            body=join(first, contents(entries), headline, design, effects, conditions, controls, judges, exclusions,
                      freeze, limits),
            styles=(charts.CHART_STYLE, charts.FOREST_STYLE, PAGE_STYLE),
            footer=paragraph('Made by SteerLab from ', join(*[code(path) for path in dict.fromkeys(sources)], sep=', '),
                             '. Every number is copied from what those folders store; nothing was recalculated. The '
                             'page is a single file. It loads nothing from the network.'))


PAGE_STYLE = """
p.headline{font-size:17px;margin:4px 0 12px}
#first{margin:20px 0}
#first ul{margin:6px 0 0}
"""


def headline_of(stored):
    """The outcome the page leads with, by the shared rule: the declared primary outcome when the run has it,
    else the first the run has in the default order. A completed evaluation supplies the judged outcome."""
    context = stored['context']
    judged = any(context.get(kind + 'Name') for kind in ('paired', 'coding'))
    return headline_outcome.select(_dict(context.get('snapshot')).get(headline_outcome.MANIFEST_KEY),
                                   context.get('outcomes') or [], [headline_outcome.JUDGED] if judged else [])


def render(stored):
    """The page for one reading of a study's stored results: the same reading in, the same text out."""
    return Page(stored).render()
