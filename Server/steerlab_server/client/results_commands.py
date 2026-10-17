"""The ``results`` family: take a study's stored results elsewhere.

``results export`` reads a completed run with its analysis and evaluation and
writes tables, transcripts, a methods summary, a codebook, and the results page
into a new folder. ``results report`` writes the results page alone, as one
HTML file outside ``runs/``. ``results check-choices`` reads a run's stored
choices again with the current choice parser and reports, per condition, how
many would read differently; it writes nothing. None of them runs a model or
changes anything under ``runs/``. The work is
:mod:`steerlab_server.client.results_export`,
:mod:`steerlab_server.client.results_report`, and
:mod:`steerlab_server.experiment.choice_reread`; the Mac command line and the
app reach the same functions through the local bridge
(``diagnostic_commands.workspace_action('results-export' | 'results-report' |
'results-check-choices', …)``).
"""
import os

from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec('results', 'export', positional='<study>',
             purpose="Export a completed run's results into a new folder outside runs/: tables that open in R, "
                     'Stata, SPSS, or a spreadsheet, transcripts for coding by hand, a methods summary, and a '
                     'codebook. Uses the newest completed run with its newest analysis and evaluation unless '
                     '--run names another. Runs no model, and recalculates nothing.',
             value_flags=frozenset({'--run', '--out'})),
    VerbSpec('results', 'report', positional='<study>',
             purpose="Write a completed run's results as one readable page (HTML) outside runs/: the headline "
                     "outcome with its interval, every stored effect, each condition, controls, the judges' "
                     'agreement, exclusions, and how the study was frozen. Uses the newest completed run with its '
                     'newest analysis and evaluation unless --run names another. The page goes under reports/ in '
                     'the workspace unless --out names the file. Runs no model, and recalculates nothing.',
             value_flags=frozenset({'--run', '--out'})),
    VerbSpec('results', 'check-choices', positional='<study>',
             purpose="Read a run's stored choices again with the current choice parser and report, per condition, "
                     'how many would read differently: changed to another option, changed to no answer, or gained '
                     'an answer, with a few records of each kind to look at. Uses the newest completed run unless '
                     '--run names another. Writes nothing and changes nothing under runs/; to analyze with the '
                     'current reading, analyze the run with --reread-choices.',
             value_flags=frozenset({'--run'})),
)

#: The bridge action names. Swift twins: ``ResultsExport.action``, ``ResultsReport.action``,
#: ``ChoiceCheck.action``.
BRIDGE_ACTION = 'results-export'
REPORT_BRIDGE_ACTION = 'results-report'
CHECK_CHOICES_BRIDGE_ACTION = 'results-check-choices'


def summary_lines(result):
    """What a person at a terminal reads after an export."""
    lines = [f"Exported {result['study']} to {result['exportDirectory']}",
             f"  from {result['run']}"
             + (f", analysis {result['analysis']}" if result.get('analysis') else '')]
    for entry in result['files']:
        lines.append(f"  {entry['file']}" + (f"  ({entry['rows']} rows)" if 'rows' in entry else ''))
    for entry in result['notAvailable']:
        lines.append(f"  not available: {entry['what']} ({entry['why']})")
    if result.get('freezeForced'):
        lines.append('  note: this study was frozen with force; methods.md says so.')
    return lines


def message(result):
    text = f"Results exported to {result['exportDirectory']}."
    if result['notAvailable']:
        text += ' Not available: ' + ', '.join(entry['what'] for entry in result['notAvailable']) + '.'
    return text + ' Nothing under runs/ was changed, and no statistic was recalculated.'


def _checked(payload, what, *, allowed=frozenset({'workspaceRoot', 'study', 'run', 'out', 'client'})):
    from ..experiment.diagnostic_archives import MalformedRequest
    if (not isinstance(payload, dict) or payload.keys() - allowed
            or not {'workspaceRoot', 'study'} <= payload.keys()
            or any(not isinstance(payload[key], str) or not payload[key] for key in payload)):
        # A request in the wrong shape is malformed (blocked/64), not a refusal.
        optional = ', '.join(sorted(allowed - {'workspaceRoot', 'study'}))
        raise MalformedRequest(f'{what} takes workspaceRoot and study, with optional {optional}, '
                               'each a nonempty string.')
    return payload


def check_choices(root, study, *, run=None, client=None):
    """The report ``results check-choices`` answers with: the run chosen as
    ``results export`` chooses it, and every stored choice in it read again
    with the current parser. Raises :class:`results_export.ResultsExportRefusal`
    (carrying the refusing gate as its code) when the run or its options
    cannot be established. Nothing is written."""
    from ..experiment import choice_reread, lifecycle_gates
    from . import results_export
    root, run_name, _steps = results_export.selected_run(
        root, study, run=run, client=client or results_export.PYTHON_CLIENT)
    try:
        report = choice_reread.check(os.path.join(root, 'runs', run_name), root, study=study)
    except lifecycle_gates.LifecycleError as exc:
        raise results_export.ResultsExportRefusal(
            str(exc), code=exc.gate or 'refused', repair_action=exc.repair_action) from exc
    return {'study': study, 'changed': False, **report, 'run': 'runs/' + run_name}


#: How each kind of change is said to a person.
CHANGE_LABELS = {'changedToOtherOption': 'to another option', 'changedToNoAnswer': 'to no answer',
                 'gainedAnswer': 'gained an answer'}


def check_lines(result):
    """What a person at a terminal reads after a check."""
    lines = [f"Checked {result['study']}, {result['run']}, with choice parser version {result['parserVersion']}",
             f"  task prompts: {result['taskPrompts']['file']} (read from {result['taskPrompts']['readFrom']})"]
    for name, counts in [('all conditions', result['total'])] + sorted(result['conditions'].items()):
        lines.append(f"  {name}: {counts['storedChoices']} stored, {counts['unchanged']} unchanged, "
                     f"{counts['changedToOtherOption']} to another option, {counts['changedToNoAnswer']} to no "
                     f"answer, {counts['gainedAnswer']} gained an answer")
        if counts['noFinishReason']:
            lines.append(f"    {counts['noFinishReason']} from an older engine that did not record whether a "
                         "reply was cut off (read as complete when a choice was stored, as possibly cut "
                         f"off when none was; {counts['noFinishReasonChanged']} changed)")
    for kind, label in CHANGE_LABELS.items():
        for entry in result['examples'][kind]:
            lines.append(f"  e.g. {label}: line {entry['line']}, {entry['condition']}, item {entry['promptID']}"
                         + (f", sample {entry['sampleIndex']}" if 'sampleIndex' in entry else '')
                         + f": {_reading(entry['stored'])} -> {_reading(entry['reread'])}")
    return lines


def _reading(value):
    """One stored or re-read choice, said for a person."""
    return 'no answer' if value is None else str(value)


def check_message(result, *, analyze_command):
    """One sentence for a person, naming the asking client's analyze command."""
    total = result['total']
    if not total['changed']:
        return (f"All {total['storedChoices']} stored choices in {result['run']} read the same with choice parser "
                f"version {result['parserVersion']}. Nothing was written.")
    return (f"{total['changed']} of {total['storedChoices']} stored choices in {result['run']} would read "
            f"differently with choice parser version {result['parserVersion']}. Nothing was written. To analyze "
            f"with the current reading: {analyze_command}")


def _refusal(done, exc):
    """A refusal as DATA beside ``done: false``, so the Mac side answers with this client's typed refusal."""
    return {done: False, 'changed': False,
            'refusal': {'code': exc.code, 'reason': exc.reason, 'repairAction': exc.repair_action,
                        'state': exc.state}}


def bridge(payload):
    """The ``results-export`` bridge action.

    A refusal comes back as DATA (``exported: false`` beside its code, reason,
    repair, and state) rather than as a failed call, so the Mac command line
    can answer with the same typed refusal this client gives.
    """
    from . import results_export
    payload = _checked(payload, 'Results export')
    try:
        result = results_export.export_results(
            payload['workspaceRoot'], payload['study'], run=payload.get('run'), out=payload.get('out'),
            client=payload.get('client', results_export.MAC_CLIENT))
    except results_export.ResultsExportRefusal as exc:
        return _refusal('exported', exc)
    return {'exported': True, **result}


def report_bridge(payload):
    """The ``results-report`` bridge action: the same page, with refusals as data in the same shape."""
    from . import results_export, results_report
    payload = _checked(payload, 'The results page')
    try:
        result = results_report.write_report(
            payload['workspaceRoot'], payload['study'], run=payload.get('run'), out=payload.get('out'),
            client=payload.get('client', results_export.MAC_CLIENT))
    except results_export.ResultsExportRefusal as exc:
        return _refusal('written', exc)
    return {'written': True, **result}


def check_bridge(payload):
    """The ``results-check-choices`` bridge action, with refusals as data in
    the shape ``results-export`` uses."""
    from . import results_export
    payload = _checked(payload, 'The choice check',
                       allowed=frozenset({'workspaceRoot', 'study', 'run', 'client'}))
    try:
        result = check_choices(payload['workspaceRoot'], payload['study'], run=payload.get('run'),
                               client=payload.get('client', results_export.MAC_CLIENT))
    except results_export.ResultsExportRefusal as exc:
        return _refusal('checked', exc)
    return {'checked': True, **result}


def run(invocation):
    from ..client_cli import ClientRefusal
    from ..experiment import paths
    from . import results_export, results_report
    verb = invocation.spec.verb
    if len(invocation.positionals) != 1:
        doing = {'export': 'export', 'check-choices': 'check'}.get(verb, 'make a results page for')
        raise ClientRefusal(code='usage', reason=f'Name one study to {doing}.',
                            repair_action=f'steerlab results {verb} <study>  (steerlab experiment list shows '
                                          'the studies in this workspace)')
    root = paths.project_root()
    chosen = invocation.one('--run')
    if (chosen and not os.path.isabs(chosen) and os.path.isdir(chosen)
            and not os.path.isdir(os.path.join(root, chosen))
            and not os.path.isdir(os.path.join(root, 'runs', chosen))):
        # A path typed relative to the current directory rather than to the workspace.
        chosen = os.path.abspath(chosen)
    if verb == 'check-choices':
        study = invocation.positionals[0]
        try:
            result = check_choices(root, study, run=chosen, client=results_export.PYTHON_CLIENT)
        except results_export.ResultsExportRefusal as exc:
            raise ClientRefusal(code=exc.code, reason=exc.reason, repair_action=exc.repair_action,
                                state=exc.state) from exc
        print('\n'.join(check_lines(result)))
        return CLIResult(message=check_message(
            result, analyze_command=f'steerlab run {study} --runner <url> --verb analyze --reread-choices'),
            changed=False, payload=result)
    destination = invocation.one('--out')
    if destination is not None:
        destination = os.path.abspath(os.path.expanduser(destination))
    try:
        if verb == 'report':
            result = results_report.write_report(root, invocation.positionals[0], run=chosen, out=destination,
                                                 client=results_export.PYTHON_CLIENT)
        else:
            result = results_export.export_results(root, invocation.positionals[0], run=chosen,
                                                   out=destination, client=results_export.PYTHON_CLIENT)
    except results_export.ResultsExportRefusal as exc:
        raise ClientRefusal(code=exc.code, reason=exc.reason, repair_action=exc.repair_action,
                            state=exc.state) from exc
    if verb == 'report':
        print('\n'.join(results_report.summary_lines(result)))
        return CLIResult(message=results_report.message(result), changed=result['changed'], payload=result)
    print('\n'.join(summary_lines(result)))
    return CLIResult(message=message(result), changed=True, payload=result)
