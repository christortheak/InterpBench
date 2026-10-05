"""The ``results`` family: take a study's stored results elsewhere.

``results export`` reads a completed run with its analysis and evaluation and
writes tables, transcripts, a methods summary, a codebook, and the results page
into a new folder. ``results report`` writes the results page alone, as one
HTML file outside ``runs/``. Neither runs a model or changes anything under
``runs/``. The work is :mod:`steerlab_server.client.results_export` and
:mod:`steerlab_server.client.results_report`; the Mac command line and the app
reach the same functions through the local bridge
(``diagnostic_commands.workspace_action('results-export' | 'results-report', …)``).
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
)

#: The bridge action names. Swift twins: ``ResultsExport.action``, ``ResultsReport.action``.
BRIDGE_ACTION = 'results-export'
REPORT_BRIDGE_ACTION = 'results-report'


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


def _checked(payload, what):
    from ..experiment.diagnostic_archives import Refusal
    allowed = {'workspaceRoot', 'study', 'run', 'out', 'client'}
    if (not isinstance(payload, dict) or payload.keys() - allowed
            or not {'workspaceRoot', 'study'} <= payload.keys()
            or any(not isinstance(payload[key], str) or not payload[key] for key in payload)):
        raise Refusal(f'{what} takes workspaceRoot and study, with optional run, out, and client, '
                      'each a nonempty string.')
    return payload


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


def run(invocation):
    from ..client_cli import ClientRefusal
    from ..experiment import paths
    from . import results_export, results_report
    verb = invocation.spec.verb
    if len(invocation.positionals) != 1:
        doing = 'export' if verb == 'export' else 'make a results page for'
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
