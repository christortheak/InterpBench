"""The ``results`` family: take a study's stored results elsewhere.

``results export`` reads a completed run with its analysis and evaluation and
writes tables, transcripts, a methods summary, and a codebook into a new
folder. It runs no model and changes nothing under ``runs/``. The work is
:mod:`steerlab_server.client.results_export`; the Mac command line and the app
reach the same function through the local bridge
(``diagnostic_commands.workspace_action('results-export', …)``).
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
)

#: The bridge action name. Swift twin: ``ResultsExport.action``.
BRIDGE_ACTION = 'results-export'


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


def bridge(payload):
    """The ``results-export`` bridge action.

    A refusal comes back as DATA (``exported: false`` beside its code, reason,
    repair, and state) rather than as a failed call, so the Mac command line
    can answer with the same typed refusal this client gives.
    """
    from ..experiment.diagnostic_archives import MalformedRequest
    from . import results_export
    allowed = {'workspaceRoot', 'study', 'run', 'out', 'client'}
    if (not isinstance(payload, dict) or payload.keys() - allowed
            or not {'workspaceRoot', 'study'} <= payload.keys()
            or any(not isinstance(payload[key], str) or not payload[key] for key in payload)):
        raise MalformedRequest('Results export takes workspaceRoot and study, with optional run, out, and client, '
                               'each a nonempty string.')
    try:
        result = results_export.export_results(
            payload['workspaceRoot'], payload['study'], run=payload.get('run'), out=payload.get('out'),
            client=payload.get('client', results_export.MAC_CLIENT))
    except results_export.ResultsExportRefusal as exc:
        return {'exported': False, 'changed': False,
                'refusal': {'code': exc.code, 'reason': exc.reason, 'repairAction': exc.repair_action,
                            'state': exc.state}}
    return {'exported': True, **result}


def run(invocation):
    from ..client_cli import ClientRefusal
    from ..experiment import paths
    from . import results_export
    if len(invocation.positionals) != 1:
        raise ClientRefusal(code='usage', reason='Name one study to export.',
                            repair_action='steerlab results export <study>  (steerlab experiment list shows '
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
        result = results_export.export_results(root, invocation.positionals[0], run=chosen, out=destination,
                                               client=results_export.PYTHON_CLIENT)
    except results_export.ResultsExportRefusal as exc:
        raise ClientRefusal(code=exc.code, reason=exc.reason, repair_action=exc.repair_action,
                            state=exc.state) from exc
    print('\n'.join(summary_lines(result)))
    return CLIResult(message=message(result), changed=True, payload=result)
