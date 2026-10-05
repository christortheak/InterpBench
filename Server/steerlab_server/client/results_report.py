"""``results report``: a study's stored results as one readable page, written outside ``runs/``.

The page is :mod:`reports.study_results`, drawn from the same reading
``results export`` makes (:func:`results_export.read_results`), so the page
this writes and the ``report.html`` an export carries are the same bytes for
the same run, analysis, and evaluation. Nothing here reads a run a second way,
recalculates a statistic, or writes inside a run directory: a page goes under
the workspace's ``reports/`` by default, or to the file ``out`` names, and
only ever replaces an earlier page of its own.

Standard library only, like the export: it works on an authoring-only install.
"""
import hashlib
from pathlib import Path

from . import results_export
from .reports import science_report, study_results

#: Where a page goes when no file is named: ``reports/<study>/<run>/report.html``.
REPORTS_DIRECTORY = science_report.DEFAULT_FOLDER
PAGE_NAME = study_results.PAGE_NAME
#: A destination the page may not be written to: inside runs/, inside a
#: completed run folder, or over a file that is not one of these pages.
DESTINATION_REFUSED_CODE = 'reportDestinationRefused'


def write_report(root, study, *, run=None, out=None, client=results_export.PYTHON_CLIENT):
    """Write the results page for ``study`` and say where it is.

    ``run`` is chosen exactly as ``results export`` chooses it. ``out`` names
    the page's file; a relative one is read from the workspace root. Raises
    :class:`results_export.ResultsExportRefusal` before anything is written
    when the request cannot be met.
    """
    stored = results_export.read_results(root, study, run=run, client=client)
    workspace = Path(root).resolve()
    default = workspace / REPORTS_DIRECTORY / study / stored['runName'] / PAGE_NAME
    try:
        # The rule every page writer keeps: never inside runs/ or a completed
        # run folder, and never over a file that is not one of these pages.
        target = science_report._destination(out, workspace, default)
    except science_report.ReportRefusal as exc:
        raise results_export.ResultsExportRefusal(
            str(exc), code=DESTINATION_REFUSED_CODE,
            repair_action=exc.repair_action) from exc
    data = study_results.render(stored).encode('utf-8')
    changed = science_report._write(target, data)
    context = stored['context']
    snapshot = context['snapshot']
    return {
        'study': study,
        'run': 'runs/' + stored['runName'],
        'analysis': context['analysisLabel'],
        'evaluations': {kind: ('runs/' + name if name else None) for kind, name in stored['names'].items()},
        'headline': study_results.headline_of(stored).as_payload(),
        'htmlPath': str(target),
        'htmlSHA256': hashlib.sha256(data).hexdigest(),
        'changed': changed,
        'notAvailable': stored['notAvailable'],
        'freezeForced': bool(snapshot.get('freezeForced')),
        'capabilityBatteryNotApplied': list(snapshot.get('capabilityBatteryNotApplied') or []),
    }


def message(result):
    text = ('The results page was written to ' if result['changed'] else
            'The results page was already up to date at ') + result['htmlPath'] + '.'
    if result['notAvailable']:
        text += ' Not available: ' + ', '.join(entry['what'] for entry in result['notAvailable']) + '.'
    return text + ' Nothing under runs/ was changed, and no statistic was recalculated.'


def summary_lines(result):
    """What a person at a terminal reads after the page is written."""
    lines = [f"Results page for {result['study']}: {result['htmlPath']}",
             f"  from {result['run']}" + (f", analysis {result['analysis']}" if result.get('analysis') else '')]
    headline = result.get('headline') or {}
    if headline.get('outcome'):
        lines.append(f"  headline outcome: {headline['outcome']}, {headline['chosenBy']}")
    for entry in result['notAvailable']:
        lines.append(f"  not available: {entry['what']} ({entry['why']})")
    if result.get('freezeForced'):
        lines.append('  note: this study was frozen with force; the page says so first.')
    lines.append('  Open the file in a web browser. It is one self-contained page you can send as it is.')
    return lines
