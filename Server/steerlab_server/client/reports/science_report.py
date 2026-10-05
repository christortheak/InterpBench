"""``science report``: find a stored report, choose where its page goes, and write it.

A run directory is evidence. Custody re-reads every file in it, and a file
added later would make that check fail, so nothing here writes inside one. A
new job's page is written by the engine before the run is marked complete and
is simply found. For any other run the page is drawn on request and saved
outside ``runs/``.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ...experiment import diagnostic_archives as archives
from . import jlens_assessment
from .page import MARKER

#: A report is a summary, not a tensor. One larger than this is not one of ours.
MAX_REPORT_BYTES = 256 * 1024 ** 2
DEFAULT_FOLDER = 'reports'


class ReportRefusal(archives.Refusal):
    code = 'reportRefused'

    def __init__(self, reason, repair):
        super().__init__(reason)
        self.repair_action = repair


@dataclass(frozen=True)
class Kind:
    id: str
    title: str
    report_name: str
    page_name: str
    render: object


#: Stored ``operation`` to the page that draws it. A later page (a study's
#: results, say) adds its line here.
KINDS = {jlens_assessment.OPERATION: Kind('jlens-assessment', 'J-lens assessment', jlens_assessment.REPORT_NAME,
                                          jlens_assessment.PAGE_NAME, jlens_assessment.render)}
CHOOSE = ('Give a run folder that holds assessment-report.json, or that file itself. '
          'This command draws J-lens assessment reports (operation jlens-fit-assess).')


def _inside(path, parent):
    return path == parent or parent in path.parents


def _source(path, root):
    selected = Path(path)
    selected = selected if selected.is_absolute() else root / selected
    if not selected.exists():
        raise ReportRefusal(f'Nothing was found at {path} in this workspace.', CHOOSE)
    selected = selected.resolve()
    if not _inside(selected, root):
        raise ReportRefusal('Choose a report inside this workspace.',
                            'Copy the report file into the workspace, or name the workspace that holds it '
                            '(--root for steerlab, --workspace for steerlab-cli).')
    if selected.is_dir():
        found = [selected / kind.report_name for kind in KINDS.values() if (selected / kind.report_name).is_file()]
        if not found:
            raise ReportRefusal('This folder holds no report that can be drawn as a page.', CHOOSE)
        selected = found[0]
    if not selected.is_file():
        raise ReportRefusal('The report must be an ordinary file.', CHOOSE)
    if selected.stat().st_size > MAX_REPORT_BYTES:
        raise ReportRefusal('This file is too large to be a report.', CHOOSE)
    return selected


def _kind(data):
    try:
        document = json.loads(data)
    except ValueError as exc:
        raise ReportRefusal('This file is not readable JSON, so no page was made.', CHOOSE) from exc
    if isinstance(document, dict) and document.get('operation') in KINDS:
        return KINDS[document['operation']]
    if isinstance(document, dict) and 'comparisonSHA256' in document and 'operation' not in document:
        raise ReportRefusal('This file is one comparison taken from an assessment, not the whole report.',
                            'Give the run folder, or the assessment-report.json that sits beside the comparisons folder.')
    raise ReportRefusal('This is not a report this command can draw as a page.', CHOOSE)


def _destination(out, root, default):
    """Where the page goes. Never inside ``runs/``, a completed run, or over a file that is not one of these pages."""
    if out is None:
        target = default
    else:
        if not isinstance(out, str) or not out:
            raise ReportRefusal('--out needs a file name.', 'For example: --out reports/my-report.html')
        target = Path(out).expanduser()
        target = target if target.is_absolute() else root / target
    target = target.parent.resolve() / target.name
    elsewhere = f'Choose a file outside any run folder, for example {DEFAULT_FOLDER}/{target.name}.'
    if _inside(target, root / 'runs'):
        raise ReportRefusal('A page is never written inside runs/. A run folder is evidence, and adding a file '
                            'to it would break its custody check.', elsewhere)
    for folder in target.parents:
        if (folder / 'COMPLETED').exists():
            raise ReportRefusal('That location is inside a completed run folder, which is never written to.', elsewhere)
    if target.is_symlink() or target.is_dir():
        raise ReportRefusal('--out must name a file, not a folder or a link.', elsewhere)
    if target.exists():
        with open(target, 'rb') as handle:
            if handle.read(len(MARKER)) != MARKER.encode():
                raise ReportRefusal('Another file is already at that location, and it is not a report page. '
                                    'Nothing was changed.', 'Choose a different --out, or move that file yourself.')
    return target


def _write(target, data):
    if target.exists() and target.read_bytes() == data:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix='.' + target.name + '.')
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return True


def render(path, root, out=None):
    """Make, or find, the page for one stored report. Returns where it is and what it was made from."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ReportRefusal('The workspace folder does not exist.', 'Name an existing workspace.')
    source = _source(path, root)
    data = source.read_bytes()
    kind = _kind(data)
    digest = hashlib.sha256(data).hexdigest()
    in_run = source.name == kind.report_name
    stored = source.with_name(kind.page_name) if in_run else None
    result = {'kind': kind.id, 'title': kind.title, 'sourceReport': source.relative_to(root).as_posix(),
              'sourceSHA256': digest}
    if out is None and stored is not None and stored.is_file() and not stored.is_symlink():
        # The engine wrote this page with the run. It is the same file the evidence carries.
        return {**result, 'htmlPath': str(stored), 'htmlSHA256': archives.file_hash(stored), 'rendered': False,
                'changed': False, 'inRunDirectory': True,
                'message': 'The run already holds its report page. Nothing was written.'}
    run = source.parent
    name = run.name if in_run and run.parent == root / 'runs' else 'report-' + digest[:12]
    target = _destination(out, root, root / DEFAULT_FOLDER / name / kind.page_name)
    try:
        page = kind.render(data).encode('utf-8')
    except jlens_assessment.ReportError as exc:
        raise ReportRefusal(str(exc), CHOOSE) from exc
    changed = _write(target, page)
    return {**result, 'htmlPath': str(target), 'htmlSHA256': hashlib.sha256(page).hexdigest(), 'rendered': True,
            'changed': changed, 'inRunDirectory': False,
            'message': ('The report page was written. ' if changed else 'The report page was already up to date. ')
                       + 'The run folder was not changed.'}
