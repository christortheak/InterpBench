"""The page frame and its building blocks. Nothing here knows the subject.

A page is one HTML file with its styles inline: no script, no image, no font,
and no link that leaves the file. It reads in light and dark appearance and on
paper. Every string that comes from stored data passes through :func:`text`,
so a lens ID or a file name can never become markup.
"""
from dataclasses import dataclass
from html import escape as _escape

#: Bumped when the same input would give different bytes, so a reader can tell
#: which layout drew a page. It is not the product version.
PAGE_FORMAT = 1
#: The first bytes of every page. A writer checks for them before replacing a
#: file, so it only ever replaces one of its own pages.
MARKER = '<!doctype html>\n<!-- SteerLab report page -->\n'
NO_VALUE = 'no value'


class Markup(str):
    """Text that is already HTML. Any other string is escaped where it is placed."""
    __slots__ = ()


def text(value):
    """Escape a stored string for use as text or as an attribute value."""
    return value if isinstance(value, Markup) else Markup(_escape(str(value), quote=True))


def _flat(parts):
    for part in parts:
        if part is None or part is False:
            continue
        if isinstance(part, (list, tuple)) or hasattr(part, '__next__'):
            yield from _flat(part)
        else:
            yield part


def join(*parts, sep=''):
    return Markup(text(sep).join(text(part) for part in _flat(parts)))


_VOID = frozenset({'meta', 'br', 'hr', 'col'})
_SHAPES = frozenset({'line', 'circle', 'path', 'rect'})
#: A line break after these keeps the file readable and its diffs small.
_BLOCK = frozenset({
    'html', 'head', 'body', 'title', 'meta', 'style', 'header', 'main', 'footer', 'nav', 'section', 'div',
    'h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'caption', 'thead', 'tbody',
    'tr', 'th', 'td', 'figure', 'figcaption', 'svg', 'g', 'desc', 'text', 'line', 'circle', 'path', 'rect', 'hr'})


def el(tag, *children, **attributes):
    """One element. ``class_`` is ``class``; other underscores become hyphens."""
    parts = ['<' + tag]
    for key, value in attributes.items():
        if value is None or value is False:
            continue
        name = key.rstrip('_').replace('_', '-')
        parts.append(' ' + name if value is True else f' {name}="{text(value)}"')
    body = join(*children)
    end = '\n' if tag in _BLOCK else ''
    if tag in _VOID:
        return Markup(''.join(parts) + '>' + end)
    if tag in _SHAPES and not body:
        return Markup(''.join(parts) + '/>' + end)
    opener = '>\n' if tag in ('html', 'head', 'body', 'main', 'section', 'table', 'thead', 'tbody', 'tr', 'ul', 'ol',
                              'dl', 'figure', 'svg', 'g', 'nav', 'header', 'footer') else '>'
    return Markup(''.join(parts) + opener + body + f'</{tag}>' + end)


def code(value):
    """An identifier, hash, or path: fixed width, shown whole, and never cut.

    A long one (a hash) may wrap at any character so that it fits a narrow
    column. A short one stays in one piece.
    """
    return el('code', value, class_='long' if len(str(value)) >= 40 else None)


def paragraph(*children, class_=None):
    return el('p', *children, class_=class_)


def bullets(items, *, class_=None):
    return el('ul', [el('li', item) for item in items], class_=class_)


def facts(pairs):
    """Label and value pairs. A value of ``None`` is shown as not recorded."""
    return el('dl', [(el('dt', label), el('dd', 'not recorded' if value is None else value))
                     for label, value in pairs], class_='facts')


def note(*children, label='Note'):
    """A set-off remark. The label is words, so it does not depend on colour."""
    return el('div', el('p', el('strong', label + ': '), *children), class_='note')


def block(heading, *children, key=None, level=2):
    """A headed part of the page. Level 2 is a card; deeper levels sit inside one."""
    title = el(f'h{level}', heading, id=key + '-h' if key else None)
    if level == 2:
        return el('section', title, *children, id=key, aria_labelledby=key + '-h' if key else None)
    return el('div', title, *children, id=key, class_='sub')


def contents(entries):
    """In-page links only: ``(key, label)`` pairs."""
    return el('nav', el('ol', [el('li', el('a', label, href='#' + key)) for key, label in entries]),
              aria_label='Contents', class_='contents')


def link(key, label):
    return el('a', label, href='#' + key)


@dataclass(frozen=True)
class Column:
    label: object
    numeric: bool = False
    #: Adjacent columns with the same group share one spanning header above them.
    group: object = None


def _header(columns):
    if not any(column.group is not None for column in columns):
        return el('tr', [el('th', column.label, scope='col', class_='num' if column.numeric else None)
                         for column in columns])
    upper, lower, index = [], [], 0
    while index < len(columns):
        column = columns[index]
        if column.group is None:
            upper.append(el('th', column.label, scope='col', rowspan=2, class_='num' if column.numeric else None))
            index += 1
            continue
        span = index
        while span < len(columns) and columns[span].group == column.group:
            lower.append(el('th', columns[span].label, scope='col', class_='num' if columns[span].numeric else None))
            span += 1
        upper.append(el('th', column.group, scope='colgroup', colspan=span - index, class_='group'))
        index = span
    return join(el('tr', upper), el('tr', lower))


def table(caption, columns, rows, *, key, footnote=None, row_header=True):
    """A real table: a caption, scoped headers, and every row it is given.

    ``rows`` is a list of cell lists. A cell of ``None`` reads "no value".
    The wrapper scrolls sideways on a narrow screen instead of cutting columns.
    """
    body = []
    for row in rows:
        cells = []
        for index, (column, cell) in enumerate(zip(columns, row)):
            value = el('span', NO_VALUE, class_='none') if cell is None else cell
            kind = 'num' if column.numeric else None
            cells.append(el('th', value, scope='row', class_=kind) if row_header and index == 0
                         else el('td', value, class_=kind))
        body.append(el('tr', cells))
    drawn = el('table', el('caption', caption, id=key + '-caption'), el('thead', _header(columns)),
               el('tbody', body), id=key)
    return join(el('div', drawn, class_='table-wrap', role='region', aria_labelledby=key + '-caption', tabindex=0),
                el('p', footnote, class_='table-note') if footnote is not None else None)


BASE_STYLE = """
:root{color-scheme:light dark;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--wash:#f0efec}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--wash:#242422}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header.masthead,main,footer{max-width:1040px;margin:0 auto;padding:0 20px}
header.masthead{padding-top:28px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px}
h2{font-size:19px;margin:0 0 12px}
h3{font-size:16px;margin:22px 0 8px}
h4{font-size:14px;margin:18px 0 6px}
p{margin:0 0 10px}
.lead{color:var(--ink-2);margin:0}
section{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px 24px;margin:20px 0}
.sub{border-top:1px solid var(--grid);margin-top:20px}
a{color:inherit;text-decoration:underline;text-decoration-color:var(--axis);text-underline-offset:2px}
code{font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:nowrap}
code.long{white-space:normal;overflow-wrap:anywhere}
ul,ol{margin:0 0 10px;padding-left:22px}
li{margin:2px 0}
nav.contents ol{columns:2;column-gap:32px;margin:14px 0 0}
dl.facts{display:grid;grid-template-columns:minmax(150px,230px) 1fr;gap:4px 18px;margin:0 0 12px}
dl.facts dt{color:var(--ink-2)}
dl.facts dd{margin:0;min-width:0}
.note{background:var(--wash);border-left:3px solid var(--axis);padding:8px 12px;margin:12px 0}
.note p{margin:0}
.table-wrap{overflow-x:auto;margin:10px 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
caption{text-align:left;font-weight:600;padding:0 0 6px}
th,td{padding:4px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--grid);font-weight:400}
thead th{color:var(--ink-2);border-bottom:1px solid var(--axis);vertical-align:bottom}
thead th.group{text-align:center;border-bottom:1px solid var(--grid)}
.num{text-align:right;white-space:nowrap}
thead th.num{white-space:normal}
td strong{font-weight:700}
.none{color:var(--muted)}
.table-note{color:var(--ink-2);font-size:13px;margin:4px 0 12px}
footer{color:var(--ink-2);font-size:13px;padding-top:8px;padding-bottom:48px}
@media (max-width:640px){dl.facts{grid-template-columns:1fr;gap:0 0}dl.facts dd{margin-bottom:8px}
nav.contents ol{columns:1}section{padding:16px}}
@media print{:root{--page:#fff;--surface:#fff;--ink:#000;--ink-2:#222;--muted:#555;--grid:#ccc;--axis:#888;
--border:#999;--wash:#f3f3f3}
body{font-size:11pt}header.masthead,main,footer{max-width:none;padding:0}
section{border:0;border-radius:0;padding:0;margin:18pt 0}
h2,h3,h4,caption{break-after:avoid}tr,figure,.note{break-inside:avoid}thead{display:table-header-group}
.table-wrap{overflow:visible}table{font-size:9pt}a{text-decoration:none}}
"""


def document(*, title, lead=None, body, styles=(), footer=None, language='en'):
    """The whole file. ``styles`` adds the style sheets of the parts used on the page."""
    policy = "default-src 'none'; style-src 'unsafe-inline'"
    head = el('head',
              el('meta', charset='utf-8'),
              el('meta', name='viewport', content='width=device-width, initial-scale=1'),
              # The browser itself then refuses any request the page might try to make.
              el('meta', http_equiv='Content-Security-Policy', content=Markup(policy)),
              el('meta', name='generator', content=f'SteerLab report page, format {PAGE_FORMAT}'),
              el('title', title),
              el('style', Markup(''.join([BASE_STYLE, *styles]))))
    page = el('body',
              el('header', el('h1', title), el('p', lead, class_='lead') if lead is not None else None,
                 class_='masthead'),
              el('main', body),
              el('footer', footer) if footer is not None else None)
    return MARKER + el('html', head, page, lang=language)
