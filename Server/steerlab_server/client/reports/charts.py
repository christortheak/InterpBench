"""Inline SVG line charts with a text alternative. No script; nothing is fetched.

A chart draws the values it is given and nothing else: no smoothing, no fitted
line, no average. A missing value is a gap in the line. Every drawn value is
also reachable without the picture: the chart carries a title and a text
description, each point names itself on hover, and the caller places the same
numbers in a table beside it.

Line identity never rests on colour alone. The legend names each line beside
its sample, the plain baseline is dotted and the reference is the page's own
ink, and on paper (or with forced colours) every coloured line also takes its
own dash pattern.
"""
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
import math

from .page import el, join

BASELINE, REFERENCE = 'baseline', 'reference'
#: Coloured lines per chart. More than this would repeat a colour, so the
#: caller splits the lines over several charts instead.
SLOTS = 8

WIDTH, HEIGHT = 720, 312
LEFT, RIGHT, TOP, BOTTOM = 52, 706, 14, 262


@dataclass(frozen=True)
class Series:
    label: str
    #: ``(x, y)`` pairs in x order; ``y`` is ``None`` where no value is stored.
    points: tuple
    #: ``BASELINE``, ``REFERENCE``, or a colour slot from 0 to ``SLOTS - 1``.
    style: object = 0

    @property
    def css(self):
        return {BASELINE: 'base', REFERENCE: 'ref'}.get(self.style, f'c{self.style}')


def _plain(value):
    """A tick label without trailing zeros or an exponent."""
    value = value.normalize()
    return format(value if value else Decimal(0), 'f')


def ticks_to(maximum):
    """Round ticks from 0 to at least ``maximum``: at most five steps of 1, 2, 2.5, or 5 times a power of ten.

    Decimal arithmetic, so the same input gives the same ticks on every machine.
    """
    if not (isinstance(maximum, (int, float)) and math.isfinite(maximum) and maximum > 0):
        return [Decimal(0), Decimal(1)]
    value = Decimal(repr(float(maximum)))
    for power in (value.adjusted() - 1, value.adjusted(), value.adjusted() + 1):
        for mantissa in ('1', '2', '2.5', '5'):
            step = Decimal(mantissa).scaleb(power)
            count = math.ceil(value / step)
            if count <= 5:
                return [step * index for index in range(count + 1)]
    raise AssertionError('unreachable: a step of ten times the value always fits')


def _x_ticks(low, high):
    """Whole-number ticks, at most thirteen of them."""
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        first = -(-low // step) * step
        if (high - first) // step + 1 <= 13:
            return list(range(first, high + 1, step))
    return [low, high]


def _n(value):
    return f'{value:.2f}'


def line_chart(*, key, title, series, x_name, value_text, y_max=None, summary=None):
    """One figure: a titled chart, its legend, and its text description.

    ``x_name`` names the horizontal axis in the singular ("Layer").
    ``value_text`` formats a stored value exactly as the page's tables do.
    ``y_max`` is the natural top of the scale (1 for a share). Without it,
    or when a stored value is larger, the scale ends at the first round tick
    at or above the largest stored value. The scale starts at zero: it is for
    measures that are never negative.
    """
    drawn = [(x, y) for line in series for x, y in line.points
             if isinstance(y, (int, float)) and not isinstance(y, bool) and math.isfinite(y)]
    if not drawn:
        return el('figure', el('figcaption', title), el('p', 'No value is stored for this chart, so nothing is drawn.'),
                  class_='chart', id=key)
    xs = sorted({x for line in series for x, _ in line.points})
    low, high = xs[0], xs[-1]
    largest = max(y for _, y in drawn)
    ticks = ticks_to(largest if y_max is None else max(y_max, largest))
    top = float(ticks[-1])

    def px(x):
        return (LEFT + RIGHT) / 2 if high == low else LEFT + (x - low) / (high - low) * (RIGHT - LEFT)

    def py(y):
        return BOTTOM - min(max(y, 0.0), top) / top * (BOTTOM - TOP)

    parts = []
    for tick in ticks:
        y = _n(py(float(tick)))
        parts.append(el('line', x1=LEFT, y1=y, x2=RIGHT, y2=y, class_='axis' if not tick else 'grid'))
        parts.append(el('text', _plain(tick), x=LEFT - 8, y=_n(py(float(tick)) + 4), text_anchor='end'))
    for tick in _x_ticks(low, high):
        parts.append(el('line', x1=_n(px(tick)), y1=BOTTOM, x2=_n(px(tick)), y2=BOTTOM + 5, class_='axis'))
        parts.append(el('text', tick, x=_n(px(tick)), y=BOTTOM + 19, text_anchor='middle'))
    parts.append(el('text', x_name, x=_n((LEFT + RIGHT) / 2), y=HEIGHT - 8, text_anchor='middle', class_='axis-name'))

    order = sorted(range(len(series)), key=lambda i: (series[i].style not in (BASELINE, REFERENCE),
                                                      series[i].style == REFERENCE, i))
    marks, targets, spoken = [], [], []
    for index in order:
        line = series[index]
        kept = [(x, y) if isinstance(y, (int, float)) and not isinstance(y, bool) and math.isfinite(y) else (x, None)
                for x, y in line.points]
        runs, run = [], []
        for x, y in kept:
            if y is None:
                if run: runs.append(run)
                run = []
            else:
                run.append((x, y))
        if run: runs.append(run)
        for run in runs:
            if len(run) > 1:
                path = ' '.join(('M' if i == 0 else 'L') + f'{_n(px(x))} {_n(py(y))}' for i, (x, y) in enumerate(run))
                marks.append(el('path', d=path, class_=f'ser ser-{line.css}'))
            if len(run) == 1 or len(xs) <= 16:
                marks.extend(el('circle', cx=_n(px(x)), cy=_n(py(y)), r=4, class_=f'dot dot-{line.css}')
                             for x, y in run)
            targets.extend(el('circle', el('title', f'{line.label}, {x_name.lower()} {x}: {value_text(y)}'),
                              cx=_n(px(x)), cy=_n(py(y)), r=7, class_='hit') for x, y in run)
        stored = [(x, y) for run in runs for x, y in run]
        spoken.append(f'{line.label}: no value stored.' if not stored else
                      f'{line.label}: {value_text(stored[0][1])} at {x_name.lower()} {stored[0][0]}' +
                      ('.' if len(stored) == 1 else
                       f', {value_text(stored[-1][1])} at {x_name.lower()} {stored[-1][0]}.'))
    description = ' '.join([f'A line for each of {len(series)} readings, by {x_name.lower()} '
                            f'from {low} to {high}.' if high != low else
                            f'A point for each of {len(series)} readings at {x_name.lower()} {low}.',
                            *spoken, *([summary] if summary else [])])
    picture = el('svg', el('title', title, id=key + '-t'), el('desc', description, id=key + '-d'),
                 *parts, *marks, *targets,
                 viewBox=f'0 0 {WIDTH} {HEIGHT}', role='img', aria_labelledby=f'{key}-t {key}-d')
    legend = el('ul', [el('li', el('svg', el('line', x1=2, y1=5, x2=26, y2=5, class_=f'ser ser-{line.css}'),
                                   viewBox='0 0 28 10', width=28, height=10, aria_hidden='true', focusable='false'),
                          line.label) for line in series], class_='legend')
    return el('figure', el('figcaption', title), picture, legend, class_='chart', id=key)


CHART_STYLE = """
:root{--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#eda100;--s4:#e87ba4;--s5:#008300;--s6:#4a3aa7;--s7:#e34948}
@media (prefers-color-scheme:dark){:root{--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181;
--s5:#008300;--s6:#9085e9;--s7:#e66767}}
figure.chart{margin:14px 0 6px}
figure.chart figcaption{font-weight:600;margin-bottom:6px}
figure.chart>svg{display:block;width:100%;max-width:760px;height:auto;background:var(--surface)}
figure.chart text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
figure.chart text.axis-name{fill:var(--ink-2);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ser{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.ser-base{stroke:var(--muted);stroke-dasharray:1 5}
.ser-ref{stroke:var(--ink)}
.ser-c0{stroke:var(--s0)}.ser-c1{stroke:var(--s1)}.ser-c2{stroke:var(--s2)}.ser-c3{stroke:var(--s3)}
.ser-c4{stroke:var(--s4)}.ser-c5{stroke:var(--s5)}.ser-c6{stroke:var(--s6)}.ser-c7{stroke:var(--s7)}
.dot{stroke:var(--surface);stroke-width:2}
.dot-base{fill:var(--muted)}.dot-ref{fill:var(--ink)}
.dot-c0{fill:var(--s0)}.dot-c1{fill:var(--s1)}.dot-c2{fill:var(--s2)}.dot-c3{fill:var(--s3)}
.dot-c4{fill:var(--s4)}.dot-c5{fill:var(--s5)}.dot-c6{fill:var(--s6)}.dot-c7{fill:var(--s7)}
.hit{fill:transparent;stroke:none}
ul.legend{list-style:none;display:flex;flex-wrap:wrap;gap:4px 22px;padding:0;margin:8px 0 0;font-size:13px;
color:var(--ink-2)}
ul.legend li{margin:0}
ul.legend svg{vertical-align:middle;margin-right:6px}
@media print,(forced-colors:active){
:root{--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#eda100;--s4:#e87ba4;--s5:#008300;--s6:#4a3aa7;--s7:#e34948}
.ser-c0{stroke-dasharray:10 5}.ser-c1{stroke-dasharray:4 5}.ser-c2{stroke-dasharray:10 5 2 5}
.ser-c3{stroke-dasharray:14 4}.ser-c4{stroke-dasharray:6 4 6 8}.ser-c5{stroke-dasharray:2 6}
.ser-c6{stroke-dasharray:14 4 2 4 2 4}.ser-c7{stroke-dasharray:6 3 1 3}}
"""


# --- forest charts: stored estimates with their stored intervals ---------------


@dataclass(frozen=True)
class Estimate:
    """One row of a forest chart: a stored estimate and, where it is drawn, its stored interval."""
    label: str
    value: object = None
    lower: object = None
    upper: object = None
    #: Said beside the label when the stored interval is not drawn, and why ("too few pairs", say).
    note: object = None

    @property
    def drawn_interval(self):
        return self.note is None and _finite(self.lower) and _finite(self.upper)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ticks_between(low, high):
    """Round ticks from at or below ``low`` to at or above ``high``.

    At most six steps of 1, 2, 2.5, or 5 times a power of ten, in Decimal
    arithmetic, so the same input gives the same ticks on every machine. A
    scale with no width is widened to one step either side of its value.
    """
    if not (_finite(low) and _finite(high)) or low > high:
        return [Decimal(-1), Decimal(0), Decimal(1)]
    lo, hi = Decimal(repr(float(low))), Decimal(repr(float(high)))
    if lo == hi:
        width = abs(lo) or Decimal(1)
        lo, hi = lo - width, hi + width
    span = hi - lo
    for power in (span.adjusted() - 1, span.adjusted(), span.adjusted() + 1):
        for mantissa in ('1', '2', '2.5', '5'):
            step = Decimal(mantissa).scaleb(power)
            first = int((lo / step).to_integral_value(rounding=ROUND_FLOOR))
            last = int((hi / step).to_integral_value(rounding=ROUND_CEILING))
            if last - first <= 6:
                return [step * index for index in range(first, last + 1)]
    raise AssertionError('unreachable: a step of ten times the span always fits')


FOREST_WIDTH, FOREST_ROW = 720, 40
FOREST_LEFT, FOREST_RIGHT, FOREST_TOP = 24, 696, 8


def forest_chart(*, key, title, rows, value_text, axis_name, estimate_text=None, summary=None):
    """One figure: each row's stored estimate as a dot and its stored interval as a line.

    Rows are drawn in the order given, each with its label on its own line
    above the mark, so a long label never runs into the scale. The scale
    always includes zero, which is marked with a dashed line. A row whose
    ``note`` is set draws no interval and says why beside its label; a row
    with no stored estimate says so. ``value_text`` formats a stored value as
    the page's tables do, and ``estimate_text`` (by default the same) formats
    the estimate itself.
    """
    estimate_text = estimate_text or value_text
    drawn = [value for row in rows for value in
             ([row.value] + ([row.lower, row.upper] if row.drawn_interval else [])) if _finite(value)]
    if not drawn:
        return el('figure', el('figcaption', title), el('p', 'No value is stored for this chart, so nothing is drawn.'),
                  class_='chart forest', id=key)
    ticks = ticks_between(min(drawn + [0.0]), max(drawn + [0.0]))
    low, high = float(ticks[0]), float(ticks[-1])
    bottom = FOREST_TOP + len(rows) * FOREST_ROW
    height = bottom + 46

    def px(x):
        return FOREST_LEFT + (x - low) / (high - low) * (FOREST_RIGHT - FOREST_LEFT)

    parts = []
    for tick in ticks:
        x = _n(px(float(tick)))
        parts.append(el('line', x1=x, y1=FOREST_TOP, x2=x, y2=bottom, class_='zero' if not tick else 'grid'))
        parts.append(el('line', x1=x, y1=bottom, x2=x, y2=bottom + 5, class_='axis'))
        parts.append(el('text', _plain(tick), x=x, y=bottom + 19, text_anchor='middle'))
    parts.append(el('line', x1=FOREST_LEFT, y1=bottom, x2=FOREST_RIGHT, y2=bottom, class_='axis'))
    parts.append(el('text', axis_name, x=_n((FOREST_LEFT + FOREST_RIGHT) / 2), y=height - 6, text_anchor='middle',
                    class_='axis-name'))
    marks, targets, spoken = [], [], []
    for index, row in enumerate(rows):
        top = FOREST_TOP + index * FOREST_ROW
        middle = _n(top + 28)
        if not _finite(row.value):
            aside, said = ' (no estimate stored)', f'{row.label}: no estimate stored.'
        elif row.note is not None:
            aside = f' ({row.note})'
            said = f'{row.label}: {estimate_text(row.value)}; {row.note}.'
        elif row.drawn_interval:
            aside = None
            said = f'{row.label}: {estimate_text(row.value)}, interval {value_text(row.lower)} to {value_text(row.upper)}.'
        else:
            aside = ' (no interval stored)'
            said = f'{row.label}: {estimate_text(row.value)}; no interval stored.'
        spoken.append(said)
        marks.append(el('text', el('tspan', row.label), el('tspan', aside, class_='aside') if aside else None,
                        x=FOREST_LEFT, y=_n(top + 14), class_='row-label'))
        if row.drawn_interval:
            left, right = _n(px(row.lower)), _n(px(row.upper))
            marks.append(el('line', x1=left, y1=middle, x2=right, y2=middle, class_='ci'))
            for end in (left, right):
                marks.append(el('line', x1=end, y1=_n(top + 23), x2=end, y2=_n(top + 33), class_='ci'))
        if _finite(row.value):
            marks.append(el('circle', cx=_n(px(row.value)), cy=middle, r=4.5, class_='est'))
            targets.append(el('circle', el('title', said), cx=_n(px(row.value)), cy=middle, r=9, class_='hit'))
    description = ' '.join([f'One row for each of {len(rows)} readings: the stored estimate as a dot and its stored '
                            f'interval as a line, on a scale from {_plain(ticks[0])} to {_plain(ticks[-1])}. '
                            'The dashed line marks zero.', *spoken, *([summary] if summary else [])])
    picture = el('svg', el('title', title, id=key + '-t'), el('desc', description, id=key + '-d'),
                 *parts, *marks, *targets,
                 viewBox=f'0 0 {FOREST_WIDTH} {height}', role='img', aria_labelledby=f'{key}-t {key}-d')
    return el('figure', el('figcaption', title), picture,
              el('p', 'Dot: the estimate. Line: its interval. Dashed line: zero.', class_='chart-key'),
              class_='chart forest', id=key)


FOREST_STYLE = """
figure.forest text.row-label{fill:var(--ink);font-size:12.5px;paint-order:stroke;stroke:var(--surface);
stroke-width:4px;stroke-linejoin:round}
figure.forest text.row-label tspan.aside{fill:var(--ink-2)}
.zero{stroke:var(--ink-2);stroke-width:1.25;stroke-dasharray:4 4}
.ci{stroke:var(--s0);stroke-width:2;stroke-linecap:round}
.est{fill:var(--s0);stroke:var(--surface);stroke-width:2}
p.chart-key{color:var(--ink-2);font-size:13px;margin:6px 0 0}
"""
