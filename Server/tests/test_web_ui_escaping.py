"""The bundled web workbench must never build markup out of unescaped data.

``api/static/index.html`` keeps a bearer token in ``localStorage`` and sends it
on every request, so ONE ``innerHTML`` sink fed by a server string is a
token-theft primitive: API ``detail`` text quotes caller-supplied names and
paths straight back, and a study, concept, run, or model id is caller-supplied
too. ``toast()`` was exactly that sink.

There is no JS test runner in this package, so the ratchet is a scanner: it
walks the page's template literals, keeps the interpolations that land in
MARKUP, and requires each one to render something that cannot carry tags —
an ``esc()`` call, a computed number, or a literal chosen by a ternary. A new
unescaped interpolation fails here until its author escapes it (or records it
below with a reason), which is the same shape as the mutating-route ratchet in
``test_wp_s_hardening.py``.
"""

import os
import re

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "steerlab_server", "api", "static", "index.html")


def _source() -> str:
    with open(INDEX, encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------
# A very small JS scanner: enough to know which `${…}` sit inside a template
# literal that carries markup. It tracks template/expression nesting, quoted
# strings, comments, and regex literals — no more, because no more is needed
# to answer that one question.
# --------------------------------------------------------------------------

_REGEX_PRECEDERS = "(,=:[!&|?{;+~^%<>*-"


def _matching_brace(src: str, start: int) -> int:
    depth, i = 1, start
    while i < len(src) and depth:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return i - 1


def markup_interpolations(src: str) -> list[tuple[int, str]]:
    """``(line, expression)`` for every ``${…}`` inside a markup template."""
    literals: list[dict] = []      # open template-literal frames
    finished: list[dict] = []
    stack: list[str] = []          # "tmpl" | "expr"
    prev = ""
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        top = stack[-1] if stack else None
        if top == "tmpl":
            frame = literals[-1]
            if c == "\\":
                i += 2
                continue
            if c == "`":
                stack.pop()
                finished.append(literals.pop())
                i += 1
                continue
            if c == "$" and i + 1 < n and src[i + 1] == "{":
                end = _matching_brace(src, i + 2)
                frame["holes"].append((src.count("\n", 0, i) + 1,
                                       src[i + 2:end]))
                stack.append("expr")
                i += 2
                continue
            frame["static"] += c
            i += 1
            continue
        if c == "`":
            stack.append("tmpl")
            literals.append({"static": "", "holes": []})
            i += 1
            continue
        if c in "'\"":
            quote, i = c, i + 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
            i += 1
            prev = quote
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            nl = src.find("\n", i)
            i = n if nl < 0 else nl
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i = src.find("*/", i) + 2
            continue
        if c == "/" and prev in _REGEX_PRECEDERS:
            i += 1
            while i < n and src[i] != "/":
                i += 2 if src[i] == "\\" else 1
            i += 1
            prev = "/"
            continue
        if c == "{" and top == "expr":
            stack.append("expr")
            i += 1
            continue
        if c == "}" and top == "expr":
            stack.pop()
            i += 1
            continue
        if not c.isspace():
            prev = c
        i += 1

    markup = re.compile(r"<[a-zA-Z/!]")
    out: list[tuple[int, str]] = []
    for frame in finished:
        # A template is a markup context when it carries tags itself, OR when
        # one of its holes splices tags in — `${cond ? `<span>…` : "…"}` makes
        # the WHOLE literal markup even though its own static text is prose.
        if markup.search(frame["static"]) or any(
                markup.search(expr) for _, expr in frame["holes"]):
            out.extend(frame["holes"])
    return out


# --------------------------------------------------------------------------
# What may be rendered into markup without escaping.
# --------------------------------------------------------------------------

def _scan_top_level(expr: str):
    """Yield ``(index, char)`` for characters at nesting depth 0, outside any
    quoted string or template literal."""
    depth = 0
    q = ""
    i = 0
    while i < len(expr):
        c = expr[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = ""
            i += 1
            continue
        if c in "'\"`":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0:
            yield i, c
        i += 1


def _ternary_branches(expr: str) -> tuple[str, str] | None:
    """The two BRANCHES of a top-level ternary; the condition is dropped
    because it is tested, never rendered."""
    qmark = None
    for i, c in _scan_top_level(expr):
        if c == "?" and expr[i:i + 2] != "??" and qmark is None:
            qmark = i
        elif c == ":" and qmark is not None:
            return expr[qmark + 1:i], expr[i + 1:]
    return None


def _concat_parts(expr: str) -> list[str] | None:
    """Operands of a top-level ``+`` concatenation."""
    cuts = [i for i, c in _scan_top_level(expr) if c == "+"]
    if not cuts:
        return None
    parts, last = [], 0
    for i in cuts:
        parts.append(expr[last:i])
        last = i + 1
    parts.append(expr[last:])
    return [p for p in parts if p.strip()]


def _is_literal(expr: str) -> bool:
    """A quoted string or a template literal — its own holes, if any, were
    collected as separate frames and are checked on their own."""
    expr = expr.strip()
    return len(expr) >= 2 and expr[0] == expr[-1] and expr[0] in "'\"`"


#: Expression forms that cannot carry markup no matter what the server says:
#: arithmetic, a fixed-decimal rendering of a number, or an array length.
_NUMERIC = re.compile(r"(^Math\.)|(^Number\()|(\.toFixed\(\d*\)$)|(\.length$)")

#: Helpers declared in this file that RETURN markup and escape their own
#: inputs (they render the evidence-tier / reportability pills).
_MARKUP_HELPERS = ("tierPill(", "claimPill(")


def renders_safely(expr: str) -> bool:
    expr = " ".join(expr.split()).strip()
    while expr.startswith("(") and _matching_paren(expr, 1) == len(expr) - 1:
        expr = expr[1:-1].strip()
    if not expr:
        return True
    if expr.startswith("esc(") and _matching_paren(expr, 4) == len(expr) - 1:
        return True
    if expr.startswith("encodeURIComponent("):
        return True
    if expr.startswith(_MARKUP_HELPERS):
        return True
    if _is_literal(expr):
        return True
    if _NUMERIC.search(expr):
        return True
    # `xs.map(x => `…`).join("")` — the callback's own template literal is a
    # separate frame and is checked there.
    if re.search(r"\.map\(.*\)\.join\(", expr, re.S):
        return True
    branches = _ternary_branches(expr)
    if branches is not None:
        return all(renders_safely(part) for part in branches)
    parts = _concat_parts(expr)
    if parts is not None:
        return all(renders_safely(part) for part in parts)
    return False


def _matching_paren(expr: str, start: int) -> int:
    depth, i = 1, start
    while i < len(expr) and depth:
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
        i += 1
    return i - 1


#: The deliberate exceptions, by exact expression text. Escaping any of these
#: would BREAK the page rather than protect it: each already holds finished
#: markup this file built (and escaped) a line or two above, so passing it
#: through ``esc`` again would render the tags as text. Loop counters are here
#: for the opposite reason — they are this page's own integers, and escaping
#: them is noise. Anything the server or the researcher supplied belongs in
#: ``esc()``, not in this set.
_TRUSTED_LITERAL_SOURCES = {
    # Markup fragments assembled just above the sink:
    "opts",     # the <option> list for one steering slot
    "spark",    # the per-layer norm sparkline (<span> bars)
    "ctrl",     # the control-cosine tags
    "author",   # the draft-authoring controls block
    # Loop counters over this page's own arrays (SLOTS, PROBE.rows, …), used
    # as data-i="" handles for the change listeners below each render.
    "i", "j",
    # A CSS colour this function computed from two numbers on the line above.
    "b",
    # A bar height in px, from Math.round on the line above.
    "h",
}


def test_toast_builds_a_text_node_instead_of_markup():
    src = _source()
    start = src.index("function toast(msg, bad)")
    body = src[start:src.index("\n\n", start)]
    assert "innerHTML" not in body, (
        "toast() renders API error text, which quotes caller-supplied names "
        "and paths — it must not go through innerHTML")
    assert "insertAdjacentHTML" not in body
    assert "textContent" in body


def test_no_markup_template_interpolates_an_unescaped_value():
    offenders = []
    for line, expr in markup_interpolations(_source()):
        flat = " ".join(expr.split())
        if renders_safely(flat) or flat in _TRUSTED_LITERAL_SOURCES:
            continue
        offenders.append(f"index.html:{line}: ${{{flat[:110]}}}")
    assert not offenders, (
        "These values are interpolated into MARKUP without escaping. Wrap "
        "each in esc(), or build the node and set textContent; if the value "
        "is a literal this page itself authored, add it to "
        "_TRUSTED_LITERAL_SOURCES with a reason:\n  "
        + "\n  ".join(sorted(offenders)))


def test_the_scanner_actually_finds_the_pages_markup_holes():
    # Guards the guard: a scanner that silently matched nothing would make
    # the ratchet above vacuously green.
    holes = markup_interpolations(_source())
    assert len(holes) > 150, len(holes)
    assert any("esc(" in expr for _, expr in holes)


def test_the_scanner_flags_a_planted_unescaped_sink():
    planted = '''
      function render(x){ el.innerHTML = `<td class="mono">${x.name}</td>`; }
    '''
    holes = markup_interpolations(planted)
    assert holes and not renders_safely(holes[0][1])
    safe = 'el.innerHTML = `<td>${esc(x.name)}</td>`;'
    assert renders_safely(markup_interpolations(safe)[0][1])
