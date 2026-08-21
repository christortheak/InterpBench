"""Reasoning-style measurement as pinned DATA (Swift twin:
``Sources/ExperimentKit/ReasoningStyle.swift``).

A taxonomy is a versioned JSON file of surface features that measure how a
generated response ARGUES (hedging, certainty, enumeration, …) — distinct from
the paired judge's holistic preference and from concept marker density.
Everything here is concept- and domain-agnostic: features arrive as data
(``prompts/taxonomies/<name>.json``), pinned by hash into the manifest
(``reasoningStyleTaxonomyPath`` + ``reasoningStyleTaxonomyHash``), and scored
deterministically from the generated text — no model access, so results are
recomputable post-hoc (``experiment rescore-style``).

Cross-engine scoring contract (fixture-tested to 1e-9 on both engines,
``tests/fixtures/reasoning-style/reasoning-style-parity.json``):

- generated text AND every pattern are NFC-normalized before any matching
  (Swift ``precomposedStringWithCanonicalMapping`` /
  ``unicodedata.normalize("NFC", …)``) — decomposed and precomposed accents
  score identically;
- words for matching = maximal runs of Unicode scalars whose general
  category is a letter (Lu/Ll/Lt/Lm/Lo) or a decimal digit (Nd), taken from
  the NFC text after mapping EACH scalar through its unconditional full
  lowercase mapping (per-scalar, no Final_Sigma context — CPython's
  ``str.lower`` applies Final_Sigma, Swift's ``lowercased()`` does not, so
  neither whole-string API is the rule);
- ``wordList`` patterns match as whole-word (contiguous multi-word) sequences;
- ``regex`` patterns are case-insensitive non-overlapping leftmost match
  counts, restricted to a PARSED portable grammar (see
  ``portable_regex_violation`` — anything outside it is a load error).
  Pinned semantics: ``.`` matches any character except ``\n`` (including
  ``\r`` — Swift passes ``.useUnixLineSeparators`` to match Python's
  default); ``^``/``$`` anchor to the whole text (``$`` also matches just
  before one trailing ``\n`` on both engines); empty-match advancement
  (e.g. ``(?:ab)*``) is fixture-pinned;
- ``perSentence`` divides by sentence count (a ``.``/``!``/``?`` followed by
  whitespace or end of text; minimum 1);
- ``per1kWords`` scales by 1000 / whitespace-token count (minimum 1);
- ``rawCount`` is the count itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field

KINDS = ("wordList", "regex")
NORMALIZE_MODES = ("perSentence", "per1kWords", "rawCount")
_FEATURE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class TaxonomyError(Exception):
    """A taxonomy that cannot score identically on both engines must never
    be pinned — every schema/pattern problem is a LOAD error."""


# --- portable regex grammar (the cross-engine contract) --------------------
#
# "Compiles on both engines" is NOT a portability proof: ICU accepts \R and
# (?<name>…) which Python rejects, Python accepts (?P<name>…) and a literal
# bare "{" which ICU rejects, and class syntax silently DIVERGES (ICU treats
# "[[a]]" as a nested set and "[a&&b]" as an intersection; Python treats both
# as literals). So the validator PARSES a restricted grammar and rejects
# everything outside it by construct name + scalar position. Accepted grammar
# (identical recursive-descent implementation in Swift `PortableRegexParser`;
# shared acceptance/rejection vectors in
# ``tests/fixtures/reasoning-style/portable-regex-vectors.json``)::
#
#     pattern     ::= alternation                      (then end of pattern)
#     alternation ::= sequence ("|" sequence)*         (empty branches legal)
#     sequence    ::= term*
#     term        ::= atom quantifier?
#     atom        ::= literal | "." | "^" | "$" | escape | class
#                   | "(?:" alternation ")"            (non-capturing only)
#     literal     ::= any scalar EXCEPT  \ ^ $ . | ? * + ( ) [ { }
#     escape      ::= "\" ( d D w W s S               class escapes
#                         | b                          word boundary (not in
#                                                      classes, unquantifiable)
#                         | n t r                      control literals
#                         | ASCII punctuation )        escaped literal
#     class       ::= "[" "^"? member+ "]"
#     member      ::= class-literal | escape-in-class | range
#     range       ::= single-char "-" single-char      (left <= right; "-" is
#                                                      literal at start, before
#                                                      "]", or escaped)
#     quantifier  ::= ("*" | "+" | "?" | "{m}" | "{m,}" | "{m,n}") "?"?
#                                                      (m <= n <= 9999; lazy
#                                                      "?" ok, possessive "+"
#                                                      rejected)
#
# Rejected by parse (not by substring): capturing "(", named groups (both
# dialects), backreferences/octal, all four lookarounds, inline flags,
# atomic/conditional/comment groups, \p{…}, \x/\u/\U/\N, \R and every other
# letter escape, nested "[" / "&&" / "\b" inside classes, class escapes in
# ranges, reversed ranges, bare "{" "}" that are not well-formed quantifiers.

_CLASS_SET_ESCAPES = "dDwWsS"
_CONTROL_ESCAPES = "ntr"
_ASCII_DIGITS = "0123456789"
_ASCII_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_MAX_GROUP_DEPTH = 64
_MAX_QUANTIFIER_BOUND = 9999


class _RegexViolation(Exception):
    def __init__(self, construct: str, position: int):
        super().__init__(f"{construct} at position {position}")


class _PortableRegexParser:
    """Recursive-descent validator for the portable grammar above. Positions
    are 0-based Unicode scalar offsets (== Python string indices)."""

    def __init__(self, pattern: str):
        self.chars = pattern
        self.i = 0
        self.depth = 0

    def peek(self, offset: int = 0):
        j = self.i + offset
        return self.chars[j] if j < len(self.chars) else None

    def parse(self) -> None:
        self.parse_alternation()
        if self.i < len(self.chars):  # only a stray ')' can stop the parse
            raise _RegexViolation("unmatched ')'", self.i)

    def parse_alternation(self) -> None:
        self.parse_sequence()
        while self.peek() == "|":
            self.i += 1
            self.parse_sequence()

    def parse_sequence(self) -> None:
        while True:
            c = self.peek()
            if c is None or c == "|" or c == ")":
                return
            self.parse_term()

    def parse_term(self) -> None:
        quantifiable = self.parse_atom()
        self.parse_optional_quantifier(quantifiable)

    def parse_atom(self) -> bool:
        """Consume one atom; True when a quantifier may follow it."""
        c = self.peek()
        pos = self.i
        if c == "(":
            self.parse_group()
            return True
        if c == "[":
            self.parse_class()
            return True
        if c in ("*", "+", "?"):
            raise _RegexViolation("quantifier with nothing to repeat", pos)
        if c == "{":
            if self.scan_brace_quantifier() is not None:
                raise _RegexViolation("quantifier with nothing to repeat", pos)
            raise _RegexViolation("literal '{' must be escaped as '\\{'", pos)
        if c == "}":
            raise _RegexViolation("literal '}' must be escaped as '\\}'", pos)
        if c in ("^", "$"):
            self.i += 1
            return False
        if c == "\\":
            kind, _ = self.parse_escape(in_class=False)
            return kind != "boundary"
        self.i += 1  # every other scalar is a literal (including ']' and '.')
        return True

    def parse_optional_quantifier(self, quantifiable: bool) -> None:
        c = self.peek()
        start = self.i
        if c in ("*", "+", "?"):
            self.i += 1
        elif c == "{":
            scanned = self.scan_brace_quantifier()
            if scanned is None:
                raise _RegexViolation(
                    "literal '{' must be escaped as '\\{'", start)
            end, low, high = scanned
            if low > _MAX_QUANTIFIER_BOUND or (
                    high is not None and high > _MAX_QUANTIFIER_BOUND):
                raise _RegexViolation("quantifier bound too large", start)
            if high is not None and low > high:
                raise _RegexViolation("reversed '{m,n}' quantifier", start)
            self.i = end
        else:
            return
        if not quantifiable:
            raise _RegexViolation("quantifier on an anchor", start)
        if self.peek() == "?":  # lazy
            self.i += 1
        if self.peek() == "+":
            raise _RegexViolation("possessive quantifier", self.i)

    def scan_brace_quantifier(self):
        """Probe (no consume) a '{m}'/'{m,}'/'{m,n}' at self.i. Returns
        (index-after-'}', m, n-or-None) or None when it is not one — the
        two engines DISAGREE about a bare '{' (Python: literal, ICU: error),
        so a '{' must be a well-formed quantifier or an escaped literal."""
        j = self.i + 1
        low_digits = ""
        while j < len(self.chars) and self.chars[j] in _ASCII_DIGITS:
            low_digits += self.chars[j]
            j += 1
        if not low_digits:
            return None
        low = int(low_digits)
        if j < len(self.chars) and self.chars[j] == "}":
            return (j + 1, low, low)
        if j < len(self.chars) and self.chars[j] == ",":
            j += 1
            high_digits = ""
            while j < len(self.chars) and self.chars[j] in _ASCII_DIGITS:
                high_digits += self.chars[j]
                j += 1
            if j < len(self.chars) and self.chars[j] == "}":
                return (j + 1, low, int(high_digits) if high_digits else None)
        return None

    def parse_group(self) -> None:
        start = self.i
        self.i += 1  # '('
        if self.peek() != "?":
            raise _RegexViolation("capturing group '('", start)
        self.i += 1
        c = self.peek()
        if c == ":":
            self.i += 1
        elif c == "P":
            raise _RegexViolation("named group '(?P'", start)
        elif c == "<":
            nxt = self.peek(1)
            if nxt == "=":
                raise _RegexViolation("lookbehind '(?<='", start)
            if nxt == "!":
                raise _RegexViolation("lookbehind '(?<!'", start)
            raise _RegexViolation("named group '(?<'", start)
        elif c == "=":
            raise _RegexViolation("lookahead '(?='", start)
        elif c == "!":
            raise _RegexViolation("lookahead '(?!'", start)
        elif c == ">":
            raise _RegexViolation("atomic group '(?>'", start)
        elif c == "(":
            raise _RegexViolation("conditional group '(?('", start)
        elif c == "#":
            raise _RegexViolation("comment group '(?#'", start)
        else:
            raise _RegexViolation("inline flags '(?'", start)
        self.depth += 1
        if self.depth > _MAX_GROUP_DEPTH:
            raise _RegexViolation("group nesting too deep", start)
        self.parse_alternation()
        if self.peek() != ")":
            raise _RegexViolation("unterminated group", start)
        self.i += 1
        self.depth -= 1

    def parse_escape(self, in_class: bool):
        """Consume a '\\'-escape. Returns ("set", None) for \\d-style class
        escapes, ("boundary", None) for \\b, ("literal", char) otherwise."""
        start = self.i
        self.i += 1  # '\'
        c = self.peek()
        if c is None:
            raise _RegexViolation("trailing backslash", start)
        if c in _CLASS_SET_ESCAPES:
            self.i += 1
            return ("set", None)
        if c == "b":
            if in_class:
                raise _RegexViolation(
                    "'\\b' inside a character class", start)
            self.i += 1
            return ("boundary", None)
        if c in _CONTROL_ESCAPES:
            self.i += 1
            return ("literal", {"n": "\n", "t": "\t", "r": "\r"}[c])
        if c in "123456789":
            raise _RegexViolation(f"backreference escape '\\{c}'", start)
        if c == "0":
            raise _RegexViolation("octal escape '\\0'", start)
        if c in "pP":
            raise _RegexViolation(f"unicode property escape '\\{c}'", start)
        if c in "xuUN":
            raise _RegexViolation(f"hex/unicode escape '\\{c}'", start)
        if c in _ASCII_PUNCTUATION:
            self.i += 1
            return ("literal", c)
        raise _RegexViolation(f"unsupported escape '\\{c}'", start)

    def parse_class(self) -> None:
        start = self.i
        self.i += 1  # '['
        if self.peek() == "^":
            self.i += 1
        if self.peek() == "]":
            raise _RegexViolation("empty character class", start)
        last = None  # None | ("single", ch) | ("set",) | ("range",)
        while True:
            c = self.peek()
            if c is None:
                raise _RegexViolation("unterminated character class", start)
            if c == "]":
                self.i += 1
                return
            if c == "[":
                raise _RegexViolation(
                    "nested '[' in a character class", self.i)
            if c == "&" and self.peek(1) == "&":
                raise _RegexViolation(
                    "set operation '&&' in a character class", self.i)
            if c == "-" and last is not None and self.peek(1) != "]":
                dash = self.i
                if last[0] == "set":
                    raise _RegexViolation(
                        "class escape in a character range", dash)
                if last[0] == "range":
                    raise _RegexViolation(
                        "ambiguous '-' in a character class", dash)
                self.i += 1
                low = last[1]
                high = self.parse_range_endpoint(dash, start)
                if ord(low) > ord(high):
                    raise _RegexViolation("reversed character range", dash)
                last = ("range",)
                continue
            if c == "\\":
                kind, literal = self.parse_escape(in_class=True)
                last = ("set",) if kind == "set" else ("single", literal)
                continue
            self.i += 1
            last = ("single", c)

    def parse_range_endpoint(self, dash: int, class_start: int):
        c = self.peek()
        if c is None:
            raise _RegexViolation("unterminated character class", class_start)
        if c == "[":
            raise _RegexViolation("nested '[' in a character class", self.i)
        if c == "\\":
            kind, literal = self.parse_escape(in_class=True)
            if kind == "set":
                raise _RegexViolation(
                    "class escape in a character range", dash)
            return literal
        self.i += 1
        return c


def portable_regex_violation(pattern: str) -> str | None:
    """None when `pattern` is inside the portable cross-engine grammar;
    otherwise "<construct> at position <N>" (0-based scalar offsets).
    Identical implementation in Swift `PortableRegexParser.violation(in:)`;
    the two are held together by the shared vectors fixture."""
    try:
        _PortableRegexParser(pattern).parse()
        return None
    except _RegexViolation as exc:
        return str(exc)


@dataclass(frozen=True)
class Feature:
    id: str
    title: str
    kind: str                       # wordList | regex
    patterns: tuple[str, ...]
    normalize: str                  # perSentence | per1kWords | rawCount
    # Pre-processed match programs (built at load, after validation).
    token_patterns: tuple[tuple[str, ...], ...] = field(default=())
    compiled: tuple[re.Pattern, ...] = field(default=())


@dataclass(frozen=True)
class Taxonomy:
    name: str
    features: tuple[Feature, ...]

    @property
    def feature_ids(self) -> list[str]:
        """Feature ids in declared taxonomy order — the deterministic column
        / endpoint order (``rs_<id>``) on both engines."""
        return [f.id for f in self.features]

    # --- load + validation --------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Taxonomy":
        if not isinstance(d, dict):
            raise TaxonomyError("taxonomy must be a JSON object")
        if d.get("schemaVersion") != 1:
            raise TaxonomyError(
                f"taxonomy schemaVersion must be 1 (got {d.get('schemaVersion')!r})")
        name = str(d.get("name") or "").strip()
        if not name:
            raise TaxonomyError('taxonomy needs a non-empty "name"')
        raw_features = d.get("features")
        if not isinstance(raw_features, list) or not raw_features:
            raise TaxonomyError('taxonomy needs a non-empty "features" list')
        seen: set[str] = set()
        features: list[Feature] = []
        for index, raw in enumerate(raw_features):
            if not isinstance(raw, dict):
                raise TaxonomyError(f"feature #{index + 1} must be an object")
            fid = str(raw.get("id") or "").strip()
            if not fid:
                raise TaxonomyError(f"feature #{index + 1} has no id")
            if not _FEATURE_ID_RE.match(fid):
                raise TaxonomyError(
                    f"feature id {fid!r} must use only [A-Za-z0-9_.-] "
                    "(it becomes the rs_<id> metric column)")
            if fid in seen:
                raise TaxonomyError(f"duplicate feature id {fid!r}")
            seen.add(fid)
            kind = raw.get("kind")
            if kind not in KINDS:
                raise TaxonomyError(
                    f"feature {fid!r}: unknown kind {kind!r} (expected wordList|regex)")
            normalize = raw.get("normalize")
            if normalize not in NORMALIZE_MODES:
                raise TaxonomyError(
                    f"feature {fid!r}: unknown normalize {normalize!r} "
                    "(expected perSentence|per1kWords|rawCount)")
            # Patterns are NFC-normalized at load — matching happens in NFC
            # space on both engines (generated text is normalized in score()).
            patterns = tuple(
                unicodedata.normalize("NFC", p)
                for p in (raw.get("patterns") or [])
                if isinstance(p, str) and p.strip())
            if not patterns:
                raise TaxonomyError(
                    f'feature {fid!r} needs a non-empty "patterns" list')
            token_patterns: tuple[tuple[str, ...], ...] = ()
            compiled: tuple[re.Pattern, ...] = ()
            if kind == "wordList":
                token_lists = []
                for pattern in patterns:
                    tokens = tuple(match_tokens(pattern))
                    if not tokens:
                        raise TaxonomyError(
                            f"feature {fid!r}: word-list pattern {pattern!r} "
                            "contains no matchable words")
                    token_lists.append(tokens)
                token_patterns = tuple(token_lists)
            else:
                programs = []
                for pattern in patterns:
                    # Portability guard: the pattern must PARSE inside the
                    # restricted cross-engine grammar — "compiles here" is
                    # not a portability proof (ICU and re each accept
                    # constructs the other rejects or reads differently).
                    violation = portable_regex_violation(pattern)
                    if violation is not None:
                        raise TaxonomyError(
                            f"feature {fid!r}: regex {pattern!r}: "
                            f"{violation} — not in the portable "
                            "cross-engine regex subset (see "
                            "prompts/templates/reasoning-style/README.md)")
                    try:
                        programs.append(re.compile(pattern, re.IGNORECASE))
                    except re.error as exc:  # unreachable post-validation
                        raise TaxonomyError(
                            f"feature {fid!r}: regex {pattern!r} does not "
                            f"compile: {exc}") from exc
                compiled = tuple(programs)
            features.append(Feature(
                id=fid, title=str(raw.get("title") or fid), kind=kind,
                patterns=patterns, normalize=normalize,
                token_patterns=token_patterns, compiled=compiled))
        return cls(name=name, features=tuple(features))

    @classmethod
    def from_bytes(cls, data: bytes) -> "Taxonomy":
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TaxonomyError(f"taxonomy is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def load(cls, path: str) -> "Taxonomy":
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError as exc:
            raise TaxonomyError(f"cannot read taxonomy at {path}: {exc}") from exc
        return cls.from_bytes(data)

    # --- scoring (the cross-engine math) ------------------------------------

    def score(self, text: str) -> dict[str, float]:
        """Per-generation feature values, keyed by feature id. Deterministic,
        pure CPU; empty text scores 0 on every feature. The text is
        NFC-normalized up front (patterns were normalized at load), so
        decomposed and precomposed input scores identically on both engines."""
        text = unicodedata.normalize("NFC", text)
        tokens = match_tokens(text)
        sentences = float(sentence_count(text))
        words = float(whitespace_word_count(text))
        values: dict[str, float] = {}
        for feature in self.features:
            count = float(_match_count(feature, text, tokens))
            if feature.normalize == "perSentence":
                values[feature.id] = count / sentences
            elif feature.normalize == "per1kWords":
                values[feature.id] = count * 1000.0 / words
            else:  # rawCount
                values[feature.id] = count
        return values


def _is_token_scalar(character: str) -> bool:
    """Token scalars are letters (Lu/Ll/Lt/Lm/Lo) and DECIMAL digits (Nd) —
    a general-category rule, not the languages' native isalpha/isdigit
    (Python's ``isdigit`` accepts superscript ², Swift's ``isNumber``
    accepts Roman numerals; the category rule is identical on both)."""
    category = unicodedata.category(character)
    return category[0] == "L" or category == "Nd"


def match_tokens(text: str) -> list[str]:
    """The WORD tokenizer for ``wordList`` matching (identical rule in Swift
    ``matchTokens``): NFC-normalize, then map EACH scalar through its
    unconditional full lowercase mapping (per-scalar ``ch.lower()`` — NOT
    whole-string ``text.lower()``, whose Final_Sigma context rule ICU's
    ``lowercased()`` does not share), then take maximal runs of
    letter/decimal-digit scalars. Iterates code points (== Swift Unicode
    scalars), never grapheme clusters."""
    tokens: list[str] = []
    current: list[str] = []
    for original in unicodedata.normalize("NFC", text):
        for character in original.lower():
            if _is_token_scalar(character):
                current.append(character)
            elif current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def sentence_count(text: str) -> int:
    """Sentence count: ``.``/``!``/``?`` followed by whitespace or end of
    text; minimum 1 (so empty text scores 0, never divides by zero)."""
    count = 0
    last = len(text) - 1
    for index, character in enumerate(text):
        if character in ".!?" and (index == last or text[index + 1].isspace()):
            count += 1
    return max(count, 1)


def whitespace_word_count(text: str) -> int:
    """Whitespace-separated token count, minimum 1 — the ``per1kWords``
    denominator (deliberately the same "words" as the wordCount metric)."""
    return max(len(text.split()), 1)


def _match_count(feature: Feature, text: str, tokens: list[str]) -> int:
    if feature.kind == "wordList":
        total = 0
        for pattern_tokens in feature.token_patterns:
            width = len(pattern_tokens)
            if width == 0 or width > len(tokens):
                continue
            pattern_list = list(pattern_tokens)
            for start in range(len(tokens) - width + 1):
                if tokens[start:start + width] == pattern_list:
                    total += 1
        return total
    return sum(
        sum(1 for _ in program.finditer(text)) for program in feature.compiled)


# --- manifest pin plumbing ------------------------------------------------


@dataclass(frozen=True)
class PinnedStyle:
    """A hash-verified, ready-to-score pinned taxonomy."""
    taxonomy: Taxonomy
    path: str
    hash: str


def load_pinned(manifest, root: str | None = None) -> PinnedStyle | None:
    """Load the manifest's pinned reasoning-style taxonomy, HASH-CHECKED
    against the pinned ``reasoningStyleTaxonomyHash`` (drift raises — the
    scoring path must never read a file verify() would reject). Returns None
    when the manifest pins no taxonomy (absent = no reasoning-style scoring,
    no violation)."""
    from . import paths
    path = getattr(manifest, "reasoning_style_taxonomy_path", None)
    pinned = getattr(manifest, "reasoning_style_taxonomy_hash", None)
    if not path:
        if pinned:
            raise TaxonomyError(
                "reasoningStyleTaxonomyHash pinned without a path — an "
                "unresolvable pin certifies nothing")
        return None
    if not pinned:
        raise TaxonomyError(
            "reasoning-style taxonomy pin is incomplete — "
            "reasoningStyleTaxonomyPath and reasoningStyleTaxonomyHash must "
            "both be set")
    resolved = path if os.path.isabs(path) else os.path.join(
        paths.project_root() if root is None else root, path)
    try:
        with open(resolved, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise TaxonomyError(
            f"pinned reasoning-style taxonomy missing ({path}): {exc}") from exc
    live = hashlib.sha256(data).hexdigest()
    if live != pinned:
        raise TaxonomyError(
            f"reasoning-style taxonomy changed since pinning "
            f"(have {live[:12]}…, pinned {pinned[:12]}…)")
    return PinnedStyle(taxonomy=Taxonomy.from_bytes(data), path=path, hash=pinned)
