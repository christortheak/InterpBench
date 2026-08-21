# reasoning-style taxonomies — how a response ARGUES, as pinned data

A reasoning-style taxonomy is a versioned JSON file of surface features that
measure the CHARACTER of a generated argument (hedging, certainty,
rule-application language, enumeration, …) — per generation, deterministically,
with no model or judge in the loop. It closes the gap between vocabulary
(marker density) and decisions (choice/months parsers): the headline contrast
"the decision moved without the style moving" needs a style instrument.

```json
{
  "schemaVersion": 1,
  "name": "my-style-taxonomy-v1",
  "features": [
    {
      "id": "hedging",
      "title": "Hedging language",
      "kind": "wordList",
      "patterns": ["perhaps", "might", "on the other hand"],
      "normalize": "per1kWords"
    }
  ]
}
```

- `schemaVersion` — must be `1`.
- `id` — unique per taxonomy; becomes the `rs_<id>` column in metrics.csv and
  the `rs_<id>` effect-size endpoint. Restricted to `[A-Za-z0-9_.-]`.
- `kind` — `wordList` or `regex`. **`wordList` is the conservative choice**:
  its whole-word semantics are simple, trivially identical across engines,
  and immune to regex-dialect drift — prefer it for evidence-grade
  taxonomies unless a feature genuinely needs structure (`regex` exists for
  that, under the restricted grammar below).
- `patterns` — non-empty list. Matching is case-insensitive on both kinds.
  Text and patterns are **NFC-normalized before any matching** (Swift
  `precomposedStringWithCanonicalMapping`, Python
  `unicodedata.normalize("NFC", …)`), so decomposed and precomposed accents
  (`é` vs `e`+U+0301) score identically on both engines.
  - `wordList`: each pattern matches as WHOLE WORDS (a multi-word pattern
    matches a contiguous word sequence). Words are maximal runs of Unicode
    scalars whose general category is a letter (Lu/Ll/Lt/Lm/Lo) or a decimal
    digit (Nd), taken from the NFC text after mapping each scalar through
    its unconditional full lowercase mapping. (Consequences: superscript ²
    and Roman-numeral characters are separators, not word characters;
    `ΛΟΓΟΣ` tokenizes to `λογοσ` with a medial σ, never a final ς; `İ`
    lowercases to `i` + combining dot, which splits the token.)
  - `regex`: counted as case-insensitive, non-overlapping, leftmost matches.
    Patterns must PARSE inside a restricted portable grammar — "it compiles
    here" is not enough, because ICU (Swift) and Python `re` each accept
    syntax the other rejects or silently reads differently. Anything outside
    the grammar is rejected at load with the construct name and position.

    Accepted grammar (the full contract; identical parser on both engines):

    ```text
    pattern     ::= alternation                      (then end of pattern)
    alternation ::= sequence ("|" sequence)*         (empty branches allowed)
    sequence    ::= term*
    term        ::= atom quantifier?
    atom        ::= literal | "." | "^" | "$" | escape | class
                  | "(?:" alternation ")"            (non-capturing only)
    literal     ::= any character EXCEPT  \ ^ $ . | ? * + ( ) [ { }
    escape      ::= "\" ( d D w W s S    class escapes
                        | b              word boundary (not inside classes,
                                         not quantifiable)
                        | n t r          control literals
                        | ASCII punctuation, e.g. \. \? \{ \} \[ \] \\ )
    class       ::= "[" "^"? member+ "]"
    member      ::= literal | escape | range a-b     ("-" is literal at the
                                                     start, before "]", or
                                                     escaped; ranges need
                                                     left <= right)
    quantifier  ::= ("*" | "+" | "?" | "{m}" | "{m,}" | "{m,n}") "?"?
                                                     (m <= n <= 9999; lazy
                                                     "?" allowed)
    ```

    Rejected by parse (a load error, never a silent divergence): capturing
    `(…)`, named groups (`(?P<…>` and `(?<…>`), backreferences and octal
    escapes, **all four lookarounds** (`(?= (?! (?<= (?<!`), inline flags
    `(?i)`, atomic/conditional/comment groups, `\p{…}` properties,
    `\x`/`\u`/`\U`/`\N` escapes, `\R`, `\A`/`\Z`/`\B` and every other letter
    escape, possessive quantifiers, nested `[`, `&&` and `\b` inside
    classes, and bare `{` `}` that are not well-formed quantifiers (escape
    them as `\{` `\}`).

    Pinned matching semantics (encoded in the cross-engine parity fixture):
    `.` matches any character except `\n` — including `\r` (Swift passes
    `.useUnixLineSeparators` to agree with Python's default); `^` and `$`
    anchor to the whole text, never per-line, and `$` also matches just
    before one trailing `\n`; empty-matchable patterns (e.g. `(?:ab)*`)
    advance one character after each empty match on both engines.
- `normalize` — how the raw match count becomes the per-generation value:
  - `perSentence`: count / sentences. A sentence boundary is `.`, `!` or `?`
    followed by whitespace or end of text; minimum 1 sentence.
  - `per1kWords`: count × 1000 / words, where words are whitespace-separated
    tokens; minimum 1 word.
  - `rawCount`: the count itself.

## Where the file lives in a workspace

`prompts/taxonomies/<name>.json`. Pin it into a draft study with

```bash
steerlab-cli experiment set-style-taxonomy <experiment> prompts/taxonomies/<name>.json
```

which stamps `reasoningStyleTaxonomyPath` + `reasoningStyleTaxonomyHash`
(SHA-256 of the file bytes) into the manifest. After that the taxonomy is
under the same drift firewall as every other pinned input: editing the file
after pinning is a verify violation. No pin = no reasoning-style scoring
(and no violation).

## What you get

- `metrics.csv` gains one `rs_<id>` column per feature (both engines).
- `report.json` gains a per-condition `reasoningStyle` block
  (`{"taxonomy", "taxonomyHash", "features": {"<id>": {"mean", "n"}}}`).
- Every `rs_<id>` joins the paired effect-size machinery (bootstrap CI +
  Wilcoxon against the same-item baseline) in run reports and `analyze`.
- `experiment rescore-style <name>` recomputes the columns post-hoc from a
  completed run's generations into NEW files (`reasoning-style.csv` +
  `reasoning-style.json`) in a fresh run directory — original runs are never
  mutated.

## The two example files are STARTING POINTS, not instruments

`reasoning-style-generic-template.json` (word lists) and
`reasoning-style-structure-template.json` (regex/structural) are deliberately
domain-neutral demonstrations of the schema. A real study must adapt the
features to its own domain and hypotheses, version the file, and hash-pin it
BEFORE behavioral runs (the circularity firewall applies to measurement
instruments too). Like marker density, these are surface features: report
them as style endpoints alongside — never instead of — outcome endpoints.
