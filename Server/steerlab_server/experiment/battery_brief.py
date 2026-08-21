"""The capability-battery authoring brief — an LLM prompt assembled from the
battery contract and the linter's own rules.

Parallel to :func:`steerlab_server.experiment.authoring.generation_prompt`,
which emits the paste-elsewhere brief for a contrastive STIMULUS set. This is
the same helper for the other authored artifact a study pins: the capability
battery.

The brief is assembled from the real contracts rather than restated from
memory — the format-2 header/item schema comes from :mod:`.battery`, the rules
it must satisfy come from :mod:`.battery_lint`'s own thresholds and finding
codes. A draft written against this brief should therefore pass
``steerlab-server battery lint`` on the first try, and when a threshold moves
the brief moves with it.

Concept-agnostic by construction: nothing here knows what the study is about.
Item content is a study-design decision, so the brief states the MECHANICAL
properties a control must have and takes the domain to steer clear of as free
text from the caller (``avoid``), exactly as the stimulus brief takes a concept
name.
"""

from __future__ import annotations

import os
import textwrap

from . import battery as battery_mod
from . import battery_lint, paths

#: The brief is read by a person as often as by a model; keep it in a column
#: that survives a terminal, a chat box, and a paste into a text file.
_WIDTH = 78

#: The shipped worked example, workspace-relative.
TEMPLATE_PATH = "prompts/templates/battery/capability-battery-v2.template.jsonl"

#: How many items to ask for by default. The shipped batteries run 10–20 items
#: (``prompts/batteries/*.jsonl``: six at 10, one at 20), and
#: :data:`battery_lint.MIN_ITEMS` warns below 10 — 20 sits at the top of the
#: shipped range and leaves headroom to drop an item that lints badly.
DEFAULT_ITEM_COUNT = 20

# The template's bytes, inlined so the brief always carries a worked example
# even where the workspace has no templates tree (an imported workspace, a
# release tree without WorkspaceSeed/). Preferred source is still the file —
# see _worked_example. Kept byte-identical to
# WorkspaceSeed/prompts/templates/battery/capability-battery-v2.template.jsonl
# by test_battery_generation_prompt.py, which fails if the two drift.
_TEMPLATE_FALLBACK = r"""{"batteryFormat": 2, "scoring": "choiceProbability", "promptMode": "chatAssistant", "maxTokens": 24, "description": "TEMPLATE — capability battery, repaired format. Copy into your WORKSPACE prompts/batteries/, replace every item, lint it, then pin it. The items below are EXAMPLES of the format, not a study-ready battery: item authoring is a study-design decision (how hard, which capabilities, how many). Lint it with: steerlab-server battery lint prompts/batteries/<yours>.jsonl"}
{"id": "arith-sum", "prompt": "What is 17 + 26?", "answer": "43", "options": ["43", "33", "44", "36"]}
{"id": "arith-product", "prompt": "What is 9 times 8?", "answer": "72", "options": ["72", "63", "81", "78"]}
{"id": "seq-next", "prompt": "Continue the sequence: 2, 4, 6, 8, ...", "answer": "10", "options": ["10", "12", "9", "16"]}
{"id": "magnitude", "prompt": "Which number is larger, 0.7 or 0.07? Answer 0.7, 0.07, or equal.", "answer": "0.7", "options": ["0.7", "0.07", "equal"]}
{"id": "geo-capital", "prompt": "What is the capital of France?", "answer": "Paris", "options": ["Paris", "Lyon", "Marseille", "Bordeaux"]}
{"id": "sci-mixing", "prompt": "What color results from mixing blue and yellow paint?", "answer": "green", "options": ["green", "purple", "orange", "brown"]}
{"id": "cal-weekday", "prompt": "Which day of the week comes immediately after Tuesday?", "answer": "Wednesday", "options": ["Wednesday", "Monday", "Thursday", "Sunday"]}
{"id": "logic-subset", "prompt": "Is every rectangle a square? Answer yes, no, or sometimes.", "answer": "no", "options": ["no", "yes", "sometimes"]}
{"id": "logic-superset", "prompt": "Is every square a rectangle? Answer yes, no, or sometimes.", "answer": "yes", "options": ["yes", "no", "sometimes"]}
{"id": "sci-boiling", "prompt": "At sea level, water boils at how many degrees Celsius?", "answer": "100", "options": ["100", "212", "50", "0"]}
{"id": "str-reverse", "prompt": "The word 'cat' spelled backwards is which of these?", "answer": "tac", "options": ["tac", "cta", "act", "atc"]}
{"id": "sort-alpha", "prompt": "Which word comes first in alphabetical order?", "answer": "apple", "options": ["apple", "lemon", "orange", "pear"]}
{"id": "gen-free-example", "prompt": "Reply with exactly the token OKAY and nothing else.", "answer": "okay", "scoring": "generatedText", "grading": "token_exact", "note": "ESCAPE HATCH, shown for completeness. generatedText items generate and text-match, so they remain length- and format-sensitive; lint flags this one on purpose. Use it only for a capability that genuinely cannot be posed as a choice, and expect to defend it in the methods note."}
"""


def _checkout_seed_template() -> str:
    """``<checkout>/WorkspaceSeed/<TEMPLATE_PATH>``, derived from this module's
    location. The seed tree is where the template is maintained; a workspace
    only has a copy if it was seeded from it."""
    here = os.path.dirname(os.path.abspath(__file__))          # …/experiment
    checkout = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(checkout, "WorkspaceSeed", *TEMPLATE_PATH.split("/"))


def _worked_example(root: str | None = None) -> tuple[str, str]:
    """``(text, source)`` for the worked example.

    Prefers the WORKSPACE's own copy (so a researcher who edited their
    template sees their edit), then the checkout's seed tree, then the inlined
    bytes. ``source`` names where it came from, and is stated in the brief so a
    reader can go find the file.
    """
    for candidate in (paths.resolve(TEMPLATE_PATH, root),
                      _checkout_seed_template()):
        try:
            with open(candidate, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        if text.strip():
            return text if text.endswith("\n") else text + "\n", TEMPLATE_PATH
    return _TEMPLATE_FALLBACK, f"{TEMPLATE_PATH} (inlined copy — file not found here)"


def _numbered(number: int, text: str) -> str:
    """One numbered rule, wrapped to the brief's column with the hanging
    indent the hand-written rules use."""
    return textwrap.fill(text, width=_WIDTH,
                         initial_indent=f"{number}. ",
                         subsequent_indent="   ")


def _avoid_rule(avoid: str) -> str:
    avoid = (avoid or "").strip()
    if avoid:
        return _numbered(6,
            "STAY OUT OF THE STUDY'S DOMAIN. Do not write an item that "
            "touches, references, or requires knowledge of: "
            f"{avoid}. An item in the domain the study intervenes on is not a "
            "capability control — it is a second, unblinded outcome measure, "
            "and a score change there is exactly the thing the experiment is "
            "trying to attribute.")
    return _numbered(6,
        "STAY OUT OF THE STUDY'S DOMAIN. Do not write an item that touches "
        "the subject matter the experiment itself intervenes on or measures. "
        "An item in that domain is not a capability control — it is a second, "
        "unblinded outcome measure. (The domain to avoid was not supplied "
        "here; ask for it, or keep every item to plainly generic knowledge "
        "and reasoning.)")


def generation_prompt(count: int = DEFAULT_ITEM_COUNT, *, avoid: str = "",
                      root: str | None = None) -> str:
    """An LLM prompt the researcher can paste elsewhere to draft a format-2
    capability battery.

    ``count`` is how many items to ask for; ``avoid`` is free text naming the
    domain the battery must not touch (the study's own subject matter).
    """
    count = max(1, int(count))
    example, example_source = _worked_example(root)
    grading = textwrap.fill(
        ", ".join(f'"{mode}"' for mode in battery_mod.GRADING_MODES),
        width=_WIDTH, initial_indent=" " * 13, subsequent_indent=" " * 13)
    below_floor = ""
    if count < battery_lint.MIN_ITEMS:
        below_floor = "\n" + textwrap.fill(
            f"NOTE: {count} item(s) is below the linter's floor of "
            f"{battery_lint.MIN_ITEMS}, so the draft will earn a `fewItems` "
            "warning. That is intentional only if these items are being ADDED "
            "to an existing battery.", width=_WIDTH) + "\n"

    return f"""\
CAPABILITY BATTERY — AUTHORING BRIEF

You are drafting a capability battery for an activation-steering experiment.
Produce {count} items in the file format specified below. Read the whole brief
before writing anything.
{below_floor}
================================================================
1. WHAT A BATTERY IS FOR
================================================================

The experiment injects a direction into a language model's residual stream and
measures whether behaviour on the study's own items moves. A capability
battery answers a different question: did the intervention cost the model
general capability?

So the battery is a CONTROL. Its entire job is to HOLD STILL while the
intervention moves. A battery whose score drifts for any reason other than
lost capability — response length, output format, a persona in the surrounding
prompt, an ambiguous item, a lucky guess — is not a control and cannot be
cited as one. Every rule below follows from that single job.

================================================================
2. RULES FOR ITEMS
================================================================

1. DOMAIN-NEUTRAL. Plainly general knowledge, arithmetic, ordering, elementary
   logic, everyday science, language mechanics. Nothing specialist, nothing
   culturally narrow, nothing that a reasonable person would answer
   differently in a different country or decade.
2. UNAMBIGUOUS AND STABLE. Exactly one defensible answer, and it must be the
   same answer next year. No opinions, no "it depends", no facts with a
   moving value (populations, prices, office-holders, current events).
3. NOT AT THE FLOOR OR THE CEILING. An item every model gets right under every
   condition measures nothing, and neither does one no model ever gets right.
   Aim for items a competent model answers reliably but that a genuinely
   degraded model would start to miss.
4. DIVERSE IN SURFACE FORM. Vary the phrasing, the length, the question type,
   and the capability probed across the set. If every item is "What is X + Y?"
   the battery measures one narrow skill and calls it "capability".
5. INDEPENDENT. No item may give away another's answer, and no two items may
   restate the same question — duplicated prompts cannot fail independently,
   so the effective item count is lower than it looks.
{_avoid_rule(avoid)}
7. NO RESPONSE-FORMAT INSTRUCTIONS. Never write "answer with just one word",
   "reply with only the number", "and nothing else", or similar. Under the
   default scoring nothing is generated, so the instruction does no work — and
   it is the exact coupling that once let a study's own output-format demand
   fight a battery's, making the same battery read very differently under two
   instruments. Put descriptive text in the prompt; keep the options short.

================================================================
3. THE FILE FORMAT — batteryFormat {battery_mod.FORMAT_CURRENT}, JSONL
================================================================

One JSON object per line. The FIRST non-empty line is a header; every line
after it is an item. Write format {battery_mod.FORMAT_CURRENT} and nothing else — the headerless legacy
format is scored under the surrounding study's prompt settings, which is why
it can no longer be certified as a control.

HEADER (first line) keys:
  "batteryFormat"        REQUIRED, must be {battery_mod.FORMAT_CURRENT}.
  "scoring"              "{battery_mod.SCORING_CHOICE}" (default, and what you should
                         write) or "{battery_mod.SCORING_GENERATED}".
  "promptMode"           non-empty string; default "{battery_mod._DEFAULT_PROMPT_MODE}".
  "systemPrompt"         string or omitted. OMIT IT. A persona in the
                         battery's own context can suppress plain answers, and
                         declaring one earns a warning.
  "qwenThinkingEnabled"  boolean; default false. Leave it false.
  "maxTokens"            positive integer; default {battery_mod.BATTERY_MAX_TOKENS} — capability answers
                         are short.
  "description"          free text, optional, ignored by the scorer — a good
                         place to say what this battery probes.

The header is what makes the battery ISOLATED: this arming is applied
identically to the baseline, to every steering condition, and to every variant
condition, so the intervention is the only thing that differs across
conditions.

ITEM LINE — choiceProbability (the default; write these):
  {{"id": "...", "prompt": "...", "answer": "...", "options": ["...", "...", "..."]}}
  "id"       optional but recommended; a non-empty string, unique per file.
  "prompt"   REQUIRED, non-empty string. The question, plainly asked.
  "answer"   REQUIRED string, and it MUST appear verbatim in "options" —
             otherwise the item can never be scored correct.
  "options"  REQUIRED list of at least 2 non-empty, non-repeating strings.
             Write at least {battery_lint.MIN_DISCRIMINATIVE_OPTIONS}: with 2 options chance alone scores 50%,
             so the item has almost no room to fall.
  Do not add "grading" here — nothing is generated, so there is nothing to
  text-match.

  Scoring: the model's answer-token distribution over the declared options is
  read directly. Nothing is generated, so the score cannot move with response
  length, verbosity, or format compliance. That is the whole point.

ITEM LINE — generatedText (the escape hatch; avoid unless forced):
  {{"id": "...", "prompt": "...", "answer": "...", "scoring": "{battery_mod.SCORING_GENERATED}",
   "grading": "..."}}
  "grading"  REQUIRED for these items. An inferred mode is refused, because
             it changes silently when the answer text is edited. One of:
{grading}
  "options"  MUST NOT be present.
  Use this only for a capability that genuinely cannot be posed as a choice.
  Every grading mode here is length- or format-sensitive and the linter will
  say so.

================================================================
4. WHAT THE LINTER WILL CHECK
================================================================

The draft will be run through `steerlab-server battery lint`, an offline,
byte-only check. Write to these rules and it passes first time.

BLOCKERS (the battery cannot be used as a capability control):
  - the file does not parse, or an item violates the schema above;
  - no header line (the legacy format);
  - "answer" not among "options"; a repeated option; two options that
    normalize to the same string (case/punctuation-insensitive);
  - a generatedText item with no declared "grading".

WARNINGS (usable, but each one has to be defended in the methods note):
  - fewer than {battery_lint.MIN_ITEMS} items — one item then swings more than a tenth of the
    score, so item noise reads as a capability change;
  - fewer than {battery_lint.MIN_DISCRIMINATIVE_OPTIONS} options on an item (chance floor);
  - option lengths more than {battery_lint.MAX_OPTION_LENGTH_RATIO:g}x apart — joint logprobs favour short
    continuations, so keep options to short canonical labels of comparable
    length and put the descriptive text in the prompt;
  - one option being a prefix of another (the shorter borrows the longer's
    probability mass);
  - two items with the same prompt;
  - a declared "systemPrompt" in the header;
  - a response-format instruction anywhere in a prompt;
  - any generatedText item, per its grading mode's specific sensitivity.

================================================================
5. WORKED EXAMPLE
================================================================

The shipped template, from
{example_source}
Its items are EXAMPLES OF THE FORMAT, not a battery to copy — write your own
{count}, and do not reuse these:

{example}
================================================================
6. WHAT TO RETURN
================================================================

Return ONLY the JSONL: one header line, then {count} item lines. No prose
before or after it, no code fence, no trailing commentary. Prefer
choiceProbability for every item.

================================================================
7. THEN, IN THE WORKSPACE
================================================================

1. Save the JSONL as prompts/batteries/<name>.jsonl in the workspace.
2. Run: steerlab-server battery lint prompts/batteries/<name>.jsonl
3. Fix every blocker, and every warning you can fix without weakening the
   items. Re-lint until it reports no blockers.
4. Only then pin it in an experiment manifest. A battery pinned before it
   lints clean is evidence of nothing, and freeze will have stamped it.
"""
