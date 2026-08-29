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


def _charter() -> str:
    """The battery charter, as the brief states it.

    The maintainer's three rulings, verbatim in substance including the
    real-analysis boundary example — the example is in the brief and not only
    in the docs because it is the one an author most often gets wrong, and a
    brief that stated the principle without it would be agreed with and then
    disregarded. Assembled from :mod:`.battery`'s own constants so the numbers
    a drafter is told cannot drift from the numbers the loader enforces.
    """
    protocol = battery_mod.GenerativeProtocol()
    return f"""\
================================================================
1a. THE CHARTER — read this before you write a single item
================================================================

Three rulings govern every battery in this project. They are not style
preferences; an item that violates one makes the battery unusable as the thing
it is for.

(1) EX ANTE JUSTIFIED, STUDY-BLIND, AND FIXED. An agent exists before any
    study it appears in, so the battery inherits NO study's protocol, domain,
    or difficulty target. It is a FLOOR — instruction-following, basic
    multi-step reasoning, factual recall, fluent extended generation, moderate
    difficulty — and never a frontier differentiator.

    THE BOUNDARY EXAMPLE, because this is the mistake that gets made: suppose
    a study measures relative performance on very hard, lengthy real-analysis
    proofs. The battery must NOT probe that capability, and must certainly not
    gate on it. Do not write a hard proof item, a graduate-mathematics item,
    or anything else pitched at the study's own frontier — not even "to be
    thorough". Capability at the study's own hard task is the STUDY's
    business. The battery's only question is whether the model still works at
    all.

    Practically: aim at items a competent model answers reliably and a
    genuinely degraded one starts to miss. If you find yourself reaching for
    difficulty to make an item discriminative, you have crossed the line.
    Whether the model is good at the study's own hard task is settled by
    performance in the study, not here.

(2) BOTH OPERATING REGIMES, AND THEY BELONG TO THIS FILE. A battery declares
    its own protocol, and it declares TWO:
      - GRADED items: short, greedy, scored 0/1.
      - LONG-FORM items: generated at a standard positive temperature
        (default {protocol.temperature}), with a generous token budget
        (default {protocol.max_tokens}) and several samples each (default
        {protocol.samples_per_item}), read for GENERATION HEALTH — length,
        distinct-2, completion rate — and never scored right or wrong.

    The justification for the second regime is that agents are USED
    generatively. It is not "because some study samples" — the numbers are the
    battery's, chosen ex ante, and no study may change them. The reason it
    exists at all: a short greedy battery scored accuracy 1.0 at a dose that
    three independent instruments had already confirmed was degraded. Nothing
    was wrong with the scoring. Twenty-four greedy tokens simply have no room
    for length inflation, variance collapse, or incoherence to appear in.

(3) SENSITIVITY IS VALIDATED, NEVER DEFINED, BY KNOWN POSITIVES. Before a
    battery ships it is run against a state already known to be degraded (a
    deliberately overdosed positive control), and it must FAIL there. If it
    cannot, it gets revised — on EX ANTE grounds, by rule (1), never by
    reaching for whatever happens to separate that particular control.

    The known positive is a check on the instrument, not a tuning target. A
    battery tuned until one specific dose fails has become a difficulty target
    wearing a control's name, which is exactly what rule (1) forbids. Write
    the items you would defend without ever having seen the control.

"""


def generation_prompt(count: int = DEFAULT_ITEM_COUNT, *, avoid: str = "",
                      root: str | None = None) -> str:
    """An LLM prompt the researcher can paste elsewhere to draft a format-2
    capability battery.

    ``count`` is how many items to ask for; ``avoid`` is free text naming the
    domain the battery must not touch (the study's own subject matter).
    """
    count = max(1, int(count))
    charter = _charter()
    protocol = battery_mod.GenerativeProtocol()
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

{charter}
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
3. THE FILE FORMAT — batteryFormat {battery_mod.FORMAT_CURRENT} or {battery_mod.FORMAT_TWO_REGIME}, JSONL
================================================================

One JSON object per line. The FIRST non-empty line is a header; every line
after it is an item. Never write the headerless legacy format: it is scored
under the surrounding study's prompt settings, which is why it can no longer
be certified as a control.

WHICH FORMAT TO WRITE — the charter's rule (2) decides it:
  {battery_mod.FORMAT_TWO_REGIME}  A FLOOR battery, both regimes, read by `steerlab-server battery
     run` against agents. This is what you want unless you were told
     otherwise: it is the only format that can see a generative failure.
  {battery_mod.FORMAT_CURRENT}  The PINNED per-condition control a study freezes into its
     manifest and scores inside its run matrix. One regime, graded only.
     A study may pin ONLY this (a sampled multi-sample regime scored per
     condition would be a second outcome measure wearing a control's name),
     so write it when the battery is destined for a manifest pin.

HEADER (first line) keys:
  "batteryFormat"        REQUIRED, {battery_mod.FORMAT_CURRENT} or {battery_mod.FORMAT_TWO_REGIME}.
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
  "{battery_mod.GENERATIVE_PROTOCOL_KEY}"  format {battery_mod.FORMAT_TWO_REGIME} ONLY, and the second regime's whole
                         protocol. An object with exactly three keys, each
                         optional:
                           "temperature"     > 0; default {protocol.temperature}
                           "maxTokens"       positive int; default {protocol.max_tokens}
                           "samplesPerItem"  positive int; default {protocol.samples_per_item}
                         Write the defaults unless you have a reason you can
                         state. They are the BATTERY's numbers, chosen ex ante
                         from how agents are used; no study may override them.

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

ITEM LINE — {battery_mod.SCORING_HEALTH} (format {battery_mod.FORMAT_TWO_REGIME} only; the second regime):
  {{"id": "...", "prompt": "...", "scoring": "{battery_mod.SCORING_HEALTH}"}}
  "prompt"   REQUIRED. An open request for EXTENDED prose — several
             paragraphs' worth. Ordinary generative work of the kind an agent
             is actually put to: explain something, summarize something,
             write a short piece, walk through a plan.
  "answer"   MUST NOT be present. Nothing is graded, so there is nothing for
             it to be right about.
  "grading"  MUST NOT be present, for the same reason.
  "options"  MUST NOT be present.

  Reading: the item is generated {protocol.samples_per_item} times (whatever
  "samplesPerItem" says), each sample seeded deterministically, and read for
  word count, distinct-2 (repetition collapse drives it toward 0) and
  completion rate (did generation END, or hit the token budget?). Means and
  population spreads are reported per agent, and compared against the
  baseline agent's. Nothing here is ever scored right or wrong.

  Write these to the charter's rule (1) as strictly as the graded items:
  moderate, generic, plainly-worded requests. NOT hard, NOT specialist, NOT
  in the study's domain. And do NOT ask for brevity — "in one sentence", "be
  concise", "keep it short" makes every agent's reading the same short one
  and the regime measures nothing.

  Aim for at least {battery_lint.MIN_HEALTH_ITEMS} of them in a format-{battery_mod.FORMAT_TWO_REGIME} battery.

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
  - any generatedText item, per its grading mode's specific sensitivity;
  - a format-{battery_mod.FORMAT_CURRENT} file (one regime — fine for a pinned control, not a floor
    battery: `singleRegime`);
  - a format-{battery_mod.FORMAT_TWO_REGIME} file with fewer than {battery_lint.MIN_HEALTH_ITEMS} generationHealth items,
    "samplesPerItem" below {battery_lint.MIN_GENERATIVE_SAMPLES} (no spread to lose, so variance collapse
    is unreadable), or a generative "maxTokens" below {battery_lint.MIN_GENERATIVE_MAX_TOKENS} (a budget that
    short clips every agent at the same number, so length inflation is
    invisible by construction);
  - a response-format instruction in a generationHealth prompt.

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
choiceProbability for every GRADED item; where you were asked for a
format-{battery_mod.FORMAT_TWO_REGIME} floor battery, spend {battery_lint.MIN_HEALTH_ITEMS} or more of the {count} on
{battery_mod.SCORING_HEALTH} items and the rest on choiceProbability.

================================================================
7. THEN, IN THE WORKSPACE
================================================================

1. Save the JSONL as prompts/batteries/<name>.jsonl in the workspace.
2. Run: steerlab-server battery lint prompts/batteries/<name>.jsonl
3. Fix every blocker, and every warning you can fix without weakening the
   items. Re-lint until it reports no blockers.
4. VALIDATE THE SENSITIVITY (charter rule 3). Run the battery against a
   baseline AND a state already known to be degraded — a deliberately
   overdosed positive control:

     steerlab-server battery run prompts/batteries/<name>.jsonl \\
       --model <id> --agent baseline --agent <concept>:<layer>:<a-large-alpha>

   The control must FAIL — its accuracy, or its generation health, has to
   move. If it does not, the battery is not sensitive enough to be evidence:
   revise it on ex ante grounds (rule 1) and run it again. Do NOT edit items
   toward whatever happens to separate this particular control.
5. Only then pin it in an experiment manifest (format {battery_mod.FORMAT_CURRENT} only) or cite a
   floor reading from it. A battery pinned before it lints clean is evidence
   of nothing, and freeze will have stamped it.
"""
