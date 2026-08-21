# Starter concept: formality — A STARTING POINT, NOT A STUDY DESIGN

A worked, domain-neutral example concept so your first extraction runs on
day one. **Modify it — don't ship it.** Swap in your own concept by editing
these four files in place (or by creating a sibling directory) and keeping
the same shapes.

The concept is *formality of register*: the positive stimuli say the same
thing as the matching negative line, in formal register instead of casual
register. Content matching is the point — positives and negatives differ
*only* in the concept, never in topic, so the extracted direction tracks
register, not subject matter.

## Files and shapes (verified against the loaders)

| File | Shape | Consumer |
|---|---|---|
| `positive.jsonl` | one `{"text": "…"}` per line | `StimulusSet` (extraction) |
| `negative.jsonl` | one `{"text": "…"}` per line, content-matched to the same line of positive.jsonl | `StimulusSet` (extraction) |
| `validation.jsonl` | one `{"text": "…", "expresses": true|false}` per line | the convergent-validity gate (`validate`) |
| `markers.json` | `{"words": ["…"]}` | marker-density diagnostics (never a promotion objective) |

## Rules this example follows (keep them when you adapt it)

- **Validation scenarios are never-named:** they exhibit or avoid the
  concept *without using its vocabulary* (no "formal", "casual",
  "polite" anywhere in `validation.jsonl`), and they play no role in
  extraction. Freeze requires the validate evidence they produce.
- **Stimuli are independent of outcomes:** nothing here mentions the task
  domain you will measure. If your study measures X, no X content belongs
  in these files.
- **Markers are a diagnostic**, a surface-vocabulary manipulation check —
  behavioral endpoints (choice, judge score) carry the claims.

In a workspace this directory lives at `prompts/concepts/formality/`;
attach it to a study from Studies → Concepts → Attach Concept.
