# CAA Paired Stimulus Generator

You are helping build a contrastive activation-addition stimulus set for
activation-steering research.

Concept: `{{concept}}`

Generate `{{count}}` matched pairs. Each pair has:

- `positive`: a sentence that depicts the concept implicitly.
- `negative`: a content-matched sentence that does not express the concept.

Requirements:

- Do not name the concept or use obvious synonyms.
- Keep positive and negative similar in topic, length, syntax, specificity,
  valence, arousal, and concreteness except for the target concept.
- Use diverse topics and settings.
- Avoid content from your study's measurement domain (the task domain your
  steered runs will be judged on) unless the concept itself requires it.
- One sentence per side, roughly 10-25 words.
- No examples should depend on stereotypes or protected-class assumptions.

Output strict JSON Lines only, one object per line:

```jsonl
{"positive":"...","negative":"...","topic":"...","notes":"..."}
```

No prose, numbering, comments, or code fences in the final answer.
