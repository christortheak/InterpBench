# Emotion Grand-Mean Story Corpus Generator

You are helping build a multi-concept story corpus for emotion-vector
extraction. This is not positive/negative pair generation.

Concepts: `{{concepts}}`
Topics: `{{topics}}`
Stories per concept/topic: `{{stories_per_concept_topic}}`
Split: use `train`, `dev`, or `test`.

For each concept/topic combination, write short paragraph stories that depict
the concept through situation, behavior, bodily sensation, attention, and
interpretation. The concept may be named in metadata, but avoid directly
naming it inside the story text.

Requirements:

- Each story should be long enough for token-position pooling. Aim for
  90-160 words.
- Match topics across every concept so concept vectors are not topic vectors.
- Keep narrator/person, tense, register, and detail level balanced.
- Do not use content from your study's measurement domain (the task your
  steered runs are judged on).
- Do not make one concept systematically longer, more violent, more social,
  or more first-person than the others unless that is explicitly under study.
- Include concept, topic, split, and source metadata.

Output strict JSON Lines only:

```jsonl
{"id":"{{concept}}-{{topic}}-001","concept":"{{concept}}","topic":"{{topic}}","text":"...","split":"train","source":"llm-draft","notes":"..."}
```

No prose, numbering, comments, or code fences in the final answer.
