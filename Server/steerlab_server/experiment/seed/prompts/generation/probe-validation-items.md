# Probe Validation Item Generator

You are helping build held-out labeled sentence examples for training and
validating a reading probe. These examples must not be reused for vector
extraction.

Concept: `{{concept}}`
Count: `{{count}}`
Label balance: exactly half concept-present, half concept-absent unless
specified.

Requirements:

- Each item should be one self-contained sentence.
- `expresses: true` items should depict the concept without naming it.
- `expresses: false` items should be hard negatives: matched topic/style but
  lacking the concept. Prefer contrastive near-neighbors over easy negatives.
- Avoid lexical shortcuts and obvious marker words.
- Do not use measured study prompts or near-paraphrases of extraction data.
- Use varied ordinary topics, balanced across both labels.
- Include split metadata (`dev` or `test`) and a brief topic.

Output strict JSON Lines only:

```jsonl
{"id":"{{concept}}-probe-001","text":"...","expresses":true,"topic":"...","split":"test","notes":"..."}
```

No prose, numbering, comments, or code fences in the final answer.
