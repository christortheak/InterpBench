# RepE / LAT Paired Reader Data Generator

You are helping build paired data for Representation Engineering reading
vectors and scalar concept estimation.

Concept: `{{concept}}`
Template or answer scaffold: `{{template_or_scaffold}}`
Count: `{{count}}`

Generate paired examples where `positive` and `negative` are matched prompts
or prompt fragments. The positive side should elicit a representation of the
concept; the negative side should be a close control.

Requirements:

- The pair must differ mainly in the target concept.
- Keep length and surface style closely matched.
- If using an answer scaffold, both sides must end with the same scaffold.
- Do not include the correct label in the text unless the recipe explicitly
  asks for label-word reading.
- Avoid study-task prompts; this is representation calibration data.
- Cover a balanced range of ordinary topics across the complete set.

Output strict JSON Lines only:

```jsonl
{"positive":"...","negative":"..."}
```

No prose, numbering, comments, or code fences in the final answer.

SteerLab assigns stable row IDs when it saves the paired artifact. For RepE
readers, the held-out count selected in Concept Lab determines the train/test
split; do not invent IDs or split labels in this response.
