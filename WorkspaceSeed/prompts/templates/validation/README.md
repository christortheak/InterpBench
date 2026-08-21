# validation-template.jsonl — never-named validation scenarios

One JSON object per line:

```json
{"text": "<a scenario>", "expresses": true | false}
```

- `text` — a short scenario. **The never-named rule (hard requirement):
  the text must EVOKE the concept without ever naming it or using its
  obvious synonyms.** If the concept word (or a near-synonym) appears in
  the text, the validation is circular: a vector could pass by tracking
  the word instead of the concept. The example rows here evoke a concept
  like "urgency" — note that no row contains the word.
- `expresses` — whether the scenario expresses the concept (`true`) or is
  a matched neutral/contrast scenario (`false`). Aim for a balanced set;
  4 rows is a floor for smoke checks, 20+ for evidence use.

## Where the file lives in a workspace

Resolution follows the extraction method pinned for the concept
(`ExperimentStore.conceptValidationRelativePath`):

- paired concepts (CAA mean-difference, LAT):
  `prompts/concepts/<concept>/validation.jsonl`
- grand-mean concepts:
  `prompts/emotions/<concept>/validation.jsonl`

The `validate` verb reads this file as the convergent-validity gate; its
hash is pinned into the manifest at attach, so edits after attach surface
as freeze/verify drift (re-attach to re-pin).
