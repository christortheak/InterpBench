# markers-template.json — surface-marker word list

```json
{"words": ["word1", "word2", …]}
```

Lower-case surface tokens whose density in generated text is counted as a
cheap expression diagnostic (marker density = marker tokens / total
tokens). Matching is textual, so include inflected forms explicitly.

## Where the file lives in a workspace

`prompts/concepts/<concept>/markers.json` — for EVERY attached concept,
grand-mean concepts included (scoring always resolves markers under
`prompts/concepts/`).

## Scope of the metric

Marker density is a diagnostic / manipulation check only. It measures
surface prose, not substance — never use it as a promotion objective for
outcome-level claims; select on judge score or logprob shift instead. The
file's aggregate hash is pinned into the manifest at freeze
(`markersHash`), so later edits surface as verify violations.
