# human-baseline-template.csv — transcribed human-effect table

The first four columns are REQUIRED — they are exactly what the analyze
step's loader reads (one row per endpoint; `R = delta_model −
delta_human` is computed per endpoint against this table). Extra columns
are fine and travel with the file; missing required ones make the pin
refuse.

| column | meaning |
|---|---|
| `endpoint` (required) | the study endpoint this human effect anchors — must match the endpoint name your study measures |
| `deltaHuman` (required) | the human effect size (the `delta_human` in `R = delta_model − delta_human`) |
| `ciLower`, `ciUpper` (required) | confidence interval bounds as published |
| `source` | citation + table/figure the number came from |
| `n` | sample size behind the estimate |
| `notes` | transcription date + anything needed to re-verify |

## Transcription discipline (hard requirement)

Every row must be verified against the source paper — open the cited
table and check the number, the CI, and the n before committing. Never
transcribe from memory, from an abstract, or from a secondary citation.
Record the transcription date in `notes`. Do NOT commit the paper's full
text to the repo (copyright); cite it.

## Where the file lives in a workspace

`prompts/baselines/<experiment>-human-baseline.csv` (any path works —
the manifest pins whatever path you choose, by hash).

This table is required only for human-anchored (R) claims. Studies
without it remain valid at the model-internal result layer; pinning it
(`humanBaseline` in the manifest) is what unlocks the alien-residual
computation.
