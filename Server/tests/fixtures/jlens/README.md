# J-lens report fixture

`jlens-readout.jsonl` is a hand-shaped trace, not a captured one: two
conditions × two layers × twelve steps, with values chosen so every aggregate
in `jlens report` can be computed on paper and asserted exactly.

What is planted in it, and why each plant is there:

| Plant | Purpose |
|---|---|
| `baseline` watched scores `[1, 2, 3]` at every layer and step | a flat reference, so any movement in the report is the plant and not noise |
| `steered` watched score `5.0` for token 10 **at layer 5 only** | the baseline-vs-condition delta must be found at the right layer and must be absent at layer 9 |
| token 10 mention-masked from step 6 onward | the mention mask must exclude those steps from every aggregate — half the steps, so a report that ignored it would be off by a factor the test can see |
| `topKIDs` `[10, 20]` under `steered` at layer 5, `[20, 30]` elsewhere | occupancy, and the "new in condition" flag for a token absent from the baseline's top-k entirely |
| a logit-lens companion at a *different* constant | the companion must be reported beside the J-lens number, never in place of it |
| every row carries the same `configHash` |  pooling must be able to see when two rows came from *different* readout configurations; the test flips one |
| a third row with `traceComplete: false` | incomplete rows must be excluded AND counted, never averaged in |

Regenerating it by hand is a mistake: the numbers are the test. If the trace
schema changes, change the fixture deliberately and update the hand-computed
expectations in `tests/test_jlens_report.py` in the same commit.
