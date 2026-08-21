# rubric-template.md — starting-point judge rubric

A domain-neutral paired-judging rubric: the criteria a judge scores a
blinded baseline-vs-condition response pair against. The **entire raw
file** is the rubric text handed to the judge (see your workspace's
`prompts/rubrics/README.md` for the pin convention), so there is no
schema beyond "plain text/markdown the judge can follow".

This is a STARTING POINT, not study content: replace or extend the
criteria with what your study's outcome actually turns on (e.g. "which
response imposes the more severe sanction", "which response relies on
rule-text rather than consequences"). Keep criteria observable in the
response text — a judge cannot score intent.

## Where the file lives in a workspace

The scaffold copies this template to
`prompts/rubrics/<experiment>-rubric.md` (the Evaluation pane's
"Create from template" uses the same destination). Any file under
`prompts/rubrics/` is offered by the rubric picker; pinning records the
file's SHA-256 into the manifest (`judgeRubricFile` +
`judgeRubricHash`), so later edits surface as drift until re-pinned.
Freezing a judge-evaluated study requires a pinned rubric file and at
least two judges.
