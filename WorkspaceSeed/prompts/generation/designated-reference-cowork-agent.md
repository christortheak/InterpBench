# Author a designated-reference comparison

Only generate data after the researcher approves the scope, authoring model and
worker roles. Otherwise return a proposal or a prompt they can use elsewhere.

Target construct: {{concept}}
Explicit reference construct: {{reference}}
Requested examples per population: {{count}}

Explain what distinguishes the target from this reference. Keep topics, formats
and incidental cues comparable. Ask about any unresolved definition. Author the
two populations separately; never replace the designated reference with a pooled
grand mean of other concepts. Return one JSONL file per population, with each row
{"text": "an example"}, for the respective prompts/emotions/<concept>/stories.jsonl
locations. Do not write into the workspace yourself. Keep final validation
examples separate from construction data and identify their intended role.

If multiple authors are approved, give them explicit populations and disjoint
assignments. Give an independent reviewer both definitions, examples and the
rationale. The reviewer should check the intended contrast, topic/style matching,
leakage and unsupported assertions using the actual rows, with plain findings
and affected row IDs. Report unresolved issues; do not manufacture successful
checks. The researcher reviews files before importing or running extraction.
