# scenario-template.json — minimal multi-agent scenario

A `MultiAgentScenario` (schemaVersion 1): shared materials, a list of
agents, and an ordered list of turns. This template is a two-agent
propose/review exchange over a neutral resource pool — replace everything
with your study's protocol; the shape is what matters.

Key fields:

- `agents[]` — `id`, `name`, `baseModelID`, `systemPrompt`; optionally
  `variantArtifactPath` + `variantArtifactHash` to run an agent as a
  saved model variant (steered/adapted) instead of the stock model.
- `turns[]` — `speakerAgentID` names who generates;
  `promptTemplate` is the turn's instruction; `routing` is one of
  `all` | `speakerOnly` | `selected` | `none` (with `routedAgentIDs`
  when `selected`) and controls which agents see the turn's output;
  `includeScenarioMaterials` / `includeSpeakerContext` gate what enters
  the speaker's context; `maxTokens` optionally caps the turn.
- `turns[].endpoint` — OPTIONAL. The quantity that turn is supposed to
  produce, parsed out of the generated text by the runner at write time and
  stamped on the turn record (`analyze` then aggregates it into
  `panel-endpoints.csv`). Two kinds:

  ```json
  {"name": "vote", "kind": "choice", "marker": "Vote:",
   "vocabulary": ["affirm", "reverse", "vacate", "remand"]}
  {"name": "months", "kind": "number", "marker": "Sentence:",
   "min": 0, "max": 600}
  ```

  Parsing is a literal, case-insensitive scan — no regex: find the FIRST
  occurrence of `marker`, then read the following 80 characters for a
  whole-word vocabulary member (earliest position wins) or the first
  decimal number (refused when outside `min`/`max`). Nothing found is
  recorded as unparsed, never guessed. So the `promptTemplate` must
  INSTRUCT the format — the declaration and the instruction travel
  together in this one reviewed file, as they do on the Review turn here.
  A malformed declaration (unknown kind, empty marker, a choice with no
  vocabulary) refuses the whole scenario at load.
- `temperature` / `maxTokens` — scenario-wide generation settings.

## Where the file lives in a workspace

`prompts/panels/<slug>.json` — the subtree the Multi-Agent panel scans
(`MultiAgentScenarioStore.directory`); the study-data scaffolder writes
`prompts/panels/<experiment>-scenario.json`. A panel script is ex-ante
input, hash-pinned like stimuli and task prompts, so it lives under
`prompts/` with them. The old home, `runs/multi-agent-scenarios/<name>/
scenario.json`, is legacy and read-only: `runs/` is gitignored, so the
freeze cleanliness gate ("every pinned input must be committed")
structurally could not see a study's most important input. A multi-agent
study pins the scenario file by hash (`multiAgentScenarioPath` +
`multiAgentScenarioHash`), so edits after pinning surface as verify
violations. There are no timestamps in the file for the same reason:
volatile metadata inside a content-hashed pinned input turns a no-op
re-save into hash drift; git holds the history, the file holds the
recipe.
