# scenario-template.json — minimal multi-agent scenario

A `MultiAgentScenario` (schemaVersion 1): shared materials, a list of
agents, and an ordered list of turns. This template is a two-agent
propose/review exchange over a neutral resource pool — replace everything
with your study's protocol; the shape is what matters.

Key fields:

- `agents[]` — `id`, `name`, `baseModelID`, `systemPrompt`; optionally
  `variantArtifactPath` + `variantArtifactHash` to run an agent as a
  saved model variant (steered/adapted) instead of the stock model, and
  `temperature` to give that seat its own sampling temperature (0 or
  more). A seat without one runs at the study's temperature; an agent
  file's temperature is never used.
- `includeOwnReasoning` — OPTIONAL, default `false`. A seat that thinks
  passes on its answer and never its reasoning: other seats, and
  `{{outputs.<label>}}`, get the answer alone. With this `true`, a seat's
  own earlier turns (its routed transcript entries and its own contract
  inputs) come back to it with the reasoning it wrote. A turn whose
  reasoning was cut off before it finished passes nothing on.
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
  vocabulary) refuses the whole scenario at load. A thinking seat's
  endpoint is read from its answer, never its reasoning; the stamp says
  `markerInReasoning: true` when the reasoning also wrote the marker.
- `temperature` / `maxTokens` — scenario-wide generation settings for a
  panel run on its own. A study that runs the panel sets the temperature of
  every seat that declares none. Whether a seat thinks is its occupant's: a
  cast agent's own setting, or the study's `reasoningEffort` for a seat cast
  as baseline. A thinking seat gets the study's `reasoningMaxTokens` when
  the study declares one.

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
