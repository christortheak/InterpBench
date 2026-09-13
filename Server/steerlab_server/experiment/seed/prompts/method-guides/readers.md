# Train and use an activation reader

Measure an activation-based instrument separately from an intervention.

Decide the construct, template, label interpretation, training split, held-out evaluation split and scoring position with the researcher. A reader's score is not automatically an intervention or a behavioral outcome.

Use the existing reader template registry and reader fit/score interfaces. Request the schema and authoring prompt for the selected template; do not substitute ordinary contrastive-pair files for template-mediated reader inputs. Preserve label alignment, model revision, tokenizer/rendering settings and the exact fitted artifact.

Inspect the Readers pane or POST /api/reader/fit and /api/reader/score on a workbench engine. The registry below names supported public paths; these workbench routes are not advertised as batch-runner submissions. Treat training accuracy separately from held-out instrument performance. If importing a reader-derived steering vector, use experiment inspect-artifact and attach-artifact with exact artifact/sidecar/manifest reviews and retain its reader provenance.

Outputs are the fitted reader artifact and scored records. Report the training/evaluation split and applicability to this model and rendering. A successful fit does not establish causal control of the named construct.

## Dataset and handoff

Choose exactly one row shape. `contentPair` compares content under one template:

```json
{"id":"pair-1","concept":"construct","positiveStimulus":"First matched text.","negativeStimulus":"Second matched text.","topic":"shared topic","split":"train","templateID":"chosen-template"}
```

`singleStimulus` compares two template instructions on the same content:

```json
{"id":"row-1","concept":"construct","stimulus":"An ambiguous situation.","topic":"shared topic","split":"train","templateID":"chosen-template-pair"}
```

Emit `authoring prompt reader-pairs --concept <concept> --positive <definition> --negative <definition> --template-id <template> --shape <contentPair|singleStimulus> --json`. Trailing `split:"test"` rows orient the fitted sign in this implementation; reserve an additional independent evaluation set before making a generalization claim. Do not relabel sign-selection data as untouched final evaluation. Templates are real registered inputs, not arbitrary IDs to invent. The fit/score request schema is available in the workbench's `/openapi.json`.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.


## Probe library: shared discovery and inspection

The Probes section is the library for saved activation probes. Inspect it from
an agent with `steerlab science probe-list --root <workspace> --json`, or
`steerlab-cli science probe-list --workspace <workspace> --json`. Use
`science probe-inspect runs/<run>/<name>.probe.json` with the same workspace flag
to inspect one entry. Legacy Python `<name>-probe.json` files are also recognized.
Both workbench HTTP implementations expose
`POST /api/science/workspace/probe-list` with `{"workspaceRoot":"<workspace>"}`,
and `POST /api/science/workspace/probe-inspect` with the same root and a `path`.
These actions inspect the serving workbench's workspace; they do not access a
runner's remote library or copy files automatically. Collect run evidence first.

A list reports `probes`, `issues`, and `count`. Inspection reports the relative
path, exact-byte SHA-256, format, model, layer, method, limitations, and original
JSON document. The portable CLI places this under `result`; the native CLI's
existing workspace adapter places it under `result.response`. HTTP returns it
directly. All operations are read-only; malformed matching files are reported as
issues rather than silently dropped. Inspection is bounded to 64 MiB per JSON
file; activations belong in separate datasets. Hashes identify the original file,
not a re-encoding of the returned document.

Existing Mac readers and Python readers retain their original formats and score
semantics. In particular, the legacy Python reader's `heldOutAccuracy` selected
a layer; it is not final-test accuracy. Missing model, rendering, or coordinate
pins stay unknown. Discovery does not establish cross-backend compatibility.
Native Playground reading still uses its existing supported reader path; the
new portable classifier artifact is not silently inserted into that picker.

The new `activation-probe` v1 format supports explicit binary linear and small
ReLU classifier parameters, preprocessing, input bindings, and score semantics.
Its CPU reference scorer defines inspectable arithmetic. The managed capture,
fitting, and evaluation workflow below produces these instruments. Study
measurements and conditional interventions remain subsequent slices. Do not invent training verbs
or generate data because a researcher has only named a concept. Offer existing
files, pasted data, author/reviewer prompts, or explicitly chosen coworkers.

## Capture, fit, and evaluate a portable probe

Use `probe-capture`, `probe-train`, and `probe-evaluate` with `science interview`,
`science draft`, and `science publish` in either client. The app's Probes section
opens exactly these interviews. Requests then use the standard managed plan,
submit, jobs, export/fetch/import, and custody workflow. No new engine CLI verb
or implicit background training is needed. Capture runs on a Python model engine;
fitting and evaluation are CPU operations without a loaded model.

Explain the research choices in plain language: what the label means, which
text population it represents, where to read the model, and what independent
examples would show whether the probe generalizes. A probe reads activations;
a steering vector or intervention changes them. Offer an existing file, pasted
data prepared by the researcher, an authoring prompt, or explicitly selected
coworkers. Do not start authoring datasets or delegating work from a concept name.

### Text input and role assignment

Capture takes a pinned UTF-8 JSONL file with one example per line:

```json
{"id":"example-1","group":"document-1","text":"Replace with a labeled example.","label":true,"split":"fit"}
{"id":"example-2","group":"document-2","text":"Replace with a matched control.","label":false,"split":"selection"}
{"id":"example-3","group":"document-3","text":"Replace with an independent example.","label":true,"split":"finalTest"}
```

These three format examples are not enough to train or validate a useful probe.
IDs must be unique; related passages, paraphrases, or repeated observations share
one group and role. Include both label classes in each intended role. Capture
writes separate `fit-activations.json`, `selection-activations.json`, and
`finalTest-activations.json` files for nonempty roles. Fitting needs at least two
rows per class, but useful evaluation usually needs many independent groups.

If role assignment is undecided, propose a split first. The explicit alternative
`splitPolicy: groupHash` omits the `split` key and assigns whole groups using a
seeded hash (60% fitting, 20% selection, and 20% final testing in expectation).
Small splits are not automatically rebalanced. Review counts before training.

Choose a zero-based decoder layer, input or output of the block, raw or chat
rendering, and final non-padding token or each non-padding token. Chat capture
uses one user message plus a generation cue. Raw complete-text capture replays
text; it does not measure the original live generation. A full-text label is
not automatically meaningful at every prefix. Capped tokenization and the first
N selected rows are visible in the capture report; inspect actual token IDs.

### Fit and assess

Select fitting activations, choose a family, and give the probe a name. Mean
difference is a simple linear reader; the regularized linear classifier learns
a boundary; the small ReLU classifier can represent nonlinear boundaries but
has more opportunity to overfit. Standardization uses fitting data alone.
Optional selection activations report a comparison without changing the fit.
Changing settings after reviewing that comparison is model selection: record it,
and reserve final-test data. The optional shuffled-label fit is a separate control.

Review input bytes and the effective settings, then explicitly execute. Capture
has a 64 MiB activation JSON pilot budget and a row cap; fitting reuses those
saved activations without reloading the model. CPU fitting uses float64 full-batch
gradient descent with fixed steps, a recorded local seed, and weight-only L2.
No final-test labels choose preprocessing, layers, thresholds, or stopping times.

Evaluate a pinned `trained.probe.json` on matching saved activations. Read the
new `evaluation-report.json`: class and group counts, confusion counts, accuracy,
balanced accuracy, precision, recall, specificity, F1, ROC AUC, and constant-class
baselines. Null metrics have undefined denominators; scores are not calibrated
probabilities. The report flags known row/group/text or file-hash reuse. Reused
inputs remain usable for exploration, but do not describe them as independent
final-test evidence. Different hashes alone do not prove independence.

New probes appear after evidence collection and library refresh. Existing native
readers remain available for Playground highlighting. Study measurements execute on the Python engine through the settings below.
Conditional interventions remain a later step; successful fitting does not change behavior.

### Prompt to offer a data author

Prepare labeled text only after I approve the scope and number of examples.
Use the JSONL schema above. State the positive and negative label definitions,
match plausible confounds such as topic and length, and group related examples.
Keep whole groups in one role. Supply the data and a short explanation of sources,
label uncertainty, coverage, and known limitations. Do not claim to have run
measurements or independent review that you have not performed.

### Prompt to offer an independent reviewer

Check the actual rows against the approved label definitions and JSONL schema.
Inspect label correctness, duplicate text, related examples across splits, class
balance, and confounding cues. Explain concrete problems in ordinary language,
with row IDs and suggested repairs. Distinguish a format check from evidence
that the labels measure the intended concept. Do not invent measurements.


### Record probes during a study

The researcher selects trained probes as **study measurements**, independently
of the agent (base model, vectors, and adapters) whose responses are measured.
Use Studies → Measurements in the app, or the same reviewed commands in either
client (`--root` for `steerlab`; `--workspace` for `steerlab-cli`):

```sh
steerlab science measurements-review example --settings measurements.json --root /path/to/workspace --json
steerlab science measurements-save example --settings measurements.json --plan-sha256 <review-hash> --root /path/to/workspace --json
```

Prepare `measurements.json` using exact paths and digests from `science probe-list`:

```json
{
  "schemaVersion": 1,
  "probes": [{
    "id": "selected-reader",
    "probe": {"path": "runs/fit/trained.probe.json", "sha256": "<exact SHA-256>"},
    "conditions": [], "agents": [], "stages": ["prefill", "decode"],
    "recordingStage": "postAction"
  }],
  "onError": "recordMissing", "maxReadings": 4096,
  "retainActivations": false, "maxActivationBytes": 1048576
}
```

An empty conditions list means every condition. An empty agents list means every
agent instance; panel studies use seat IDs, and ordinary studies use condition
names. `prefill` reads prompt positions; `decode` reads the generated prefix.
The artifact declares block input/output, layer, and last/every prompt position.
`preAction` and `postAction` are readings before/after existing steering at that
site. Earlier sites may already be affected. Reading does not inject anything.

Review, save, verify, and freeze normally. Freezing copies run-resident probes
into the study's pinned inputs without changing their bytes. The study bundle
carries them to the Python engine; collected evidence carries the scores home.
Select Python Compute, including a local Python server, in the app. Native MLX
cannot execute these portable PyTorch probes and gives a direct routing repair.
Direct-logprob-only studies do not generate responses for this observer; select
sampled text if response measurements are wanted. No hidden computation is added.

Read `probeMeasurements` in each `generations.jsonl` response or panel turn.
Results → Generations displays the probe readings, scores, positions, stages,
and missing-status reasons. Panel records retain seat identity and replicate
identity. The enclosing response supplies the prompt, condition, and sample.
Each activation at input position t predicts t+1. The final sampled token is
not read unless a subsequent forward pass naturally consumes it. Token IDs,
rather than guessed text offsets, are authoritative alignment.

Unknown qualification or population transfer is guidance, not a refusal. A probe
trained on full examples may not predict meaningful labels at every prefix.
Runtime model, revision, tokenizer, template, substrate, coordinates, width, and
precision must match the artifact to compute its declared reading. Incompatible
bindings stop before generation. Non-finite/failed score calculations use the
selected `recordMissing` or `stop` policy. Stopped generations preserve a new
`probe-failure-*.json` alongside partial evidence, not a completed response.

Limits apply per response across its selected probes. Scores are retained up to
`maxReadings`; later scheduled readings are counted as omitted. Activations are
opt-in, with a separate bound on encoded activation-array bytes; this is not a
bound on model RAM or the entire evidence file. Scores remain when activation
retention reaches its limit. Distinguish omissions, missing readings, and genuine
scores. Do not describe scores as calibrated probabilities or causal effects.
Do not silently choose labels, datasets, or cooperating agents for the researcher.

Study measurements use an explicit order: requested readings before the site's
steering actions, the existing ordered action chain, and requested readings after
those actions. Before an action at this layer does not mean earlier layers or
tokens were unsteered. Readings and their state belong to the current response
and agent seat; a cancelled or superseded response must not affect another one.
Existing additions and subspace removals retain their distinct units and token
scopes. Conditional rules and token-selection actions belong to separately saved
intervention policies; measurement settings never turn a reading into a policy.

## Intervention policies: change an agent using a probe

A study measurement records scores. An intervention policy changes an agent's
behavior using those scores. Treat these as different research decisions: ask
whether the researcher wants to observe, intervene, or compare both. Explain a
threshold as “act when this score exceeds the chosen value,” and an adaptive
rule as “increase strength with the score, within limits.” Choose thresholds
and limits using validation data; reserve untouched examples to evaluate the
complete modified agent. Do not choose those scientific decisions or launch a
job without the researcher’s agreement.

The Probes and Agents tabs offer **Intervention policies…**. Guided authoring
selects a fitted probe, a rule, and a vector or exact token IDs. The probe supplies
the model/revision, tokenizer, rendering, and activation site. Advanced settings
can combine named probe scores, set token schedules, and pin trusted Python
provider source and assets. Providers are executable expert code with the
engine's permissions, not sandboxed plugins; inspect their source with the
researcher before using them.

Both `steerlab` and `steerlab-cli` expose the same local authoring verbs:

- `science policy-list --root <workspace> --json`
- `science policy-inspect <policy-path> --root <workspace> --json`
- `science policy-review <settings.json> --root <workspace> --json`
- `science policy-publish <settings.json> --plan-sha256 <review-hash> --root <workspace> --json`
- `science policy-attach-review <attachment.json> --root <workspace> --json`
- `science policy-attach <attachment.json> --plan-sha256 <review-hash> --root <workspace> --json`

Attachment settings are `{ "agentPath": "runs/<run>/agent.json", "name": "new-agent",
"policyPaths": ["runs/<run>/intervention.policy.json"] }`. Attachment makes a new
agent version, replacing its policy list and retaining its static vectors and
adapters. An empty list makes the corresponding policy-free comparison agent.
The original agent is untouched. Attach the new version using the ordinary
agent/study or panel-seat workflow; freezing and packaging carry its embedded
policy inputs. The workbench API exposes the same owners at
`POST /api/science/workspace/<action>` with `workspaceRoot`, `settingsText`, and,
for publication, `planSHA256`; list takes only the root, and inspect adds `path`.

Execute policy agents through Python Compute in sampled-response studies or
panels. Direct choice scoring, battery qualification, native MLX execution, and
Playground’s editable steering controls do not execute these policies yet and
explain the supported route. The Python agent-chat API executes policies; study
runs are the durable comparison path. Decision traces accompany responses and
identify score inputs, action strengths, omitted records, and failures. They do
not turn a probe score into a probability or establish causal importance.
See the repository's `docs/INTERVENTION-POLICIES.md` for the version-1 format,
action ordering, provider ABI, and examples.

## Complete policy format and provider ABI

This format is included in the installed app and app-free client; no checkout is
needed to obtain it through `science guide readers`.

A minimal fixed-direction settings file (synthetic model and revision):

```json
{
  "schemaVersion": 1,
  "name": "fixed-comparison",
  "binding": {
    "modelID": "example/model",
    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "tokenizerSHA256": null,
    "rendering": "raw",
    "coordinateConvention": "hf-decoder-block-v1/model.layers"
  },
  "site": {"kind": "residualPost", "layer": 0},
  "stages": ["prefill", "decode"],
  "positions": "lastPosition",
  "probes": [],
  "actions": [{"id": "offset", "kind": "add", "bounds": [0, 1], "vector": [1, 0]}],
  "rules": [{"action": "offset", "kind": "fixed", "value": 1}],
  "onError": "stop",
  "maxEvents": 2048
}
```

Replace the synthetic binding and direction with the actual model and artifact.
For a library direction, replace `vector` with
`"vectorArtifactID": "runs/<run>/<vector-stem>"`. Review reads the `.safetensors`
and `.json` files, verifies known model/substrate/revision compatibility, and
embeds the selected layer's direction plus both source hashes. A missing source
revision is an explicit provenance limitation, not invented compatibility.
An explicit numeric `vector` is expert input and carries no automatic extraction
provenance. Neither direction is silently normalized for addition.

To use a probe, add `{"id":"reader","path":"runs/<run>/<name>.probe.json"}`
to `probes`. Publication embeds its exact UTF-8 text and SHA-256. A threshold rule
is `{"action":"offset","kind":"threshold","weights":{"reader":1},
"threshold":0,"below":0,"above":1}`. The comparison is strictly greater than;
equality takes `below`. An affine rule has `weights`, `slope`, and `intercept`
instead; its strength is clipped to the action bounds. Fixed and threshold
strengths must already be inside their declared bounds. Weighted sums combine
named probe scores; an MLP probe has the same scalar-score interface.

An action is one of `add`, `ablate`, `logitBias`, `allowTokens`, or `forceToken`.
Residual actions take a direction; token actions take a nonempty distinct `tokens`
list and use `site: {"kind":"logitsPreSelection"}`. Token constraints use bounds
`[0,1]` and are active above zero; `forceToken` requires one exact ID. A non-null
exact tokenizer hash is required for token actions. Token IDs outside the
vocabulary are refused at execution before selection.

`lastPosition` means the final prompt position and each consumed generated
position. `allPositions` also acts throughout prefill. The `stages` list can
select either stage or both. No additional pass reads the final sampled token.
Decisions and measurements may target different populations; validate that
transfer, rather than treating it as instrument qualification.

Attachment settings:

```json
{
  "agentPath": "runs/<source-run>/agent.json",
  "name": "new-agent-version",
  "policyPaths": ["runs/<policy-run>/intervention.policy.json"]
}
```

The selected policies replace the new agent's policy list; an empty list creates
its policy-free comparison. Static vectors, adapters, and unknown artifact
fields survive. A sweep-promotion certificate is removed from the changed
version because it was not the agent selected by that sweep. Existing agents,
old manifests, and runs are never rewritten. The exact embedded policy JSON
changes the new agent's identity and travels with it through existing closure.
Review hashes are external preconditions, never new revision fields in old
content-hashed documents.

## Expert provider ABI

Replace `rules` with `[]` and add
`"provider":{"sourcePath":"inputs/provider.py","assetPaths":{"reference":"inputs/asset.bin"}}`.
Paths must be ordinary files inside the authoring workspace. Review embeds source
text and base64 assets with exact hashes; it does not import or execute source.
At execution, source is compiled into a fresh response-local namespace. The
entry point is:

```python
def decide(context, tensor, scores, state, rng, assets):
    state["calls"] = state.get("calls", 0) + 1
    # Decision is supplied in the provider namespace.
    return [Decision("offset", min(1.0, rng.random()))]
```

- `context` supplies `site`, absolute consumed-token `offset`, `token_count`,
  `prompt_token_count`, response identity, and lazy positions/stages.
- Residual `tensor` has shape `[1, tokens, hidden]`; logits have `[1, vocabulary]`.
  The provider receives an isolated copy. Scores are named `[tokens]` float32
  tensors, with preprocessing and fitted affine/ReLU layers applied in order.
- `state` is a private dictionary for this policy and response. It is cleared on
  normal completion, failure, cancellation, or superseded-session teardown.
- `rng` is an independent `random.Random` instance seeded from the policy hash
  and sorted response context, including the study seed and panel seat. It does
  not consume the model sampler's global RNG. Expert code must use it and must
  not mutate model state or global random generators. This is a contract for
  trusted code, not a sandbox guarantee.
- `assets` maps names to the exact pinned bytes. Do not open undeclared files,
  fetch new inputs, execute another model, or change generation control flow.
- Return a list of `Decision(actionID, strength)` objects, each action at most
  once. Strength is a scalar or a tensor of shape `[tokens]`; declarations,
  finiteness, bounds, and schedules are checked before any action is applied.
  Omitting an action means no action at that call. Built-in rules and an expert
  provider are exclusive within one policy; separate policies can compose.

There is a 16 MiB policy limit and a 16 MiB complete agent-definition limit,
at most 16 policies per agent, 16 probes/actions
per policy, and 65,536 recorded events per policy/response. These are document
and recording limits, not guarantees about the model's memory use. Trace limits
do not stop decisions. Failure summaries retain at most 64 messages of 2,048
characters each, alongside the total failure count. Provider failures can stop or explicitly skip that
policy's current decision; cross-policy action conflicts and invalid output
activations/logits stop execution because there is no unambiguous safe result.

Policy probes and action arithmetic use float32 on the activation device.
Each action phase casts back to the block dtype; this is a new policy-v1
arithmetic contract, not a claim of bitwise equivalence to legacy static
injection rounding. The existing static action owners are unchanged.

Expert callbacks run once per selected forward chunk, not once per token.
Use context positions when defining a token-level algorithm; an algorithm that
counts callbacks or uses the entire supplied chunk can depend on prefill
chunking. Validate that behavior explicitly. `skipPolicy` drops the current
call's actions and records the error; it does not roll back arbitrary changes
an expert provider made to its own response state. Use `stop` when such a
continuation would be scientifically ambiguous.

During the phase-1 implementation slices, execute only with the current policy
engine. An older engine can ignore the new agent field. P6 must complete explicit
policy-runtime capability negotiation before mixed-version remote deployment;
local implementation tests do not establish that admission contract.
