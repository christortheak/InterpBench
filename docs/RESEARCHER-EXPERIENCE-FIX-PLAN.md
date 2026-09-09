# Researcher experience: first-use findings and remediation plan

2026-09-07. Based on the researcher's live Mac/app-and-agent observations and
source inspection of main `e4a06b3`. This document is a proposed implementation
handoff. No runtime changes, live model executions or dataset authoring were
performed during this investigation. Source paths below are repository-relative.

## Product decision

The app and research assistant should help a person work with a few consistent
objects: a base model, configured agents, data, interventions, validation,
templates, studies and results. They should explain a technical choice in terms
of the research question before presenting its implementation details.

The central experimental object is an **agent**: a base model with a declared
combination of steering directions and/or adapters. An unmodified base model is
the baseline agent. An agent may have several weighted vector interventions;
adapter application and scaling follow their own declared conventions, not the
same arithmetic as adding activation vectors. A study compares configured agents
or their interactions under declared conditions.

Data authoring, extraction, validation and parameter exploration are steps in
constructing and assessing those agents. Intermediate analyses remain useful in
their own right; do not force every exploratory action to manufacture a saved
agent first merely to fit this explanation.

## Shared language and explanations

| Researcher-facing term | Short explanation to use consistently |
| --- | --- |
| Concept | The property or behavior the researcher wants to investigate. Examples are an operational definition; naming a concept does not prove a vector represents it. |
| Concept data | Examples and comparisons used to estimate a steering direction. Explain which inputs the selected extraction method needs. |
| Extraction | Run the model on those examples and estimate a direction in its internal activity associated with the contrast. |
| Steering strength | How strongly the direction is applied. Introduce “dose” as the technical synonym, then state the units for this intervention. |
| Validation data | Separate examples used to check whether the extracted direction distinguishes the intended contrast beyond its construction examples. Explain the actual validation metric; it does not alone prove a causal behavioral effect. |
| Capability battery | Other tasks used to check whether the intervention damages abilities the researcher wants to preserve. Identify which abilities were and were not tested. |
| Parameter exploration (sweep) | Try a planned set of layers and strengths, compare the results, and choose a configuration using development data. Final evaluation should use separate data where the claim requires it. |
| Agent | The base model together with the selected interventions and their settings. |
| Template | A reusable starting point for creating studies. Creating a study from one does not run it. |
| Study | The declared comparison, data and measurements used to assess agents or their interactions. |
| Calibration corpus | Reference text used to measure the typical scale of model activations for supported strength conventions. |
| Projection corpus | Reference text used to estimate common activation patterns for optional projection/removal. Its suitability is a scientific choice; removal can discard useful signal as well as nuisance variation. |

Teach these at their point of use. Avoid a compulsory tutorial or a glossary wall.
Offer more detail on demand, preserve expert controls and machine identifiers, and
do not silently change numerical meanings to simplify labels.

## Priority and scope

1. Stop the neutral-PC build from blocking the app; establish cancellation and
   correct selection/state capture before optimizing presentation.
2. Repair agent interaction policy and offer explicit data-source choices before
   more live researcher trials consume generation quota.
3. Correct the model picker, Results viewer, tab content ownership and Templates
   terminology. These directly prevent or misdirect ordinary work.
4. Rebuild the neutral-corpus and Gemma Scope explanations and prerequisite flow;
   complete method-specific prompt support and extraction-control explanations.
5. Run full researcher journeys in both app and agent surfaces, then independent
   review, appropriate automated checks and the established integration process.

These are linked slices of one usability correction, not a request for another
broad architecture refactor. The crash/hang and protected-state contracts require
technical care; simple copy changes do not need speculative new test frameworks.

## UX-01 — neutral-PC build blocks the Mac app

**Observation:** clicking Build neutral PCs produced a beach ball and required a
force quit. Do not describe that as merely a slow computation.

**Source evidence:** `ChatService` is `@MainActor`. Its `buildNeutralPCBasis`
awaits capture, then calls `bank.componentsByLayer(selection:)` synchronously.
`NeutralActivationBank.componentsByLayer` loops through per-layer PCA. The
button's `Task` does not move that synchronous work off the main actor.
This is a confirmed blocking path and a likely contributor, not a reproduced
diagnosis of the entire reported hang. Model loading, memory use and file I/O
also need observation. The busy flag is set only after model readiness work.

**Owners:** `Sources/ExperimentKit/ChatService.swift`,
`Sources/SteeringKit/Extraction/ConceptExtractor.swift`,
`Sources/SteerLabApp/ConceptsPanelView.swift`.

**Required behavior:**

- Snapshot the intended workspace, model/revision, corpus/hash and settings for
  the job. UI selection changes must not retarget it after an await.
- Publish preparing/running state immediately. Separate loading, capture, PCA
  and saving in progress; show the actual corpus, not a hardcoded norm path.
- Move expensive pure numeric work and appropriate I/O to an execution owner
  outside the main actor while respecting MLX/model ownership and sendability.
- Provide meaningful progress and cancellation. If an individual numeric call
  cannot be interrupted safely, explain that cancellation waits for that unit;
  do not claim instant cancellation or publish a partial basis as complete.
- Keep navigation, scrolling and relevant cancel/status controls responsive.
- Read-only prerequisite checking should identify an empty corpus before model
  loading. Bound memory and retain the existing draw/selection provenance.

**Acceptance:** with a pinned disposable corpus and cached model, the UI remains
usable during all phases; cancellation has a truthful terminal outcome; failures
leave no apparently complete artifact. Record CPU/main-thread profile, memory,
phase timings and result identity. Do not change PCA arithmetic as part of moving
its execution; use the appropriate mechanical proof and numerical regression checks.

## UX-02 — agent starts data generation/delegation without a choice

**Observation:** agreeing on a model and concept led to manifest authoring and
delegated dataset generation without a separate data-source/cost decision.

**Source evidence:** the maintained workspace contract tells assistants to choose
scientific settings with the researcher and to give coworkers author/reviewer
prompts, but does not establish a clear data-sourcing decision before generation.
The concept-study interview explicitly permits authoring stimuli into a pack.
Those instructions leave excessive room to infer permission from concept choice.

**Owners:** `docs/AGENTS-WORKSPACE-DRAFT.md`, maintained study interviews and
method/authoring guidance, with packaged and compiled copies regenerated through
the existing pipeline. Edit maintained sources, not generated outputs alone.

**Required instruction:**

“Choosing a model, concept or method authorizes discussion and preparation of the
agreed draft. It does not by itself authorize producing a dataset, launching
workers, starting model computation or paid API calls. First establish whether
the researcher will supply data, wants a prompt for another tool, or wants you to
generate it. Prior explicit authorization persists; do not ask again for an
already agreed batch.”

Offer these parallel routes in plain language:

- Import an existing file.
- Paste existing data.
- Copy an authoring/review prompt to use with a chosen tool or model.
- Ask this assistant to generate data under an agreed scope.
- Delegate to coworkers under an agreed author/reviewer plan.

For generation/delegation, agree on dataset type and size, preferred model/tool,
number and responsibilities of workers, review approach and any quota/time limit.
Report unknown costs honestly. Do not assume the coordinating model is the best
or cheapest data author; do not assume that “use agents” means one worker or
unlimited parallel workers. The researcher chooses the route; the assistant can
recommend a modest plan without executing it first.

**Acceptance:** a cold agent given only a model and concept explains the next
scientific choices and offers data routes; it does not generate a corpus or spawn
a worker. File/paste/prompt routes work without generation. Explicit generation
authorization permits the agreed work without repeated confirmation. Repeat the
trial with novice and expert researchers and with an existing approved plan.

## UX-03 — agent explanations are too technical and theatrical

Lead with the effect on the study, then the smallest useful explanation and next
step. Keep parser errors, audit implementations and detailed evidence available
under a clearly labeled optional detail section or report link.

For the reported review problem, an appropriate researcher-facing summary is:

“The dataset needs revision before we use it. Some quality checks reported a pass
without actually measuring the examples, and the review found examples that
break the requirements. About half also exceeded the length limit. The checks
and affected examples need correction, followed by another review.”

Do not flatten a real scientific defect into “formatting trouble,” call a partial
result a success, or hide the review evidence. Avoid celebratory narration such
as “the review earned its keep”; explain what the issue means for the research.

Adapt explanation depth to the user. Introduce a term briefly on first use, then
reuse it consistently. Keep public CLI identifiers in optional technical detail
where needed for exact reproduction. A user should not have to learn filesystem
implementation labels to choose the next scientific step.

## UX-04 — Playground lists non-chat artifacts as models

**Observation:** Gemma Scope and other non-chat assets appear as choices.

**Source evidence:** Playground uses the shared `WorkspaceModelPicker`, backed
by `ChatService.workspaceModelOptions`. That path obtains a general local or
remote inventory; the picker shows memory-fit/installation state but does not
itself establish conversational-model eligibility. Trace inventory classification
and actual installed metadata before changing the shared inventory globally.

**Required behavior:** distinguish a loadable language model from an SAE, lens,
adapter or other cached artifact using capability/metadata evidence. Playground
offers suitable chat models by default. Supported base/completion models may be
available in an explicit completion mode; they should not masquerade as chat
models. Do not implement a name-substring denylist or hide unrecognized models
solely because their publisher is unfamiliar. Explain uncertain compatibility
and provide the appropriate supported discovery/load route.

**Acceptance:** mixed cache fixture with chat, base, SAE, lens and adapter assets;
correct local and remote choices; no model download triggered by inspection;
artifact availability in the actual analysis/import panels preserved.

## UX-05 — Results relies on Quick Look to read JSONL

**Source evidence:** `SectionContainers.swift` renders bounded JSONL/CSV/text
previews, then tells the user to Quick Look or open the full file elsewhere.
The reported system Quick Look has no useful JSONL viewer. An external viewer is
not a dependable completion of the app's results-reading workflow.

**Required behavior:** open full local JSONL/CSV/text content in an app-owned
reader. Use lazy rows/pagination or incremental loading for large files, with
clear total/loaded counts, search where practical, selectable/copyable values,
and expandable records. “Show all” must not mean decoding an unbounded file on
the main thread. Preserve raw content and display parse errors with line context.
For remote files, clearly offer the existing verified fetch/import route before
reading the full retained local file. Keep export/external open optional.

**Acceptance:** small file, many rows, very long single row, malformed JSONL,
Unicode, empty file and remote preview. Every retained record is reachable inside
the app; no silent omission and no new UI freeze. Do not mutate source evidence.

## UX-06 — Templates says Designs; Data content appears beside Templates

**Source evidence:** `TemplatesPanelView.swift` labels its section “Designs” and
picker “Design.” `WorkbenchSection` sends Templates to the shared activity feed,
and `ActivityFeedColumn` intentionally renders workspace-wide activity. A pinned
viewer can also override the section's viewer. Inspect whether the reported data
explanations came from that feed or retained/pinned detail state; the exact visual
cause has not been reproduced.

**Required behavior:** use **Template(s)** consistently in the app, including
actions, help, empty states and errors. The technical `design` CLI family and
serialized identities need not be renamed in this usability slice; explain the
mapping only where technical users need it. Do not change frozen hashes or
artifact names to correct a display label.

Templates should display the selected template or an explanation of templates.
Data explanations stay with Data. Preserve useful job logs in an explicitly
opened activity surface, rather than using them as unrelated default detail.
Section-specific instructional/data views must not leak across tabs through pin
state. Specify the intended pin behavior for these views explicitly.

**Acceptance:** Data → Templates → Data with no selection, selection, active job
and relevant pin states; switching templates updates template detail; workspace
switching does not retain another workspace's detail; responsive layout preserved.

## UX-07 — designated-reference data has no Cowork prompt

**Source evidence:** both `ConceptsPanelView.copyLLMPromptControls` and
`ConceptBuilder.coworkGenerationPrompt()` restrict this path to `emotionGrandMean`.
The ordinary generation prompt already handles designated-reference story input.
This is an explicit missing prompt path, not a mysterious disabled button.

**Required behavior:** provide a designated-reference-specific author/reviewer
prompt that identifies the target and chosen reference populations, matched
topics, output schema, count, data roles and review requirements. Do not merely
enable the Grand Mean prompt unchanged: its pooled/multi-concept comparison is
different. Copying a prompt runs no workers. Put imports, paste and external
prompt options beside one another, consistent with UX-02.

**Acceptance:** prompt available for a valid designated-reference setup; exact
schema consumed by its owner; target/reference roles survive; missing choices
explain the repair beside the action. Prompt copying does not invoke a model.

## UX-08 — Gemma Scope panel has no clear purpose or actionable starting point

**Source evidence:** `GeometryPanelView` presents suite/model/release/layer/SAE
identity as `LabeledContent` (read-only metadata). Execution controls appear for
exactly one selected vector and compatible model information. The supported
analysis compares that vector with published SAE feature directions through the
Python/SAELens path. It is not a general chat-model loader or proof of a concept's
causal validity.

**Required behavior:** start with “Compare a steering vector with Gemma Scope
features” and explain why that may help interpretation. Show a clear sequence:
choose compatible model/vector → review analysis settings and execution location
→ run → inspect feature matches. Provide actionable prerequisite controls or
navigation links in place of an inert metadata panel. Distinguish automatically
resolved metadata from editable settings. Keep release/SAE IDs selectable as text
and available in Advanced details; do not present them like empty form fields.

Explain layer availability/snapping before submission. Explain that feature
alignment is an interpretation aid, not proof of the intended concept or behavior.
Preserve separate direct-feature-import and analyzed-vector conventions.

**Acceptance:** no model, incompatible model, no vector, multiple vectors,
compatible vector, missing execution dependency, local and remote requests. Each
state makes the next relevant action apparent without guessing what a field is.

## UX-09 — neutral corpora mix two purposes and conceal empty state

**Source evidence:** the panel combines selected-corpus and norm-corpus counts,
projection naming/selection, prompt constraints, data paste/import and PCA build.
`Use projection corpus` changes selection by name; it does not author rows.
The empty-corpus explanation is emitted below an action row, away from the
selection button. Placeholder text resembles a pre-existing populated dataset.

**Screenshot evidence (researcher, 2026-09-07 19:06):** this is also a concrete
layout defect, not just unfamiliar scientific language. The supplied image shows:

- “assistant-dialogue-neutral” twice, once above and once inside the projection
  name input. Example-like strings beside the other empty inputs resemble saved
  values or extra labels; the image alone does not establish their SwiftUI cause.
- Inconsistent input alignment: the projection name field begins beneath its
  example-like text, whereas the next three fields begin to its right. “Matched
  domains” text is truncated even though the form occupies a wide panel.
- A selected-corpus label repeating “Norm calibration”, followed by “Selected
  rows 200” and “Norm rows 200” without an explanation of their different scope.
  The hash occupies a prominent full row while no actual examples are visible.
- Two different prompt-copy actions, an ambiguous “Import to selected”, disabled
  “Build neutral PCs” and an “All layers (expensive)” checkbox in one action area.
  The model prerequisite appears farther below as an orange sentence instead of
  a directly actionable explanation beside the disabled build control.
- A large empty JSONL editor even though the selected corpus contains 200 rows.
  There is no clear distinction between inspecting existing data and pasting new
  data. This can make a populated corpus appear empty.

**Layout requirements:** use explicit labels above or consistently beside their
inputs; place examples as clearly marked helper text below, never as ambiguous
adjacent values. Audit SwiftUI Form/TextField label behavior rather than assuming
the renderer's cause from the image. Put corpus preview and import into distinct
states (“View 200 examples” versus “Paste data to import”), and show a replacement
review when importing would replace existing rows. Move hashes into copyable
technical details. Place each prerequisite beside the affected action with a
practical next step; for the current MLX-only PC build, explain the required
execution route without implying that a same-Mac Python/MPS model is already
loaded into that route. Offer a loading action where supported. Keep layer scope
under advanced build settings, with an explanation of its compute implications.

**Field semantics verified after the screenshot:** the example-like outside text
is the literal SwiftUI `TextField` title, not a readback of its bound value.
`projectionNeutralCorpusName` initially contains `assistant-dialogue-neutral`;
the concepts, domains and exclusions drafts initially contain empty strings.
Editing updates these bound drafts without a submit action. The three list fields
are trimmed and substituted as prose into `anthropicStyleNeutralDialoguePrompt`
when “Copy projection prompt” is clicked; they are not parsed comma-separated
settings, filters, or updates to existing corpus rows. They have no effect on
“Copy neutral corpus prompt”, which uses another template. The projection name
is used by “Use projection corpus” to select the destination separately.

The displayed examples are also NOT the empty-field defaults: a blank concepts
draft asks the downstream author to obtain the concepts from the researcher;
blank domains and exclusions expand into longer built-in lists. The app must
show the effective values, including these fallbacks, in the prompt review so
the researcher is not commissioning unseen constraints. There is no `.onSubmit`
handler on these fields or in this panel for them; the researcher's observed
Return-key behavior needs live inspection of focus/default-button routing.
Do not claim Return saved, imported, or applied constraints without observing the
actual action. Add a regression check that Return cannot silently initiate an
unrelated operation, and make “Edit prompt instructions → Preview/copy prompt →
Import authored data” explicit. Outside examples should be labelled “Example”,
never presented as the effective setting.

**Required behavior:** separate two tasks visibly:

1. **Calibrate steering strength:** choose/view/import calibration reference
   text; explain the role of its activation norms.
2. **Optional: estimate patterns to remove:** choose/create a projection corpus,
   inspect examples and counts, then explicitly run the component build.

**Placement and explanation:** remove the expanded Neutral Corpora section from
the ordinary path before Concept Vector Builder. Put corpus management in a
collapsed “Advanced: calibration and projection” area below the primary workflow,
with contextual links from operations that need it. At point of use, explain
calibration as reference text used to measure typical activation size so the
selected strength convention has a meaningful scale. If compatible measured
calibration already exists, show readiness and require no corpus-management step.
If the operation needs missing calibration, offer the required measurement with
its purpose and compute cost; do not equate this with building PCs or silently
substitute scientific inputs.

Explain “Build neutral PCs” as: “Run the selected model on reference examples to
find recurring patterns in its internal activity. Save those patterns so you can
optionally remove their contribution from a steering vector and compare the
effect.” PC means principal component: a direction capturing variation in the
reference activations, not a certified nuisance or unwanted behavior. This is an
advanced experimental adjustment, not a normal prerequisite for extraction or
steering and not an automatic quality improvement. Building saves a model-specific
basis; applying removal is a separate choice. Offer a comparison with and without
removal, since it can weaken the desired signal. Use a label such as “Build
projection basis”, accompanied by this explanation and an expandable technical
description; renaming alone is insufficient. Keep this separate from the
neutral-mean adjustment used by ablation and from norm calibration.

**Researcher-specified grouping:** put the name and three authoring-instruction
fields in an interior group labelled “Create a projection prompt”. Use ordinary
field labels, with examples in tooltips. Identify the name as the intended output
corpus name; it currently does not enter the generated prompt. Do not imply that
typing it selects or creates a populated corpus. Preview/copy is the group's
action. Existing-corpus selection and importing the resulting data are separate,
explicit controls with their current destination shown.

The current “Use projection corpus” button normalizes the entered name, resolves
its workspace corpus path, changes `selectedNeutralCorpusID`, and refreshes the
listing. If the corpus already exists it selects those rows; if not it selects a
prospective empty destination (no directory or corpus file is written by this
action). It neither generates data nor activates projection in Playground. The
selected destination is used by subsequent import and PC-building actions. Retire
the ambiguous button in favor of “Choose existing corpus” and an explicit named
new-corpus/import flow. Show “No data imported yet” for a new destination rather
than making a name-only selection look like an available dataset.

Show corpus name, purpose, path/details, example rows and empty/missing/populated
status together. Label counts “Examples in this corpus” and “Calibration examples”
with actual scope; reserve token-row counts for advanced build detail. Separate
“Select existing corpus” from “Create empty corpus.” Name-only creation must say
that data still needs to be imported or authored.

Put empty-state feedback next to the selected corpus and offer Import file, Paste
data and Copy authoring prompt. Label prompt fields in ordinary language: which
properties the examples should avoid expressing, which topics to cover and which
situations to exclude. Explain that these affect the copied prompt, not a hidden
filter or instant dataset generation. Building PCs is optional, compute-bearing
analysis; it is not a required first step for ordinary concept extraction.

**Acceptance:** researcher can explain the difference between calibration and
projection data, inspect the selected content, and predict what each button does.
Blank fields/placeholders never imply an installed corpus with rows. Selection
and data import are distinct, and errors appear where the action was taken.
Verify the rendered panel at narrow and wide supported window sizes and larger
text settings. No duplicated example labels, truncated essential guidance,
misaligned inputs or unexplained disabled actions. Include the screenshot's
populated-corpus/no-loaded-MLX-model state in visual acceptance; source inspection
and string assertions alone cannot establish that this layout is fixed.

## UX-10 — reading position and raw/chat rendering need visible explanations

**Source evidence:** `ExtractionReadingControls.swift` contains technical hover
help; `ExtractionDeclarationChoices.swift` exposes `K`, `k`, `i`, `n` as parameter
captions. Some help exists, but its terminology and discoverability do not meet
the researcher's needs.

**Required behavior:** concise help for every choice, plus a persistent short
explanation of the current choice. Replace symbolic input captions with labels
such as “Start token,” “Tokens back from end,” or “Tokens after instruction,”
including indexing/range and whether positions include template tokens. Use a
small rendered-token example/preview where available; do not infer boundaries
that the tokenizer has not supplied. Keep the public stored vocabulary unchanged.

Explain raw as “send the example text without the model's conversation wrapper.”
Explain chat template as “format it as a conversation turn using this model's
template.” Show user/assistant voice and generation-prompt choices when relevant.
Explain that rendering can change the measured activation and the meaning of
“last token”; it is not merely a display preference. Offer the method's declared
default with a reason, without silently overriding an explicit scientific choice.

**Acceptance:** users can distinguish last sequence token from last example-content
token; parameter labels identify what the number changes; raw/chat choices produce
the exact declared owner inputs and unchanged scientific semantics across surfaces.

## UX-11 — direct SAE feature import belongs in vector creation

**Researcher finding:** a feature found on Neuronpedia cannot be added through an
obvious app action. Creating a vector is hidden inside Analysis.

**Verified:** the engine already implements direct import in
`Server/steerlab_server/experiment/gemma_scope.py::import_feature_by_id`, exposed
through `steerlab-server gemmascope import-id` and
`POST /api/gemmascope/import-id`. It needs a model, SAE release and dictionary
identity, feature ID, label and compatible residual-norm calibration artifact.
The optional Neuronpedia URL records discovery provenance; it is not currently a
URL-to-feature resolver. No prior cosine analysis report is needed. The app's
`GeometryPanelView` exposes report-based imports instead; these are distinct
scientific import paths. Do not fabricate a report to satisfy the UI.

**Required behavior:** give Concept Vector Builder an “Import an SAE feature”
source alongside extraction from data and J-lens derivation. Accept a supported
Neuronpedia feature URL or explicit feature identifiers, resolve and preview the
exact model, dictionary, layer and feature before importing, and request only
missing information. Unknown URL/dictionary formats need a useful explanation
and an explicit-identifier alternative; do not promise arbitrary SAE support.

Explain required calibration as determining the scale at which steering is
applied. Reuse compatible measured calibration where available; otherwise offer
the supported calibration procedure with its compute requirements. It is not a
request for concept training data or a claim of similarity to another vector.
Keep report-based import and direct import normalization conventions distinct.

Save the result to the ordinary vector library with model/revision, layer/site,
dictionary and feature identity, import convention, source URL and artifact
identity. Show the feature's description as a hypothesis to test, not a proven
behavioral intervention. Analysis remains the place to inspect feature alignment
and results, with a link into this shared creation flow.

**Acceptance:** a researcher can start with a supported feature URL or explicit
IDs, inspect the resolved feature, import it without a pre-existing cosine
report, locate the resulting vector and add it to a compatible configured agent.
App and agent paths invoke the same import owner. Test unknown mappings, model
mismatch, missing calibration, duplicate requests and failed downloads without
false success or silent substitution.

## UX-12 — J-lens success must lead to the actual vector

**Researcher finding:** a direction labelled “hello” was reportedly added, but
cannot be found in Data, Playground or Analysis. Token lookup found candidates
for the intended Gemma 3 4B model on MPS.

**Verified source defect:** `JSpacePanelSection.run(.derive)` waits for a
successful job, announces that the vector appears in the ordinary catalog, then
calls `refresh()`. That refresh fetches the lens catalog only. Its shared
`ChatService.followServerJobInActivity` helper also does not refresh vectors.
The equivalent successful build in `ConceptBuilder.buildVectorOnActiveServer`
does explicitly call `catalog.refreshRemoteVectors()`. Thus the Analysis path
can announce availability while leaving the app's vector inventory stale.

This is a confirmed missing refresh, not yet a diagnosis of the specific saved
artifact: this investigation has not inspected that workspace's job result,
artifact files or installed-app identity. Token lookup alone does not prove
derivation or publication succeeded. Inventory filters and the selected compute
workspace may also explain invisibility.

**Required behavior:** after successful derivation, resolve the returned artifact
identity against the originating workspace's vector catalog, refresh all consumers
and show a result card with its actual name, exact token (including leading
space/case), model, layers, backend/location and “Show in vector library” action.
Offer adding it to a compatible agent without silently enabling steering.
If indexing is delayed or fails, distinguish “vector saved; library refresh
failed” from a failed derivation and offer refresh without re-deriving it.
Capture the originating workspace/model before awaiting a job; switching tabs,
models or workspaces must not publish success against the wrong inventory.

**Existing builder discoverability:** J-lens derivation already has a
`jlensTokenDirection` family and rows inside Concept Vector Builder. Its recipe
selector lives higher up in Dataset Builder and filters server-only families by
compute target. A technique needing no dataset is therefore selected through a
dataset control. Move source selection into the vector-creation flow and share
the implementation with Analysis instead of maintaining two completion paths.

**Backend language:** the app's “Local” means its MLX route; a Python engine can
run on the same Mac using MPS while appearing under the server route. Do not tell
a researcher that MPS requires another physical machine. Conversely, finding
token candidates or sharing the Gemma 3 4B family name does not establish that an
artifact is interchangeable with an MLX/quantized model. Check the actual model,
revision, conventions and supported consumer path. Treat qualification gaps as
recorded limitations rather than blanket refusals; preserve concrete integrity
and execution requirements. Some existing source comments make blanket claims
about local derivation being meaningless: replace those with precise supported
backend statements rather than adopting them as scientific conclusions.

**Acceptance:** derive through each entry point, then immediately find the exact
artifact in Data and compatible agent/Playground selectors without restarting.
Exercise a same-Mac Python/MPS workspace, delayed completion, refresh failure,
model/workspace switching and a duplicate label with different token IDs. Verify
that lookup, submission, derivation, library availability and steering activation
are separately and truthfully reported.

## Unified vector-creation entry point

Present Concept Vector Builder as “Add a concept vector”, with choices for
extracting from data, deriving from a J-lens token, importing an SAE feature and
importing an existing supported vector artifact. All end in the same library,
retaining their distinct provenance and scientific meanings. Do not equate an
SAE feature, a token direction and an empirically validated concept merely because
they share the library.

Use the selected model and execution backend to show compatible sources first.
An available resource, a resource requiring acquisition and an unsupported
combination are different states. Explain unavailable choices briefly so users
can discover how to access them; do not make everything disappear without a
reason. Model compatibility includes revision, representation/site and artifact
conventions, not just a friendly model-family name.

Add UX-12's missing refresh to the immediate functional fixes. Implement UX-11
and the unified creation flow with UX-08's Gemma Scope explanation work. Extend
the live release journey to include one SAE import and one J-lens derivation,
followed by finding and using each resulting artifact.

## UX-13 — Playground vector controls must not require a distant unlock switch

**Researcher decision:** remove “Inject vectors” from Steering Controls. Name the
vector area “Steering Vectors” and control individual vectors there.

**Verified:** `ChatView.injectionMasterControls` binds that master toggle to
`service.steeringEnabled`. Each slot section is disabled/dimmed when it is false,
and “Add Vector” is disabled too. Both local and server slot sections already
contain “Include this vector” toggles bound to `slot.enabled`, but the ancestor's
disabled state prevents the researcher from using them. This is a real extra
unlock step, not merely an unclear label.

**Required behavior:** keep vector selection, editing, adding and on/off controls
usable inside “Steering Vectors”, subject only to relevant availability or
in-flight execution restrictions with nearby reasons. Per-vector switches are
the visible authority for what will apply to the next message. All switches off
means no vector intervention; it does not disable an independently selected
adapter and must not be labelled an unmodified baseline if an adapter is active.
Show a compact active-vector count/status in the section. Do not replace the
distant unlock with another mandatory section-level unlock.

Removing the UI toggle requires reconciling the runtime's master flag with these
semantics on both backends and the agent-control surface. Do not simply force the
flag true and reactivate previously muted configured slots. Loading a saved
agent, restoring a session, turning a vector off/on, and saving the current agent
must preserve the effective intervention and accurately capture which vectors
are enabled. Adding/browsing a vector must not silently enable existing disabled
ones. Changes apply to subsequent generation; do not mutate a running request or
rewrite saved study/run evidence.

**Acceptance:** from an initially unsteered session, scroll directly to Steering
Vectors, choose/configure a vector and turn it on without visiting another
section. Turn it off while retaining editable settings. Test multiple vectors,
all off, adapter-only configuration, saved-agent loading and session restoration
on local and server paths, verifying actual generation declarations and saved
agent state rather than only toggle appearance. Add this to the ordinary workflow
fixes and the live researcher journey.

## UX-14 — Concept Vector Builder must own an editable method selector

**Researcher finding:** after restarting, the builder displays “J-lens token
direction” in a read-only Recipe row with no way to choose another method there.

**Verified:** the builder renders `LabeledContent("Recipe", value:
builder.recipeFamily.label)`. The actual `recipeFamilyPicker` is inside Dataset
Builder; help tells the researcher to change it above. This explains the
read-only appearance, although the mechanism restoring the reported selection
has not been diagnosed. A restart is not a suitable repair.

**Required behavior:** put the editable vector-source/method control at the top
of Concept Vector Builder, sharing one owner with any dataset-specific view. The
selected method determines whether the next step is choosing a concept and its
data, choosing a J-lens token, or importing an SAE feature. Do not force users
through Dataset Builder to escape a method that uses no dataset. Changing method
must preserve unrelated drafts and not start compute, generate data or reinterpret
one method's rows as another method's dataset. Restore a valid, editable selection
on launch; explain a restored unavailable method and offer alternatives in place.

**Acceptance:** restart with J-lens selected, enter Concept Vector Builder and
switch directly to an extraction method, select the intended concept, inspect the
required inputs and switch back without losing drafts. Repeat across backend and
model changes. Check the rendered selector in the panel actually shown to the
researcher, not only in a separate Dataset Builder view. Treat this as an ordinary
workflow fix alongside the disabled Playground vector controls.

## UX-15 — wire existing server projection support into the app

**Correction to the investigation:** the Playground source comment says “no
server neutral-basis catalog yet”, and the initial conversational explanation
repeated it. That claim is false at the inspected main. Do not use this stale
comment to scope the fix as building a new server subsystem.

Verified server owners:

- `GET /api/neutral/corpora` returns both corpora and `neutral.list_bases()`.
- `POST /api/neutral-pcs/build` builds a saved basis through a durable job.
- `experiment/neutral.py` saves, loads and lists server
  `neutral-pc-basis.json` artifacts.
- `experiment/model_variant.py::variant_injections` resolves an agent's
  `neutralPCBasisPath` and projects the per-layer injection vector through
  `vector_math.projecting_out` before determining injection scaling.

`ChatView.neutralDirectionControls` nevertheless disables the removal toggle and
picker for server workspaces; `ClusterClient` has no corresponding neutral-basis
catalog/build client methods in the inspected source. The app can preserve a
seeded agent's basis pin, but it does not provide the equivalent interactive
selection workflow. This is a missing app integration, not scientific or hardware
unavailability. Correct source comments and availability messaging together.

**Required behavior:** expose the existing server catalog and build job through
the client, select compatible server bases in Playground, and compose/preserve
the selected basis reference through agent save/load and generation. Keep corpus
authorship local and transfer inputs through the supported execution workflow;
having an old import route is not permission to bypass the workbench/runner
authoring boundary. Handle remote artifact custody and hashes explicitly.

Do not assume byte/schema equivalence: Swift writes `neutral-pcs.json` with a
layer list and nested arrays; Python writes `neutral-pc-basis.json` with layer
keys. The Swift build defaults to a middle-third capture band while the current
server build route passes no layer-selection argument. Reconcile or explicitly
expose these semantics and test matched settings. Simply enabling the existing
Swift picker or reusing its decoder will not complete the workflow.

**Scientific framing and qualification:** principal components identify variation
in the selected reference activations, not semantically certified “neutral” or
unwanted directions. Removing them is a plausible optional intervention worth
testing; no generic behavioral benefit follows from the PCA construction. Compare
baseline, original steering and projected steering on development and separate
held-out data, including preserved capabilities and comparable strength
conventions. Record removed norm, layers and component count; distinguish a
directional improvement from simply weakening steering. Keep neutral-mean
ablation centering and extraction-time PCA settings separate from this saved
runtime basis operation.

**Acceptance:** same researcher journey on app-local and connected Python
backends: inspect corpus, build, follow job, locate basis, select removal, run,
disable, save agent and reopen it. Assert actual injection inputs change as
declared and artifact identity survives the round trip. Verify matched numerical
projection fixtures, then appropriately scoped live backend measurements; broad
CPU/MPS/CUDA residual and OptVec probes do not by themselves qualify this workflow
or prove that removal improves research outcomes. Until the journey passes,
surface parity for neutral projection must be reported as incomplete.

## UX-16 — OptVec empty state offers no way to create a vector

**Screenshot evidence (researcher, 2026-09-07 19:29):** Data → OptVec shows empty
Dataset bundles and Campaigns & results sections, followed by an attachment form
requiring an already trained artifact. Its instructions lead with nine hashed
files, folder paths, commands and scheduler details. There is no visible “Train
a vector” or data-preparation entry point. The adjacent Activity pane shows an
unrelated J-lens technique explanation and raw server startup logs. Do not copy
the private workspace/study identifiers visible in the screenshot into public
source, fixtures or commits.

**Verified:** `OptVecPanelView` describes itself as a read-only v1 inventory/results
surface plus attachment. It has no training start action. Shared `optvec-train`
and `optvec-eval` interviews exist in `workflows.json`; `ScienceGuidesView` opens
their `MethodAuthoringSheet` through “Author request…”, followed by execution and
evidence controls. This existing path is disconnected from the task's obvious
home. The optimization guide also points at Agents / Optimizations, whose
`OptimizationComposerView` is a parameter sweep over an existing vector; it must
not be confused with OptVec's gradient training of a vector. The separately named
“Scientific diagnostic…” sheet only offers battery/stability and is not an OptVec
training route.

**Required behavior:** start the tab with a plain explanation: “Train a steering
vector to encourage a chosen behavior while checking the judgments and abilities
you want to preserve.” Provide “Train a vector” and “Import an existing result”.
The training flow asks what should change, what should stay stable, which model
and execution target to use, and how the researcher wants to supply the data.
Offer file/paste import, a copyable authoring prompt, or explicitly commissioned
generation; do not infer permission to produce nine datasets or launch workers.
Explain training, selection and final evaluation roles in context. A campaign
bundle is one supported organization, not proof that every training configuration
requires precisely nine files; requirements come from the selected owner config.

Reuse the maintained scientific interview and execution owners, with current
model/workspace context prefilled and inspected identities resolved. Review the
training objective, strength convention, relevant constraints and compute plan
before execution. After training, link directly to held-out evaluation and the
saved vector in the ordinary library, then “Add to agent” and “Use in a study”.
Do not require a pre-existing draft study merely to train and inspect a vector.
Keep advanced campaign/geometry tools available after the basic journey is clear.
Add “Train with OptVec” to the unified vector-source chooser; it is distinct from
stimulus-based extraction, J-lens derivation, SAE import and sweeping an existing
vector's settings.

Replace the static orange “Only the loadings that hold across 2–3 seeds are
interpretable” footer with proportionate guidance at the relevant interpretation
step: “Different training runs can learn different directions. Repeat training
to assess stability before claiming a particular internal direction is
reproducible.” Two or three seeds are not a universal certificate; agreement is
not by itself causal or semantic validation. Routine stability guidance is not an
error and should not dominate an empty setup screen. Technical checks should
remain available under details, with actionable plain-language summaries.

**Acceptance:** from a workspace without OptVec bundles or results, enter this
tab, understand the purpose, supply approved inputs, review/train on the selected
engine, follow progress, inspect/evaluate the result and find the vector for an
agent without hunting through documentation or unrelated tabs. The adjacent
viewer explains OptVec or shows this task's progress; workspace history remains
available explicitly. Test cancellation/failure/restart and verified remote
result import. Reuse the same scientific configuration and artifact identity
across app and agent entry points. Include this as a missing surface-parity
journey, not a request to implement a new training algorithm.

## UX-17 — Adapter Trainer exposes optional bookkeeping as a setup requirement

**Researcher finding:** Training data and Validation data need explanations of
what to supply and why. “Training workspace” looks disabled while instructing the
researcher to choose a folder. “Adapter output” has no understandable meaning.

**Verified:** `FineTuningPanelView` already supplies short `.help` strings for
data rows, but the presentation leads with directories rather than research
roles. `DirectoryPathRow` disables its Reveal-in-Finder label when the path is
empty; the gray “Choose training workspace folder” is placeholder text, not a
required-field validation. The separate Choose action is rendered independently.
`trainingWorkspacePath` starts empty and is saved/restored as optional adapter
metadata. Source references show no trainer consuming it as an input directory;
it is intended to record raw source material/templates, not choose the active
SteerLab workspace or the execution machine. Creating an adapter already creates
its home and training/validation directories and leaves this field nil.

**Required behavior:** remove Training workspace from ordinary setup. If retained,
rename it “Source materials folder (optional)” under project details, explicitly
stating that its contents are not automatically used for training. Do not require
the researcher to choose an additional workspace. Default project/storage paths
to the active workspace and keep overrides in advanced storage details.

**Confirmed main-form decision:** among the existing file/location rows, show
only **Training data** and **Validation data** in the main form. Move **Project**,
**Adapter output** (renamed “Saved adapter files”), and the optional source-materials
folder into one collapsed **Storage details** section. Do not require users to
choose these internal locations during ordinary setup; use workspace-managed
defaults. Each visible data row should show the selected dataset and example
count, **Choose…** and **Preview** actions, a short purpose explanation and
format examples in help. This decision concerns location rows; model selection
and relevant training controls remain part of the training workflow.

Rename Adapter output to a clear saved-artifact location such as “Saved adapter
files”, with state distinguishing a planned output from a completed/imported
adapter. Explain that the adapter is the learned model-specific modification
loaded alongside the base model; the location holds weights/configuration, not
training examples or generated responses. Existing local MLX artifact help names
`adapters.safetensors` and `adapter_config.json`; keep backend-specific filenames
in technical details. Inspect the actual completion/import receipt for the
result location rather than assuming the editable project path is the remote
job's storage location. Preserve fresh-run and artifact-identity guarantees.

Explain data roles visibly and in tooltips, adapting to the selected training mode:

- Training: examples used to change the adapter. Document adaptation uses reference
  documents; instruction/chat tuning uses structured prompts/conversations with
  desired assistant responses. Offer a valid example and format-specific import
  guidance from the maintained schema, not a generic list suggesting PDFs work
  as chat examples.
- Validation: separate representative examples, in the same supported format,
  used to monitor performance during training without gradient updates from those
  examples. Avoid duplicate or near-duplicate source material across splits. If
  validation informs settings/checkpoint choices, reserve separate test data for
  the final research claim. Explain the actually supported selection behavior;
  do not promise automatic early stopping or best-checkpoint selection generically.

Offer file/folder selection, preview, counts and source/split checks. Empty folders
created by the app are not populated datasets. Provide copyable authoring prompts
and explicitly approved generation as alternatives; do not silently author data,
split the researcher's corpus or choose an authoring model.

**Acceptance:** create an adapter in a fresh workspace without choosing extra
storage paths, understand and supply each data role, preview its decoded format,
review/train, locate the completed adapter and add it to an agent. Verify both
training modes and connected-server custody. Optional source metadata must not
appear as a blocker; gray Reveal labels must not masquerade as disabled required
inputs. Check runtime consumption agrees with the selected training/validation
paths and that completed output status matches the actual artifact.

## Review and live acceptance

Use a new branch from the agreed current main and small reviewable commits for
runtime, agent-contract and presentation changes. Preserve external stale-write
preconditions, immutable evidence, public vocabulary hygiene and stored identifiers.
Regenerate agent/interview/client resources with the maintained generators, rebuild
the matched Mac/client payload, and run both suites sequentially under Xcode beta
with external scratch. Apply AST audits only to genuine mechanical moves; do not
move their baselines to conceal intentional numerical changes.

The release check is a real beginner journey: choose a model and construct,
discuss an agent comparison, choose a data route, import or commission only the
agreed data, explain validation and capability checks, extract, explore settings,
save the configured agent, create a study from a template and inspect its results.
Run it through the app and a fresh assistant. Include tab switching, cancel,
empty corpora, unsupported cache assets and large JSONL results.

Success means the user knows what object they are changing, why each dataset is
needed, what computation is about to run and what a failure means. Passing a unit
suite or adding hover text alone does not establish that outcome.
