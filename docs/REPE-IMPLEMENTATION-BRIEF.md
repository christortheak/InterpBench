# RepE implementation brief

**What this document is.** The specification of what SteerLab has ACTUALLY
implemented from Zou et al., *Representation Engineering: A Top-Down Approach
to AI Transparency* (arXiv:2310.01405) and its reference implementation
(`github.com/andyzoujm/representation-engineering`) — the pipeline, the file
schemas, and an itemised faithful-vs-departure table. It is written to be read
beside the code, and the code's doc comments cite it by section number.

**Why it exists.** For most of 2026 this file was cited from a dozen places
and did not exist. Meanwhile the codebase carried a family called `repeLAT`
that implemented the paper's PCA arithmetic and none of its pipeline, and a
comment attributing OUR normalization to "RepE Appendix C.1". A brief that
names what is faithful, what is ours, and what is still missing is the only
thing that keeps those two apart. Everything below describes code as of
2026-08-27; when the code changes, this changes with it.

**Two things live here, and they are not the same thing.**

| | `pairedDifferencePCA` (sidecar `recipeMethod: "repeLAT"`, manifest method `"lat"`) | `repeReaderLAT` (`RepEReader` / `repe_reader`) |
|---|---|---|
| What it is | A steering VECTOR: PC1 of normalized pair differences, norm-matched to the mean difference | A fitted measurement INSTRUMENT: template + LAT token + PCA + persisted probe |
| Template | none — raw stimulus text | a hashed registry template (§2) |
| Sign | train-label majority (§5) | held-out split, with a stamped fallback (§5) |
| Layer | declared by the study | declared by the study; the fit stamps a RECOMMENDATION (§5) |
| Fit parameters | not persisted | persisted, and reused at inference (§4) |
| Honest label | "Paired-difference PCA (RepE-inspired)" | "RepE reader LAT" |

Everything from §1 onward is about the reader unless it says otherwise.

---

## §1 — Pipeline

```
stimulus
  → render the task template               (§2)
  → apply the declared extraction rendering (§7)
  → capture the hidden state at the LAT token, every layer
  → per-pair differences                    (§3, two constructions)
  → mean-centred PCA, n_components = 1
  → PC1 signed on the HELD-OUT split        (§5)
  → ScalarProbe fitted on the TRAIN activations
  → one reader artifact per layer           (§4)
```

Inference (`scoreTexts` / `score_texts`) renders the *same* template under the
*same* rendering, captures the *same* token position, and projects through the
stored probe — the paper's "normalize test activations with the training
parameters", not a cosine-to-vector shortcut.

Implementations: `Sources/SteeringKit/Extraction/RepEReader.swift` and
`Server/steerlab_server/steering/repe_reader.py`. The Python module is the
schema/math source of truth; the Swift one is its twin, and the two are held
together by identical JSON key names, twin refusal literals, and the committed
cross-engine PCA fixture (`Tests/Fixtures/cross-engine/paired-difference-pca.json`).

---

## §2 — Task templates (`prompts/templates/<id>.json`)

The registry is **one file per id, and the id must equal the filename**.
`hash` is the SHA-256 of the file's raw bytes, so changing a template changes
every artifact fitted through it.

```json
{
  "id": "instructed-stance-pair-v1",
  "conceptSlot": false,
  "text": "{{instruction}}\nScenario: {{stimulus}}\nThe described state is",
  "latToken": "final",
  "instructionPair": {
    "experimental": "…T+…",
    "reference": "…T−…"
  },
  "divergence": "synthetic-neutral-instruction-pair — …"
}
```

| Field | Meaning |
|---|---|
| `id` | Registry id; must equal the filename stem. |
| `text` | The scaffold. Must contain `{{stimulus}}`. |
| `conceptSlot` | `true` when the text contains `{{concept}}`. A `{{concept}}` slot with `conceptSlot: false` is refused, and vice versa. |
| `latToken` | Only `"final"` is implemented; any other value is REFUSED, never coerced. |
| `instructionPair` | **Optional, added 2026-08-27.** The paper's T+/T− pair (§3). Present ⟺ the text contains `{{instruction}}`; either one alone is refused. Both strings must be non-empty and different. |
| `divergence` | Free text marking a deliberate departure (the unnamed clean-room scaffold; the custom unregistered template; the synthetic instruction pair). |
| `hash` | Not in the file — computed at load and carried in embedded copies. |

Shipped templates:

* `amount-in-scenario-v1` — the paper's construction, naming the concept.
* `unnamed-scenario-v1` — a clean-room scaffold that never names the concept
  (stamped `divergence: "unnamed-clean-room"`).
* `instructed-stance-pair-v1` — a T+/T− pair whose instructions never name the
  concept. The paper's own instructions do ("pretend you're an honest
  person…"); a study that wants that verbatim writes its own registry file and
  accepts the divergence deliberately.

An unregistered one-off template is persisted into the run directory and
hashed from its bytes exactly like a registry file, stamped
`divergence: "custom-unregistered"`.

---

## §3 — Reader datasets (`prompts/readers/<concept>/pairs.jsonl`)

One JSON object per line. Every row carries `concept` and `templateID`; rows
must share ONE concept and ONE shape. `split` defaults to `"train"`, and any
other value is held out.

**Content-pair shape** (`supervisedContent`, and the only shape before
2026-08-27):

```json
{"id":"fear-pair-0","concept":"fear","positiveStimulus":"…","negativeStimulus":"…","topic":"t","split":"train","templateID":"unnamed-scenario-v1"}
```

Two DIFFERENT stimuli under ONE template. The difference is
H(positive) − H(negative).

**Single-stimulus shape** (`unsupervisedTemplatePair`, the paper's §3.1 step
1b):

```json
{"id":"fear-row-0","concept":"fear","stimulus":"…","split":"train","templateID":"instructed-stance-pair-v1"}
```

ONE stimulus under TWO templates. The difference is H(T+) − H(T−): the
stimulus is held fixed and the INSTRUCTION carries the contrast, so the
direction cannot be a content artifact of two different texts. A row that
declares both shapes is refused; so is a file that mixes them.

The contrast mode is DERIVED from (dataset shape × template), never passed
loosely — `resolveContrastMode` / `resolve_contrast_mode`. A template pair fed
content pairs is refused (the second stimulus would be a confound); a
single-stimulus dataset fed a single template is refused (there is nothing to
contrast against).

**Not to be confused with** `prompts/repe/<concept>/pairs.jsonl`, which is the
`pairedDifferencePCA` family's mirror and has a different row shape
(`{"positive": …, "negative": …}`), read by `StimulusSet.loadPairs`.

---

## §4 — Reader artifacts (`repe-reader-lat`, schema 2)

One artifact per concept × layer × template × model × substrate. The full
template record is embedded so inference is standalone and drift-proof;
`templateID`/`templateHash` remain the registry pins. Both engines write the
same keys.

| Key | Meaning |
|---|---|
| `artifactType` | `"repe-reader-lat"`. A different value is refused at decode. |
| `schemaVersion` | `2` since 2026-08-27. Schema 1 still decodes (§10). |
| `modelID`, `revision`, `substrate` | The binding. A reader is per-model, per-revision and per-substrate; scoring and verify refuse a foreign one. |
| `concept`, `layer` | What it measures, and where. |
| `templateID`, `templateHash`, `template`, `templateDivergence` | The registry pin plus the embedded copy. |
| `datasetHash` | SHA-256 of `pairs.jsonl`'s raw bytes. |
| `latTokenPosition`, `readingPosition` | `"final"` and the resolved position label. |
| `probe` | The `ScalarProbe`: `direction`, `activationCenter`, `projectionCenter`, `projectionScale`, `orientation`, `positiveMean`, `negativeMean`. |
| `pc1ExplainedVarianceOfDifferences` | PC1's share of the DIFFERENCE CLOUD's variance. **Absent** when the cloud has none to apportion. |
| `pc1ExplainedVarianceBasis` | `"differenceCloud"`, `"degenerateDifferenceCloud"` (value absent), or `"alternatedRows"` (a schema-1 artifact's legacy number — §10). |
| `trainAccuracy`, `heldOutAccuracy` | Probe classification accuracy on each split. |
| `trainPairCount`, `heldOutPairCount` | Row counts. |
| `contrastMode` | `"supervisedContent"` \| `"unsupervisedTemplatePair"` (§3). |
| `signConvention` | `"heldOutPairAgreement"` \| `"trainMajority"` (§5). |
| `signHeldOutAccuracy` | Held-out paired-discrimination accuracy of the CHOSEN sign. Absent under `trainMajority`. |
| `signFallbackReason` | Why the held-out rule stood down, when it did (§5). |
| `orientationSeed` | The seed for the unsupervised orientation draw. Absent under `supervisedContent`. |
| `recommendedLayer`, `recommendedLayerAccuracy`, `layerRecommendationBasis`, `layerRecommendationNote` | The set's argmax layer, as a RECOMMENDATION (§5). |
| `extractionRendering` | The declared rendering block. **Absent = raw** (§7). |
| `renderingConvention` | The prose convention that matches the rendering. |
| `extractionDate` | ISO-8601. |

---

## §5 — Sign and layer selection (the paper's step 4)

**Sign.** The reference implementation's `get_signs` decides from the TRAIN
labels. The paper's TEXT describes choosing sign and layer on a held-out set
(e.g. 25 ARC-Challenge validation examples). We implement the paper.

For each layer: take the held-out rows' differences (positive − negative, or
T+ − T−), project them onto the unsigned PC1, and choose the sign under which
they project positively. That IS held-out classification accuracy on the
paired discrimination task, maximized over the two available signs.

Why it cannot come from the probe's own accuracy: `scalarProbe` derives
`orientation` from the train class means, so flipping the direction flips the
orientation too and leaves every score identical. Probe accuracy is
sign-invariant by construction; the paired projection is not.

**The fallback, and it is loud.** Below `minimumHeldOutPairsForSignSelection`
(= 2) decided pairs, or on an exact tie, the sign falls back to train-label
majority and the artifact stamps `signConvention: "trainMajority"` plus a
`signFallbackReason` naming which of the three cases occurred — no held-out
rows at all, too few off-zero projections, or an even split. A one-pair vote is
a coin flip wearing a validation split's authority.

**Layer.** The fit stamps `recommendedLayer` — the argmax of held-out accuracy
over the layers fitted together (ties to the lower index), or of train accuracy
when there are no held-out rows — into EVERY artifact of the set, with
`layerRecommendationNote` saying in the artifact's own words that it is a
recommendation. **Nothing selects a layer at use time.** Which layer a study
reads is a declarable choice recorded in its manifest, because the layer is a
scientific decision and an instrument that quietly picks its own is not
reproducible from the manifest alone.

**`pairedDifferencePCA` has no held-out split**, so its sign stays train-label
majority — and its sidecar stamps `signConvention: "trainMajority"` so the
difference is visible on the artifact rather than being a fact about which code
path produced it.

---

## §6 — Deriving a steering vector from a reader

`deriveSteeringVector` / `derive_steering_vector` is an EXPLICIT,
provenance-stamped conversion, because "we steered with a RepE reader
direction" is a different claim from "we reproduced RepE control".

The reading direction goes at the reader's layer, zeros below (the Gemma-Scope
import convention). The sidecar stamps `extractionMethod: "repeReaderLAT"`,
`recipeMethod: "repeReaderLAT"`, `source: "repe-reader-lat"`, `readerID`,
`readerHash`, `controlMode: "reading-vector activation addition"`, and the
reader's own `readerLayer` / `readerTemplateID` / `readerTemplateHash` /
`readerContrastMode` / `readerSignConvention`.

**Whose sign the bytes carry — and it depends on the reader's
`signConvention`.** `ScalarProbe.score` is `orientation · (a·direction −
centre)`, so a stored `direction` points at "more concept" only when
`orientation == +1`. When train-label majority signed every direction, applying
the orientation was simply restating that verdict in the one place a steering
vector can hold a sign — without it, a reader whose PC1 came out anti-aligned
injected the concept BACKWARDS while every provenance stamp said forwards.
Held-out sign selection then changed who decides: under
`signConvention: "heldOutPairAgreement"` the fitted `direction` ALREADY carries
the held-out-chosen sign, while `probe.orientation` still comes from the train
class means — so applying it when the two splits disagree re-flips the vector
to the direction held-out rejected. The conversion is therefore
convention-aware:

* **`heldOutPairAgreement`** — the fitted direction is authoritative and ships
  unflipped. A train/held-out disagreement (`orientation == −1`) is stamped as
  `trainHeldOutSignDisagreement: true` rather than discarded, because "the
  training labels would have signed this the other way" is a fact about the
  direction's stability. Agreement stamps `false`; the field is present either
  way whenever held-out did the signing.
* **`trainMajority`**, and every legacy artifact whose absent `signConvention`
  reads as train-majority — the orientation is applied to the bytes, and
  `trainHeldOutSignDisagreement` is absent because held-out never voted.

`readerProbeOrientation` records the orientation under both conventions, so
what the conversion did is recoverable from the artifact alone.

**Held-out accuracy is untouched by any of this.** `heldOutAccuracy`,
`trainAccuracy`, `signHeldOutAccuracy` and the layer recommendation built on
them are all invariant to the direction's sign: `scalar_probe` derives
`orientation`, `projectionCenter` and `projectionScale` from the same direction
it is handed, so flipping the direction flips the orientation and the centre
together and every score comes out identical. A sign disagreement shows up as a
sub-chance `heldOutAccuracy`, which is the honest signal, not a corrupted one.

**Attaching one.** `repeReaderLAT` is in both engines' `ExtractionMethod`
vocabulary, so a derived vector attaches as a `pinnedArtifact` whose SOURCE
method resolves. It has no source CONCEPT: its stimuli are the reader's dataset
(whose SHA-256 is the `stimulusSetHash`), there is no
`prompts/concepts/<c>/` pair set, and its held-out evidence is the reader
artifact's own accuracy, not a `validation.jsonl`. Every data-side lifecycle
branch therefore skips rather than inventing an obligation the direction cannot
meet. A reader-derived vector is born without residual norms, so attaching one
before `backfill-norms` refuses with a repair that names the verb.

---

## §7 — Rendering (the repo's `user_tag` / `assistant_tag` analogue)

A reader fit takes a declarable `extractionRendering`, the same declaration the
extraction path uses.

* **`raw` (default, and absent means this).** The rendered scaffold IS the
  whole token sequence: no chat template, no system role, no family thinking
  suffix, tokenized with the tokenizer's defaults — single BOS, LAT token =
  the scaffold's final token. Byte-identical to every reader fitted before the
  option existed.
* **`chatTemplate`.** The scaffold becomes the USER TURN's content and the
  family template supplies every special token, so the final token is the
  generation prompt's tail — exactly what the reference implementation's
  `rep_token=-1` reads in `f"{user_tag} {instruction} {scenario}
  {assistant_tag}"`. This is the more faithful of the two.

The **marker guard's reason follows the rendering.** Under `raw`, an embedded
`<bos>`/`<start_of_turn>`/… is the double-BOS / hand-tokenized-template hazard.
Under `chatTemplate` that rationale does not apply — the template emits those
markers legitimately — so what is refused instead is turn-boundary FORGERY
inside the turn's content, which would split the scaffold and move the LAT
token off the generation prompt. The manual-`<s>` check is raw-only: under a
template, a leading `<s>` in content is ordinary text.

Swift's assistant-voice refusal stands as it is: MLXLMCommon exposes only the
generation-prompt form of `applyChatTemplate`, so a completed assistant turn
cannot be rendered on that engine and the declaration is refused with a repair
naming the engine that can.

---

## §8 — The Concept Lab path

Mental model: **concept data → reader artifact → optional steering variant** —
never "positive/negative text → vector".

`ConceptBuilder` writes `prompts/readers/<name>/pairs.jsonl` (the git-versioned
recipe truth shared by both engines) through ONE row encoder that also produces
the server route's inline payload, so the dataset hash is identical on both
substrates. The local path fits through `RepEReader.fit`; the server path
queues `POST /api/reader/fit` as a durable job. Reader artifacts land in the
fitting engine's own `runs/` and never merge across substrates.

The fit-score grid shows, per layer: train accuracy, held-out accuracy, PC1's
share of the difference cloud, and WHICH rule fixed that layer's sign. The
recommended layer is marked `★` and comes off the artifacts' own stamp rather
than being re-derived in the pane.

---

## §9 — Faithful, and where it departs

Updated 2026-08-27, after implementing the ruling. "Faithful" means the paper
or the reference implementation does it and so do we.

### Faithful

| Element | Where |
|---|---|
| Task template around each stimulus (§3.1) | §2 |
| LAT token = the rendered scaffold's final token (`rep_token=-1`) | §1 |
| PCA over paired differences, mean-centring only, `n_components=1` (`PCARepReader`) | §1 |
| The `[::2] − [1::2]` difference construction over per-pair randomized orientation | §3, §5 |
| BOTH contrast constructions: supervised content, and the T+/T− instruction pair (§3.1 step 1b) | §3 |
| Held-out selection of SIGN and LAYER (the paper's step 4) | §5 |
| Inference normalizes new activations with the persisted TRAIN parameters | §1, §4 |
| Either rendering: raw scaffold, or the `user_tag`/`assistant_tag` chat template | §7 |

### Departures, each deliberate and each stamped

| Departure | Why | Stamped as |
|---|---|---|
| The orientation draw is SEEDED | The reference implementation's `random.shuffle` is unseeded, so its direction cannot be reproduced. We keep the ± symmetry and drop the irreproducibility. | `orientationSeed` |
| `latToken` supports only `"final"` | The schema carries the field so a second position is a data change; anything else is refused rather than coerced. | refusal, not a stamp |
| Scoring is a calibrated scalar probe, not the paper's logistic / `pca_model.transform` | The centre and scale come from the train projections and ARE persisted (which is the paper's rule); the classifier on top is our midpoint rule. | `probe` |
| Reader artifacts are substrate-specific | Activations do not transfer between MLX and PyTorch. The paper has one substrate. | `substrate`, and the scoring/verify refusals |
| Datasets are authored, not borrowed | The paper fits on TQA/ARC; a study here fits on its own pinned pairs. | `datasetHash` |
| The shipped T+/T− example never names the concept | The paper's instructions do. An instruction naming the concept makes the reader a concept-word detector. | template `divergence` |
| The `recommendedLayer` is a recommendation, not a selection | The layer is a scientific decision; an instrument that picks its own is not reproducible from the manifest. | `layerRecommendationNote` |
| Derived steering vectors are NOT the paper's control experiments | The paper's control uses strided layer bands and LoRRA; we add a reading vector at one layer. | `controlMode` |

### Departures in `pairedDifferencePCA` (the OTHER family)

| Departure | Why |
|---|---|
| Per-pair L2 normalization before PCA — **ours, not the paper's** | So high-norm pairs cannot dominate PC1; without it PCA is pulled toward the mean difference, eroding the method comparison the family exists to make. Corrected attribution 2026-08-27: `PCARepReader` mean-centres and never normalizes. |
| Norm-matching the unit PC to `‖meanDiff‖` — ours | So injection α means roughly the same under either method. |
| Train-label majority sign | It has no held-out split. Stamped `signConvention: "trainMajority"`. |
| No template, no persisted fit parameters, no exact inference | It is a steering vector, not an instrument. This is why it is not called RepE. |

### Not implemented at all

* **LoRRA / RepE control training.** The paper's control half (low-rank
  representation adaptation) has no implementation here. Fine-tuning in
  SteerLab is ordinary LoRA against text, not against a reading direction.
* **Strided multi-layer control bands** (`range(8, 32, 3)`). Bands exist as N
  injectors, but nothing here reproduces the paper's control protocol.
* **Prompt-span steering.** We inject at the final prompt token and every
  generated token; the paper also steers prompt spans.
* **Contrast-vector reading (`rep_reading_pipeline`'s cluster/PCA variants
  beyond PC1).** Only `n_components=1` is used.

---

## §10 — Legacy notes

**The raw value `"lat"` never changes.** The symbol was renamed
`lat` → `pairedDifferencePCA` (Swift) / `LAT` → `PAIRED_DIFFERENCE_PCA`
(Python) and the label to "Paired-difference PCA (RepE-inspired)", because
`lat` read as "this IS RepE's LAT". The *raw value* stays `"lat"`, and the
sidecar `recipeMethod` value stays `"repeLAT"`, permanently: they are written
into every sidecar, every frozen manifest concept block, and every
recipe-identity hash this workspace has produced. Changing them would break
decode of existing artifacts AND move identity hashes — which is how a frozen
experiment loses its promotions. Decode accepts them forever. The raw values
are artifact-compatibility constants; the symbols and labels carry the honesty.

**Schema-1 reader artifacts still load.** Every field added in schema 2 decodes
with an explicit legacy default: `contrastMode` absent = `supervisedContent`,
`signConvention` absent = `trainMajority`, `extractionRendering` absent = raw,
`orientationSeed` absent = none, `recommendedLayer` absent = none.

**`pc1ExplainedVariance` is not rewritten.** A schema-1 artifact's number was
computed over the ± ALTERNATED rows PCA is fitted on, where the cloud is
centred near zero by construction and the ratio is systematically flattering. A
reader reasonably assumes it describes the differences themselves, so schema 2
computes it that way and writes it under a NEW key,
`pc1ExplainedVarianceOfDifferences`, with `pc1ExplainedVarianceBasis` saying
which of the two a given artifact carries. The old key is decoded (basis
`"alternatedRows"`) and never written, because writing the old key with the new
semantics would make every pre-existing consumer silently wrong instead of
visibly out of date.

**Family labels in the cross-family report.** `lat` and `repeReaderLAT` used to
share the label `"RepE/LAT"`, putting a RepE-shaped direction and a RepE
direction in one row of the paper's own side-by-side. They are now
`paired-difference-PCA` and `RepE-reader-LAT`.
