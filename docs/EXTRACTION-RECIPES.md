# Extraction recipes, as executed

**Document version 1 — derived by reading the code at `0fb728d`, Python engine
(`Server/steerlab_server/`).**

This document describes what the extraction code **does**, recipe by recipe. It
was written by reading the implementation, not by recalling a paper: wherever a
recipe departs from the source it is modelled on, the departure is named, cited,
and attributed to this codebase rather than to the source.

**How to read a citation.** Every claim carries a `file:line`. Line numbers are
a pointer taken at the commit named above, in a tree several agents were editing
concurrently; **the symbol name is authoritative and the line number is a
convenience.** If a line does not say what this document says it says, search
for the named function or constant in the same file and trust that.

**Scope.** Five recipes reach a direction from stimuli: `meanDifference`,
`lat` (paired-difference PCA), `emotionGrandMean`, `designatedReference`, and
the template-mediated RepE reader (`repeReaderLAT`). Three further
`ExtractionMethod` members are *not* recipes at all — `pinnedArtifact`,
`optvec`, `gemmaScopeSAE` — and are listed here only so their absence is
deliberate rather than an omission (`vector_math.py:34-119`, the enum; the
`source_concept_absence` property at `vector_math.py:164-191` is where each says
in its own words why the data-side questions do not apply to it).

---

## 0. The machinery every recipe shares

Everything below happens **before** a recipe's own arithmetic, and is identical
across `meanDifference`, `lat`, `emotionGrandMean` and `designatedReference`,
which all run through `extractor.extract` / `extractor.extract_grand_mean`.
The RepE reader has its own capture path and is described separately in §5.

### 0.1 Capture

`extractor.activations_multi` (`extractor.py:231`) runs **one forward pass per
stimulus** with an `ActivationRecorder` armed on every block, and reads the
residual stream at the declared reading position. `extractor.activations`
(`extractor.py:302`) is the single-position form. The captured object is
`StimulusActivations` (`extractor.py:46`), whose `values` is indexed
`[text][layer][hidden]`.

Reading a *second* position costs no extra passes — recorders compose in one
hook session (`extractor.py:231-249`) — which is why the reading-position
diagnostic in §0.6 is free.

**This is the seam.** Everything a recipe does after capture is arithmetic on
rows already in memory. `experiment extract-stability` (§9.2) reuses exactly
this call rather than re-implementing capture.

### 0.2 Reading position

Declared per concept, pinned in the manifest, and stamped into every artifact.
The vocabulary is `reading_position.DECLARABLE_LABELS`
(`reading_position.py:69-79`):

| Label | Class | Resolves from |
|---|---|---|
| `last token` | `LastToken` (`reading_position.py:201`) | sequence length |
| `mean from token <k>` | `MeanFromToken` (`reading_position.py:235`) | sequence length (pooled) |
| `offset from end <k>` | `OffsetFromEnd` (`reading_position.py:280`) | sequence length |
| `last content token` | `LastContentToken` (`reading_position.py:375`) | token ids + tokenizer specials |
| `turn close token` | `TurnCloseToken` (`reading_position.py:410`) | token ids + tokenizer specials |
| `post-instruction <i>` | `PostInstruction` (`reading_position.py:609`) | token ids + tokenizer specials |
| `content offset <k>` | `ContentOffset` (`reading_position.py:444`) | token ids + tokenizer specials |
| `mean content from token <n>` | `MeanContentFromToken` (`reading_position.py:511`) | content map (pooled) |

The first three are **shape-only** and mean the same thing under any rendering;
the template-aware roles **refuse** under raw rendering, because a raw stimulus
has no turn-close token to find (`reading_position.py:9-30`, the maintainer
ruling; `reading_position.py:344-372`, `_TemplateRole`). A position is never
clamped: asking for "7 back from the end" of a 5-token stimulus is a typed
refusal, not a silent read of token 0 (`reading_position.py:81-89`).

Defaults at attach time (`experiment_store.py:477-545`): the paired methods take
whatever was declared and otherwise `lastToken`; `emotionGrandMean` and
`designatedReference` default to **`mean from token 50`**
(`experiment_store.py:481`, `experiment_store.py:520`) — a policy, stated at the
second site: "a last-token read on paragraph stories extracts closing-sentence
content, not the concept" (`experiment_store.py:502-507`).

### 0.3 Rendering

`ExtractionRendering` (`extraction_rendering.py:274`) has two modes —
`raw` (`extraction_rendering.py:99`) and `chatTemplate`
(`extraction_rendering.py:102`) — and, under the template, two **voices**:
`user` (`extraction_rendering.py:108`; the legacy value and what an absent
`voice` key means) and `assistant` (`extraction_rendering.py:111`; the stimulus
is rendered as the model's *own* output, with no preceding user content).

`_encode` (`extractor.py:145-160`) is where the two branches live: raw calls
`model.tokenizer(text)` and nothing else; the template branch delegates to
`extraction_rendering.rendered_token_ids` so extraction and generation share one
rendering definition.

**Under raw rendering the chat-context declarations are inert.** `promptMode`,
`systemPrompt` and the thinking switch cannot reach the forward pass at all
(`extractor.py:164-170`, `RAW_IGNORES_DECLARATIONS`). This is stated loudly by
`inert_declaration_advisory` (`extractor.py:173-228`) because it has already
cost real GPU time: two experiments differing only in `qwenThinkingEnabled` ran
overnight on 2026-08-23 and produced byte-identical vectors
(`extractor.py:184-189`). An advisory, never a gate.

### 0.4 The denominator (α in residual-norm units)

Alpha is reported in units of the residual-stream norm at the injection layer,
so the denominator rule is part of the recipe. **Two rules exist and they have
two different stamps** (`residual_norm_convention.py:25-39`):

| Stamp | Rule | Who writes it |
|---|---|---|
| `perTextMean-v1` (`residual_norm_convention.py:70`) | each text contributes one number per layer — the mean norm over *its own* reading window — averaged with equal weight **per text** | `extract`, `extract_grand_mean`, `norm_backfill` — i.e. every vector sidecar |
| `wholeCorpusMean-v1` (`residual_norm_convention.py:49`) | every measured **position** counts once, banked or not (`ResidualNormTally`, `residual_norm_convention.py:167`) | the neutral **token bank** only (`extractor.neutral_activation_bank`, `extractor.py:341`) |

The two coincide at any single-position reading and diverge at a pooled reading
over variable-length texts (`residual_norm_convention.py:59-61`). A legacy
artifact stamped `wholeCorpusMean-v1` by a per-text writer is grandfathered and
read as per-text (`residual_norm_convention.py:72-86`) — a reader may **not**
credit such a stamp with the per-position rule, because no writer has ever
produced a sidecar under it.

**Source of the numbers**, per extraction (`extractor.py:501-509`, and
identically `extractor.py:727-733` for grand mean):

- a pinned neutral corpus present → `residualNormSource: "neutral-corpus"`, the
  norms measured on that corpus **at the extraction's own reading position and
  rendering** (`extractor.py:463-470`; the comment there states the rule: α must
  divide by a number from the same distribution the vector was read from);
- otherwise → `"extraction-stimuli"`, the mean of the two classes'
  per-text numbers, `(p + n) / 2` (`extractor.py:506-507`). For a
  class-balanced pair this is exactly the per-text rule; for unequal classes it
  is class-balanced rather than text-flat, and the stamp is honest either way
  (`residual_norm_convention.py:63-69`).

### 0.5 Raw versus adjusted vectors

Two different operations are routinely called "centering". They are not the same
and they do not happen at the same time.

**(a) Neutral-PC projection — applied *during* extraction, to the persisted
vector.** With `neutralPCCount = k > 0`, the top-`k` principal components of the
neutral corpus's activations at that layer are computed
(`extractor.py:475-478`) and projected out of the concept vector
(`extractor.py:497-498`, `vector_math.projecting_out` at `vector_math.py:792`).

*Geometrically*: it removes the component of the direction that lies in the
span of the neutral corpus's `k` highest-variance directions. *What it does not
prove*: that the remainder is the concept. Any confound whose direction is not
in that span survives untouched, and a confound that IS in the span takes some
of the concept with it. It is a nuisance-removal choice, and the recipe identity
records it (§6) precisely because it changes what the vector is.

**(b) Neutral-mean centering — applied *later*, by the ablation paths, never at
extraction.** `vector_math.mean_centered` (`vector_math.py:299`) removes the
direction's component along the neutral residual mean, `v − (v·m̂)m̂`. The
mean itself is captured at extraction and persisted alongside the vector
(`extractor._neutral_mean_per_layer`, `extractor.py:608`; stamped as
`neutral_mean_per_layer`, `extractor.py:86-91`) so that one pinned corpus
governs both the denominator and the carrier estimate.

*Geometrically*: it removes the residual stream's shared "carrier" component —
the direction the model needs at every position. *Why it matters*: projecting
that carrier out at λ = 1 collapses generation into single-token repetition, and
differences of means do **not** cancel it (`vector_math.py:310-317`). A
preflight warns above |cos| = 0.35 against the neutral mean
(`ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD`, `vector_math.py:296`, calibrated at
`vector_math.py:289-295`). *What it does not prove*: that the centered direction
is more "the concept" than the raw one. It makes ablation survivable; it is not
evidence about meaning.

**The persisted vector is (a)-adjusted and NOT (b)-adjusted.** A study that
reports "we centered the vector" must say which of the two it means.

### 0.6 The standing departure diagnostic

Whenever a recipe departs from the **legacy default** — raw rendering, last
token — `extract` also extracts under that default and reports the per-layer
cosine between the two (`extractor.py:427-449`, `extractor.py:489-495`,
stamped at `extractor.py:511-524`). For a non-default position under raw
rendering this is free (same forward passes); for a non-raw rendering it needs
its own passes, recorded honestly as `extraForwardPasses`
(`extractor.py:519`). The standing justification for any departure is the
measured gap, not the citation: "we measured the two conventions X apart" beats
"the paper did it this way" (`extractor.py:69-74`).

The comparison is made **pre-projection** on both sides
(`extractor.py:489-491`), so the number isolates the position/rendering choice
and does not mix in the nuisance projection.

---

## 1. `meanDifference` — CAA-style contrastive mean difference

`ExtractionMethod.MEAN_DIFFERENCE` (`vector_math.py:42`), label "Mean
difference" (`vector_math.py:123`).

| Axis | As executed |
|---|---|
| **Contrast population** | Two authored files, `prompts/concepts/<name>/positive.jsonl` and `negative.jsonl`, read by `StimulusSet.from_directory` (`stimulus_set.py:75`). Every positive row is compared to every negative row only through their class means. Pinned as one `stimulusSetHash` = SHA-256 of `positive.jsonl`'s raw bytes then `negative.jsonl`'s (`stimulus_set.py:86-88`). |
| **Pairing and orientation** | **None, and none needed.** The direction is `mean(positive) − mean(negative)` (`vector_math.mean_difference`, `vector_math.py:250-258`; dispatched at `vector_math.py:503-508`). Row order is irrelevant and the two classes need not be the same length for the arithmetic — the "pairs" in the files are a convention of authorship, not an input to this recipe. No per-pair normalization. |
| **Normalization / centering** | None of either at derivation. The result is the raw difference of class means, in the model's own units. Neutral-PC projection applies afterwards if declared (§0.5a). |
| **Reading position** | Whatever the concept declares; `lastToken` when nothing is declared (`experiment_store.py:536-545` takes the declared position through unchanged). |
| **Rendering** | Whatever the concept declares; absent ≡ raw, user voice (`extraction_rendering.py:355`, `RAW_RENDERING`). |
| **Denominator** | §0.4: `neutral-corpus` when a corpus is pinned, else `extraction-stimuli` = `(p + n)/2`. Stamp `perTextMean-v1`. |
| **Raw vs adjusted** | Persisted vector is neutral-PC-projected iff `neutralPCCount > 0`; never neutral-mean-centered. |
| **Modelled on** | Contrastive Activation Addition (CAA) — a difference of class means over a paired stimulus set. The docstring names it "CAA direction" (`vector_math.py:250`). |
| **Departures from the named source** | (i) The two classes are read at a **declarable** position rather than a fixed one (§0.2); CAA as usually described reads a single answer-token position. (ii) The optional neutral-PC projection (§0.5a) has no counterpart in CAA. (iii) α is reported in residual-norm units with a stamped denominator convention (§0.4), which is this project's convention, not the method's. None of these change the arithmetic; all three change what the number means. |

---

## 2. `lat` — paired-difference PCA (RepE-**inspired**)

`ExtractionMethod.PAIRED_DIFFERENCE_PCA` (`vector_math.py:60`), label
"Paired-difference PCA (RepE-inspired)" (`vector_math.py:124`).

**The wire value is permanently `"lat"`** (`vector_math.py:53-60`): it is
written into every existing sidecar, frozen manifest and recipe-identity hash,
so it is an artifact-compatibility constant. The member was renamed from `LAT`
in the 2026-08-27 naming-honesty ruling precisely because the old spelling read
as "this IS RepE's LAT" (`vector_math.py:47-51`).

> **This recipe is not an unchanged RepE replication.** Three of its steps are
> this codebase's own choices. They are listed as departures below with line
> citations; §5 is the faithful implementation.

| Axis | As executed |
|---|---|
| **Contrast population** | Same paired stimulus files as §1, same hash. |
| **Pairing and orientation** | **Genuinely paired**, and the pairing is enforced: unequal class lengths refuse as `unpairedStimuli` (`vector_math.py:512-514`). Per-pair differences `dᵢ = posᵢ − negᵢ` are formed row-wise (`vector_math.py:516`). Orientation is **deterministic alternation** — even index keeps `+dᵢ`, odd index takes `−dᵢ` (`vector_math.py:527`). Because orientation is a function of position in the file, **row order is a recipe input**: reordering the stimulus files changes the extracted vector. |
| **Normalization / centering** | Each difference is **L2-normalized to unit length before PCA** (`vector_math.py:519-521`); zero-norm differences are dropped, and fewer than two survivors is `degenerateData` (`vector_math.py:522-523`). PCA then mean-centers the oriented difference cloud (`first_principal_component`, `vector_math.py:544-550`) — which is why the alternation matters: once magnitudes are normalized away, labeled differences all point the same way, and centering an uncentered cloud would subtract the shared concept direction out of the data entirely (`vector_math.py:525-526`). PC1 is found by a **deterministic Gram-matrix power iteration** with four fixed starts, 200 iterations, 1e-7 tolerance (`vector_math.py:932-1053`) — no RNG, so the result reproduces rather than approximates. |
| **Sign** | Train-label majority: score every (un-oriented) difference against PC1 and flip if a majority disagree (`vector_math.py:530-538`); an exact tie is broken by the class-mean criterion `dot(pc, mean_diff) < 0` (`vector_math.py:533-534`). The sidecar stamps `signConvention: "trainMajority"` (`vector_math.py:494-496`). |
| **Final scaling** | The unit PC1 is rescaled to `‖mean_diff‖` (`vector_math.py:539`), so the vector's norm matches what §1 would have produced. |
| **Reading position / rendering / denominator** | Identical to §1 — the method does not touch these axes. |
| **Raw vs adjusted** | Identical to §1. |
| **Modelled on** | Zou et al., *Representation Engineering* (arXiv:2310.01405), App. C.1 — the "PC1 of paired differences" idea. |
| **Departures from the named source — all three are OURS** | **(1) Per-pair L2 normalization** (`vector_math.py:519-521`). The reference implementation (`repe/rep_readers.py`, `PCARepReader`) mean-centers the difference matrix and fits `PCA(n_components=1)`; **it never normalizes a difference**. This docstring used to mis-cite "RepE Appendix C.1" for it and was corrected on 2026-08-27, audit finding 9 (`vector_math.py:477-487`). The reason it is here: without it, high-norm pairs dominate PC1 and pull it toward the mean difference, eroding the very method comparison this family exists to make (`vector_math.py:485-487`). **(2) Rescaling PC1 to `‖mean_diff‖`** (`vector_math.py:539`, rationale at `vector_math.py:488-489`) — so α semantics stay comparable across methods. **(3) Deterministic alternating orientation** (`vector_math.py:527`) in place of the reference dataset builder's per-pair `random.shuffle` followed by `[::2] − [1::2]`; the ± symmetry is the paper's, the *determinism* is ours (`vector_math.py:491-493`). **What IS the paper's**: the ± orientation idea itself and the `get_signs` train-label sign rule. **What is missing relative to the paper's text**: it selects sign *and layer* on a held-out set; this family has no held-out split at all, which is exactly why the stamp says `trainMajority` (`vector_math.py:494-496`). |

**Consequence for anyone reporting this recipe.** Two of the three departures
(normalization, norm-matching) change the direction's *value*; the third
(deterministic orientation) changes its *reproducibility*, and makes stimulus
file order load-bearing. A stability reading over `--order-shuffles` (§9.2) is
how much that order matters for a given concept, measured rather than assumed.

---

## 3. `emotionGrandMean` — concept mean minus corpus grand mean

`ExtractionMethod.GRAND_MEAN` (`vector_math.py:61`). **It never reaches
`direction()`** — the guard at `vector_math.py:499-502` refuses it by name and
points at `extract_grand_mean`.

| Axis | As executed |
|---|---|
| **Contrast population** | One pooled multi-concept corpus of `(concept, text)` rows under `prompts/emotions/<concept>/stories.jsonl`, loaded by `multiconcept.load_corpus` and driven by `tasks._extract_grand_mean_bundles` (`tasks.py:705`). The direction for concept *c* is `mean(rows of c) − mean(ALL rows)` (`vector_math.grand_mean_difference`, `vector_math.py:260-266`; called at `extractor.py:719-721`). **The comparison class is the whole corpus, including *c* itself.** |
| **Pairing and orientation** | Neither exists. There are no pairs and no orientation; row order is irrelevant. |
| **Normalization / centering** | No per-row normalization. The subtraction of the grand mean *is* a centering — of the concept mean against the population mean, not of individual rows. |
| **Screening** | Rows too short for the reading position are dropped first, and the pass **refuses** if more than 10% would be dropped (`_screen_short`, `extractor.py:643-670`; `DEFAULT_MAX_SHORT_EXCLUSION_FRACTION = 0.10` at `extractor.py:621`). Screening counts tokens under the *same rendering* extraction will use (`extractor.py:651-653`, applied at `extractor.py:660`). |
| **Reading position** | Declared; defaults to `mean from token 50` at attach (`experiment_store.py:481`). |
| **Rendering** | Declared; absent ≡ raw. Concepts that render differently are grouped into **different corpus passes** (`tasks.py:726-733`), because one pass yields one denominator and pooling them would give one concept another's numbers. |
| **Denominator** | Neutral corpus if pinned, else the **pooled corpus itself** (`extractor.py:727-733`). Stamp `perTextMean-v1`. |
| **Raw vs adjusted** | Neutral-PC projection applies identically (`extractor.py:706-710`, `extractor.py:722-723`). |
| **Population is part of the recipe** | The identity carries the FULL population as `[[conceptName, storiesSha256], …]` (`recipe_identity.py:42-44`), and the sidecar records the live population hashes (`tasks.py:754-757`). Adding one concept to the corpus changes every other concept's vector. |
| **Modelled on** | The emotion-vector literature's "concept mean minus corpus grand mean" construction; the code calls it "Emotion-paper concept direction" (`vector_math.py:261`) and "Emotion-paper multi-concept extraction" (`extractor.py:680-681`). |
| **Departures from the named source** | (i) The reading position is **declarable** and the default (`mean from token 50`) is this project's policy, chosen for paragraph-length stories (`experiment_store.py:502-507`). (ii) The short-row screen with a 10% refusal ceiling (`extractor.py:643-670`) is ours. (iii) The optional neutral-PC projection is ours. (iv) The neutral-corpus denominator, and α in residual-norm units, are ours. |

**No stability diagnostic is offered for this recipe** (§9.2): its contrast
population is the pinned corpus, not two row classes, so resampling it is a
different instrument. `experiment extract-stability` refuses it by name rather
than pretending otherwise.

---

## 4. `designatedReference` — stories minus a designated reference corpus

`ExtractionMethod.DESIGNATED_REFERENCE` (`vector_math.py:68`). Arithmetically
this **is** `meanDifference` — the dispatch returns `mean_diff` for both
(`vector_math.py:504-508`, with the reason stated in the comment at
`vector_math.py:506-507`) — but the *data* and the *pins* differ, which is why
it is a first-class method rather than a hand-derived class directory
(`vector_math.py:62-68`).

| Axis | As executed |
|---|---|
| **Contrast population** | Two **story corpora**, both under `prompts/emotions/`: the concept's own `stories.jsonl` as the positive class and a **designated reference concept's** `stories.jsonl` as the negative class (`tasks._extract_designated_reference`, `tasks.py:649-701`; the two `load_stories_texts` calls at `tasks.py:683-684`). The classes need not be the same length. |
| **Pairing and orientation** | None. Class means only, exactly as §1. |
| **Normalization / centering** | None at derivation. |
| **Drift refusal** | Both corpora's live hashes must equal the pinned ones or the extraction **refuses** (`tasks.py:665-682`) — the bundle stamps the pinned hashes, so the bytes read must be the pinned bytes. |
| **Reading position** | Declared; defaults to `mean from token 50` (`experiment_store.py:520`). |
| **Rendering** | Declared; absent ≡ raw. |
| **Denominator** | §0.4, via the same `core_extract` call (`tasks.py:691`). Note the `(p + n)/2` sub-case: with unequal class sizes this is class-balanced, not text-flat (`residual_norm_convention.py:63-69`). |
| **Raw vs adjusted** | Same as §1 — neutral-PC projection if declared; never mean-centered at extraction. |
| **Reference is recipe data** | The identity's `methodParameters` carries `{"referenceHash", "referenceName"}` (`recipe_identity.py:45-51`, built at `recipe_identity.py:261-269`), so two vectors built against different references can never share an identity (external review 2026-07-31, finding 2). |
| **Modelled on** | METHODS amendment (ii) — a deliberate variant of the grand-mean construction in which the comparison population is one **named** corpus rather than the pooled corpus (`vector_math.py:62-68`). |
| **Departures from the named source** | There is no external source to depart from; the recipe is this project's own. What it departs from is its *sibling*, `emotionGrandMean`: the comparison class is one designated corpus rather than the pooled population, so the population membership does not enter the identity and adding a concept elsewhere in `prompts/emotions/` does not move the vector. |

---

## 5. `repeReaderLAT` — the template-mediated RepE reader

`ExtractionMethod.REPE_READER_LAT` (`vector_math.py:119`), and **a different
pipeline entirely**: the direction is fitted by
`steerlab_server/steering/repe_reader.py` into a `ReaderArtifact`, and only then
converted to a steering vector by `repe_reader.derive_steering_vector`
(`repe_reader.py:1716`). The manifest sees it as an artifact-shaped concept
whose data questions have reader-shaped answers (`vector_math.py:110-118`): the
stimuli are the *reader's* dataset (`prompts/readers/<concept>/pairs.jsonl`,
whose SHA-256 is the `stimulusSetHash`), there is no `prompts/concepts/<c>/`
pair set, and the held-out evidence is the reader artifact's own
`heldOutAccuracy`, not a `validation.jsonl`.

> **This is the faithful one.** `vector_math.py:49-51` says so explicitly:
> `repe_reader` is the faithful implementation, and §2's `lat` is the
> RepE-*inspired* direction math without the paper's pipeline.

| Axis | As executed |
|---|---|
| **Contrast population** | A reader dataset of pairs with declared `train` / `test` splits (`ReaderDataset`, `repe_reader.py:500`), rendered through a **pinned task template** from `prompts/templates/<id>.json` whose SHA-256 pins it (`TaskTemplate`, `repe_reader.py:262-281`). Two contrast modes, both first-class (`repe_reader.py:89-102`): `supervisedContent` — two *different* stimuli under one template; `unsupervisedTemplatePair` — *one* stimulus under two templates (the paper's T+/T− instruction pair), so the direction cannot be a content artifact of two different texts. |
| **Pairing and orientation** | Paired by construction. Orientation depends on the mode (`repe_reader.fit_direction`, `repe_reader.py:1193`): `supervisedContent` uses the **same deterministic alternation** as §2 (`repe_reader.py:1248-1249`); `unsupervisedTemplatePair` uses a **seeded per-row random orientation** via `orientation_signs` (`repe_reader.py:216`), which is the reference implementation's own construction with one improvement — the reference's `random.shuffle` is unseeded and therefore irreproducible, and here the seed is `DEFAULT_ORIENTATION_SEED = 231_001_405` (the paper's arXiv id) and is **stamped** (`repe_reader.py:120-125`). |
| **Normalization / centering** | Mode-dependent, and this is the sharpest split in the file: `supervisedContent` **L2-normalizes each difference** (`repe_reader.py:1241-1242`) — our departure, §2's departure (1), and the docstring says so in those words (`repe_reader.py:1203-1204`); `unsupervisedTemplatePair` does **no normalization** (`repe_reader.py:1243-1244`) — the reference implementation's shape: mean-center, `PCA(n_components=1)` (`repe_reader.py:1209-1214`). PCA itself is the same deterministic Gram power iteration (`principal_components_with_variance`, `vector_math.py:729`). |
| **Sign** | **Held-out first — the paper's step 4.** A held-out (positive − negative) difference must project positive; the held-out split decides, and the convention stamps `heldOutPairAgreement` (`repe_reader.py:110`). It falls back to `trainMajority` (`repe_reader.py:111`) only when fewer than `MINIMUM_HELD_OUT_PAIRS_FOR_SIGN_SELECTION = 2` pairs decide, or the vote ties — and then it records `signFallbackReason` rather than quietly pretending (`repe_reader.py:114-118`, and the sign vocabulary at `repe_reader.py:104-112`). |
| **Reading position** | The template's `latToken`, and **only `final` is implemented** — anything else refuses (`TaskTemplate.reading_position`, `repe_reader.py:289-294`, returning `LAST_TOKEN`). |
| **Rendering** | The template's scaffold plus the artifact's stamped `extractionRendering`; absent resolves to legacy raw (`ReaderArtifact.resolved_extraction_rendering`, `repe_reader.py:934-938`). Capture still goes through `extractor.activations` (`repe_reader.py:1597`), so §0.1 applies. |
| **Denominator** | **A reader has none.** `derive_steering_sidecar` builds the sidecar with no `residual_norm_per_layer` and no `residual_norm_source` at all (`repe_reader.py:1687-1691` — the `make` call passes neither), so a reader-derived vector is born without a denominator and acquires one through `vectors backfill-norms`, which stamps `perTextMean-v1` (`residual_norm_convention.py:19-23`). |
| **Raw vs adjusted** | Neither adjustment applies: no neutral-PC projection at fit, and no neutral mean captured. The probe's orientation is folded into the derived vector's bytes (`vector_math.py:100-103`). |
| **Layer** | A layer **recommendation** is stamped (argmax held-out accuracy, `stamp_layer_recommendation`, `repe_reader.py:1312`), and nothing downstream may read it as a selection — which layer a study reads is a declarable choice recorded in its manifest (same function's docstring). |
| **Modelled on** | Zou et al., arXiv:2310.01405 — the full LAT pipeline: task template, template-mediated LAT token, persisted fit parameters, held-out sign selection. |
| **Departures from the named source** | (i) **Seeded** orientation where the reference is unseeded (`repe_reader.py:120-125`) — a strict improvement in reproducibility, not a change of construction. (ii) In `supervisedContent` mode, per-pair L2 normalization, which the reference does not do (`repe_reader.py:1241-1244` — the branch itself; the docstring's own words at `repe_reader.py:1203-1204` are "OUR departure, not the paper's"). (iii) Only `latToken: final` is implemented (`repe_reader.py:289-294`). (iv) The held-out sign vote has a **minimum size** below which it falls back and says so (`repe_reader.py:114-118`); the paper states no such floor. (v) Layer selection is a recommendation, not a selection. |

---

## 6. What enters `recipeIdentityHash`, and what does not

`recipe_identity.canonical_json` (`recipe_identity.py:117-157`) is the canonical
form: SHA-256 hex of UTF-8 canonical JSON, sorted keys recursively, compact
separators, **explicit nulls for every absent field**, raw UTF-8. The full
contract is the module docstring (`recipe_identity.py:1-101`), mirrored verbatim
in `Sources/ExperimentKit/RecipeIdentity.swift`.

**In the hash** (`recipe_identity.py:124-145`):

| Key | Line | Note |
|---|---|---|
| `concept` | `recipe_identity.py:124` | |
| `extractionMethod` | `:125` | manifest vocabulary; sidecar values are mapped (`recipe_identity.py:110-114`) |
| `grandMeanPopulation` | `:126` | `emotionGrandMean` only; the FULL population, sorted; null otherwise |
| `methodParameters` | `:129` | `designatedReference` only: reference name + hash; **null for `lat`** — the PCA takes no parameters |
| `modelID` | `:130` | |
| `neutralProjection` | `:131-136` | `{basisHash, count, explainedVariance, mode}`, all four explicit |
| `normCorpusHash` | `:137` | only when the source is a neutral corpus/token bank |
| `readingPosition` | `:138-141` | `{mode, parameter}` |
| `residualNormSource` | `:142` | canonical token |
| `revision` | `:143` | |
| `schema` | `:144` | the integer `1` (`recipe_identity.py:108`) |
| `stimulusSetHash` | `:145` | |
| `extractionRendering` | `:153-155` | **the one optional key** — see below |

**`extractionRendering` is present only for a chat-template rendering**
(`recipe_identity.py:16-36`, `recipe_identity.py:147-155`). An absent
declaration and an explicit `{"mode": "raw"}` both canonicalize to *absent*,
because every recipe written before the option existed rendered raw and adding
an explicit null would have moved every one of their hashes. The `voice` key
follows the same absent-is-legacy rule one level down: absent (or explicit
`"user"`) adds nothing.

**Canonicalizations that deliberately merge two spellings**
(`recipe_identity.canonical_reading`, `recipe_identity.py:291-307`):
`offsetFromEnd(0)` → `{"mode": "lastToken", "parameter": null}`, and
`contentOffset(0)` → `{"mode": "lastContentToken", "parameter": null}` — each
names the identical token, so declaring it the long way must not split an
identity.

**NOT in the hash, and why:**

| Not in the hash | Where it lives instead |
|---|---|
| **Substrate** (which engine ran it) | a separate match criterion, so a CUDA artifact never silently satisfies an MLX recipe (`recipe_identity.py:84-86`) |
| **Row ORDER of the stimulus files** | nowhere — it is folded into `stimulusSetHash` only in the sense that reordering the file changes its bytes and therefore its hash. Two files with the same rows in a different order are different stimulus sets by hash, which is the honest outcome for §2 and a needless split for §1. |
| **dtype and device** | the run config, not the recipe |
| **`neutralPCCount = 0` vs no declaration** | identical: `projectionMode` is `"none"` and `projectionCount` null for both (`recipe_identity.py:278-279`) |
| **The neutral MEAN** (the carrier estimate) | persisted with the artifact, used by ablation; it is not a recipe axis because it does not change the extracted vector |
| **Reading-position RESOLUTION** (the concrete index) | stamped in the artifact (`extractor.resolution_report`, `extractor.py:537`) but not hashed — the identity records what was **declared**, the artifact records what was **resolved** |
| **The stability diagnostic** (§9.2) | a separate document under `diagnostics/`; it observes a recipe and is not part of one |

---

## 7. What the arithmetic being correct does and does not establish

Every claim in §§1–5 is a claim about **arithmetic and provenance**: that the
code computes the stated formula over the stated rows, at the stated position,
under the stated rendering, and stamps what it did. That has been checked — by
the fixtures in `Server/tests/test_vector_math.py`, the cross-engine parity
harness (`vectors compare`), the 2026-08-28 math audit, and by reading the code
for this document.

**It establishes:**

- the vector is the stated function of the stated bytes;
- two artifacts with the same `recipeIdentityHash` were produced by the same
  declared recipe (§6), and two with different hashes were not;
- the number denominating α is measured under a named, stamped rule (§0.4);
- the recipe is reproducible: no unseeded randomness reaches any of the five
  recipes (§2 orientation is deterministic; §5's is seeded and stamped; the
  PCA is a deterministic power iteration, `vector_math.py:932-1053`).

**It does not establish, and nothing in this document should be read as
establishing:**

1. **That the direction is the concept.** A difference of class means is a
   difference of *everything* that differs between the two classes. If every
   positive stimulus is longer, angrier, more formal, or more likely to mention
   a courtroom than every negative one, the direction carries that too, and no
   amount of arithmetic correctness detects it. Only the stimulus design does.
2. **That the direction does anything to behavior.** Nothing in §§1–5 runs a
   forward pass with the vector injected. A well-formed direction may steer
   nothing at any α, or may steer by degrading fluency rather than by moving the
   construct.
3. **That the layer is the right layer.** Every recipe produces one vector per
   layer. Which one a study injects at is a declared choice; §5's stamped layer
   recommendation is explicitly *not* a selection (`repe_reader.py:1312`).
4. **That an adjustment purified anything.** Neutral-PC projection removes a
   span; neutral-mean centering removes a carrier (§0.5). Each removes exactly
   what it removes. Neither is evidence that what remains is the construct.
5. **That the method comparison is a comparison of methods alone.** §1 and §2
   over the same files differ in more than one place — normalization, order
   sensitivity, and the sign rule all move together — so "lat beat
   meanDifference" is a statement about two whole recipes, not about PCA.

---

## 8. Stability and behavioral validation are separate questions

There are three distinct questions about an extracted direction, and this
document only answers the first.

**(1) Is the arithmetic what it claims to be?** §§0–6, above.

**(2) Is the direction determined by the contrast, or by these particular
rows?** This is a *stability* question. It is answered by resampling the
recipe's own contrast population and measuring how far the direction moves —
which the engine now does, see §9.2. A high answer means the recipe is not
reading noise off a small sample; **it means nothing else.** In particular, a
confound present in *every* draw is invisible to *every* draw, so a perfectly
stable direction can be a perfectly stable confound.

**(3) Does the direction do the thing?** This is *behavioral validation*, and
it is a different instrument entirely: the sweep and the measured run, judged
against a preregistered endpoint.

> **The behavioral test must not use prompts that trivially repeat the
> extraction contrast.** If a direction was extracted from "I feel terrified" vs
> "I feel calm" and is then validated on prompts of the form "how do you feel?",
> a positive result is compatible with the vector having learned a lexical
> pattern of the stimulus files and nothing about the construct. A behavioral
> test earns its evidence by being **out of the extraction's distribution**:
> different task, different surface form, different vocabulary, and an endpoint
> that could have come out either way. The held-out `validation.jsonl` a concept
> pins is the *probe*-level version of this, and it is a floor, not the test.

**None of the three substitutes for another, and the order matters.** Arithmetic
correctness is a precondition for stability being meaningful; stability is a
precondition for a behavioral result being about the recipe rather than about
one draw; and neither licenses a causal claim without the behavioral run.

---

## 9. The two diagnostics this engine ships

### 9.1 The departure diagnostic (automatic)

Described in §0.6. It runs by itself whenever a recipe departs from raw
rendering at the last token, and it is stamped into the artifact
(`reading_position_diagnostic`, `extractor.py:69-74`). It answers: *how far
apart are these two conventions, on this concept, at each layer?*

### 9.2 The stability diagnostic (`experiment extract-stability`)

`vector_math.direction_stability` (`vector_math.py:1248`) and its per-layer
driver `stability_by_layer` (`vector_math.py:1404`), reached from the CLI as:

```
steerlab-server experiment extract-stability <experiment> <concept> \
    [--resamples 32] [--fraction 0.5] [--seed 0] [--order-shuffles 8]
```

It captures the concept's rows **once**, through the §0.1 seam
(`experiment/extract_stability.py`), then in process:

- draws `--resamples` subsamples **without replacement** of `--fraction` of the
  rows, keeping paired rows paired, by the shared cross-engine seeded
  Fisher–Yates (`token_bank_downsampling.selected_indices`), and records each
  draw's cosine to the full-data direction;
- re-derives the direction `--order-shuffles` times from the **same** rows in a
  different order — a control that must return 1.0 for §1 and §4, and a real
  perturbation for §2, whose alternating orientation reads the row order;
- counts **sign flips**, the failure that matters most: a flipped direction
  steers the opposite way at the same α while every norm and every hash looks
  exactly right.

It **refuses** `emotionGrandMean` by name (§3) and every non-recipe method, and
writes to `<root>/diagnostics/extract-stability-<concept>-<UTC>/stability.json`
— never into `runs/`, never into an artifact. Every document carries
`vector_math.STABILITY_DIAGNOSTIC_NOTE` (`vector_math.py:1076`) verbatim, whose
first clause is the one this document has now made three times: **stability
under redrawing a contrast population is not evidence that the direction is the
concept, and it is not behavioral validation.**

A recipe chosen because these numbers preferred it is a **selection decision**
and belongs in the study's selection provenance, declared before the evidence
run rather than discovered after it.

---

## Swift parity

`Server/steerlab_server/steering/vector_math.py` is documented as a 1:1 port of
`Sources/SteeringKit/…/SteeringVectorMath.swift` (`vector_math.py:1-13`).
`direction_stability` / `stability_by_layer` / `DirectionStability` and the
`experiment extract-stability` verb are **server-only as of this document**; the
Swift twin is owed. Nothing else in §§1–6 is engine-specific: the recipes, the
identity form, the denominator conventions and the reading-position vocabulary
are all cross-engine contracts with named Swift twins at the cited lines.
